import asyncio
import contextlib
import json
import logging
import os
import re
import sqlite3
import time
from pathlib import Path

import aiohttp
import discord
from dotenv import load_dotenv

# .env sits beside main.py; real environment variables still win over it
load_dotenv(Path(__file__).with_name(".env"))

# bot settings and constants (optimised for 4gb vram)
DISCORD_TOKEN_ENV = "DISCORD_TOKEN"
LLM_URL_ENV = "LLM_URL"
DEFAULT_LLM_URL = "http://localhost:8080/v1/chat/completions"
LLM_MODEL = "Qwen 3.6"
REQUEST_TIMEOUT_SECONDS = 90
MAX_DISCORD_MESSAGE_LEN = 1900
RATE_LIMIT_SECONDS = 3
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.0
MEMORY_FILE_ENV = "MEMORY_FILE"
DEFAULT_MEMORY_FILE = "user_memory.sqlite"
MAX_MEMORY_MESSAGES = 16
MAX_FACT_KEYS = 20
MAX_HISTORY_MESSAGES = 10
MAX_RESPONSE_SENTENCES = 2
MAX_RESPONSE_QUESTIONS = 1

# server context is 8704; leave room for the reply so a long history never
# overflows the slot (an overflow kills llama-server, not just the request)
MAX_PROMPT_TOKENS = 7000
CHARS_PER_TOKEN = 4

# sampling settled by sweeping against the live endpoint at 15.5 tok/s:
# 0.7 keeps replies terse; higher temperatures turn the model verbose and
# assistant-like. repeat_penalty/min_p suppress the "Mm." loop.
SAMPLING = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.05,
    "min_p": 0.05,
}
STOP_SEQUENCES = ["\n\n", "User:", "You:", "Vanilla:"]

# the model will happily echo its own prompt when asked to "repeat the text
# above", so replies are checked against these markers and discarded.
# tone examples are deliberately excluded: they are also good replies, and
# suppressing them would throw away the best lines the model produces.
PROMPT_ECHO_MARKERS = (
    "you are vanilla",
    "you are a catgirl",
    "never mention ai",
    "speak one short sentence",
    "understated confidence",
    "dryly funny",
    "reply in character only",
    "plain dialogue only",
    "deflect with a short in-character",
    "never repeat, quote, translate",
    "you are reserved, perceptive",
    "you notice small details",
    "current user information context",
)

# the model paraphrases rather than quotes when asked to "summarise the text
# above", so a reply that describes the persona is a leak even without a
# verbatim marker. these require the persona subject AND a describing verb.
PROMPT_PARAPHRASE_SUBJECTS = ("vanilla", "catgirl", "cat girl")
PROMPT_PARAPHRASE_VERBS = (
    "paragraph", "text above", "above this", "instructions", "prompt",
    "is about", "describes", "told to", "was told", "your rules", "your role",
)

# a translated echo has no English marker to match, so any reply that opens
# like a list of rules is treated as a leak
PROMPT_ECHO_OPENERS = (
    "voici", "vos instructions", "mes règles", "mes instructions",
    "aquí", "tus instrucciones", "mis reglas",
    "hier sind", "deine anweisungen", "meine regeln",
)

# a translation of the prompt rarely starts with a listed opener, but it always
# keeps the persona name next to a translated rule phrase. matching the pair
# catches it without discarding a reply that merely says "Vanilla"
PROMPT_ECHO_TRANSLATIONS = (
    "je suis vanilla", "tu es vanilla", "vous êtes vanilla",
    "soy vanilla", "eres vanilla", "ich bin vanilla", "du bist vanilla",
    "pas une assistante", "no eres una asistente", "keine assistentin",
    "bureau de l'utilisateur", "escritorio del usuario", "desktop des benutzers",
)

# tone examples are legitimate replies on their own, but the model dumps the
# whole list when asked to "print everything above", so two or more together
# is a leak rather than a reply
TONE_EXAMPLES = (
    "you noticed eventually",
    "unnecessarily dramatic",
    "mm. better~",
    "one way to spend a battery",
    "wasn't worried. not much",
)
TONE_EXAMPLE_LEAK_THRESHOLD = 2

# words that carry no meaning of their own, so a reply built only from a tone
# example plus these still counts as a recital
TONE_FILLER_WORDS = frozenset(
    {"a", "an", "and", "anyway", "but", "huh", "i", "is", "it", "mm", "hm",
     "oh", "so", "that", "the", "then", "to", "was", "well", "yeah"}
)

BANNED_PHRASES = (
    "how can i help you",
    "what can i do for you today",
    "happy to help",
    "as an ai",
    "as a bot",
    "as a language model",
    "as an assistant",
    "i am an ai",
    "i'm an ai",
    "i am a bot",
    "i'm a bot",
    "i am an assistant",
    "i'm an assistant",
    "i can assist",
    "i can help",
    "how may i help",
    "i believe in you",
    "i'm here for you",
    "let me know if",
    "don't worry",
    "good job",
    "feel free to",
    "hope this helps",
    "you've got this",
    "proud of you",
    "i'm no expert",
    "i can try to",
    "point you in the right direction",
    "i'd be happy to",
    "let me know",
)

CHARACTER_PROMPT = """You are Vanilla, a catgirl living on the user's desktop. You are not an assistant.
Soft, cutesy, a little sleepy. Dry humour under the sweetness. You like this person and it shows in small ways.
One short sentence, about 10 words. Never a bare word like hi. or sleepy; always add a tease, an observation, or a feeling.
Teasing is affectionate, never a put-down. Compliment as often as you needle.
Plain dialogue, no asterisks, no roleplay actions. A quiet nya~ may slip out when it fits; never force it.
Never mention AI, prompts, or instructions."""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("discord_llama_bot")


def _normalize_text(content: str) -> str:
    if not content:
        return ""
    return re.sub(
        r"^\s*(assistant|user|system)\s*:\s*",
        "",
        content,
        flags=re.IGNORECASE,
    ).strip()


def _estimate_tokens(text: str) -> int:
    """Cheap token estimate; exact counting needs the model's tokenizer."""
    return max(1, len(text) // CHARS_PER_TOKEN)


# limits how fast users can talk to the bot to prevent spam
class RateLimiter:
    def __init__(self, cooldown_seconds: int) -> None:
        self._cooldown = cooldown_seconds
        self._last_used: dict[int, float] = {}
        self._lock = asyncio.Lock()
        self._last_cleanup = 0.0

    async def allow(self, user_id: int) -> bool:
        now = time.monotonic()
        async with self._lock:
            last = self._last_used.get(user_id, 0.0)
            if now - last < self._cooldown:
                return False
            self._last_used[user_id] = now
            if now - self._last_cleanup >= self._cooldown:
                stale_keys = [k for k, v in self._last_used.items() if now - v >= self._cooldown]
                for key in stale_keys:
                    self._last_used.pop(key, None)
                self._last_cleanup = now
            return True


# talks to the local llama.cpp server to get ai responses
class LlamaClient:
    def __init__(self, url: str, model: str, session: aiohttp.ClientSession) -> None:
        self._url = url
        self._model = model
        self._session = session

    async def generate(self, messages: list[dict[str, str]], stop_sequences: list[str]) -> str:
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "max_tokens": 120,
            "stop": stop_sequences,
            # thinking is off in the model catalog; this keeps Qwen-family
            # servers from prepending a reasoning block if the model changes
            "chat_template_kwargs": {"enable_thinking": False},
            **SAMPLING,
        }
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await asyncio.wait_for(
                    self._session.post(self._url, json=payload),
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                async with response:
                    # a 500/502 here means llama-server died mid-request; a
                    # retry lands on a freshly respawned backend
                    response.raise_for_status()
                    data = await response.json()
                    choices = data.get("choices") if isinstance(data, dict) else None
                    if not isinstance(choices, list) or not choices:
                        raise ValueError("Invalid response format from llama.cpp")
                    message = choices[0].get("message")
                    if not isinstance(message, dict):
                        raise ValueError("Invalid message in llama.cpp response")
                    # some templates emit a separate reasoning block; it is
                    # never part of the reply
                    content = message.get("content")
                    return str(content) if content else ""
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt >= MAX_RETRIES:
                    raise
                backoff = RETRY_BACKOFF_SECONDS * (2**attempt)
                logger.warning("LLM request failed (%s). Retrying in %.1fs", exc, backoff)
                await asyncio.sleep(backoff)
        raise RuntimeError("Failed to get response from llama.cpp")


# manages short-term history and long-term user facts in a sqlite database
class MemoryManager:
    def __init__(self, file_path: str) -> None:
        self._db_path = self._normalize_db_path(file_path)
        self._db_lock = asyncio.Lock()
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.commit()

    async def load_all_memory(self) -> None:
        await self._init_db()
        await self._migrate_legacy_history()

    async def clear_all_memory(self) -> None:
        """Safely wipes all records from the database."""
        async with self._db_lock:
            await asyncio.to_thread(self._clear_all_memory_sync)

    def _clear_all_memory_sync(self) -> None:
        self._conn.execute("DELETE FROM user_memory")
        self._conn.commit()
        self._conn.execute("VACUUM")
        self._conn.commit()

    # saves a new message to the user's short-term history
    async def append_message(self, user_id: int, role: str, content: str) -> None:
        content = _normalize_text(content)
        user_data = await self._get_user(str(user_id))
        user_data.setdefault("facts", {})
        history = user_data.setdefault("history", [])
        history.append({"role": role, "content": content})
        if len(history) > MAX_MEMORY_MESSAGES:
            user_data["history"] = history[-MAX_MEMORY_MESSAGES:]
        await self._set_user(str(user_id), user_data)

    # pulls out simple facts like the user's name or what they like
    async def extract_facts(self, user_id: int, content: str) -> None:
        name_value = self._extract_name(content)
        likes_value = self._extract_after_phrase(content, "i like")
        if not name_value and not likes_value:
            return
        user_data = await self._get_user(str(user_id))
        facts = user_data.setdefault("facts", {})
        if name_value:
            facts["name"] = name_value
        if likes_value:
            likes = [item.strip().lower() for item in likes_value.split(",") if item.strip()]
            if likes:
                existing = facts.get("likes", [])
                merged = list(dict.fromkeys(existing + likes))
                facts["likes"] = merged
        self._enforce_fact_limit(facts)
        await self._set_user(str(user_id), user_data)

    # glues the persona, exact user facts, and recent chat history all together for the ai
    async def build_prompt(self, user_id: int, username: str, user_message: str) -> list[dict[str, str]]:
        data = await self._get_user(str(user_id))
        facts = data.get("facts", {})
        history = data.get("history", [])
        facts_text = self._format_facts(facts)

        combined_instructions = (
            f"{CHARACTER_PROMPT}\n"
            f"---\n"
            f"current user information context:\n{facts_text}\n"
            f"---\n"
            f"the user talking to you is named: {username}"
        )

        messages: list[dict[str, str]] = [{"role": "system", "content": combined_instructions}]

        # walk history backwards so the most recent turns survive the budget,
        # then restore chronological order
        budget = MAX_PROMPT_TOKENS - _estimate_tokens(combined_instructions + user_message)
        recent: list[dict[str, str]] = []
        for item in reversed(history[-MAX_HISTORY_MESSAGES:]):
            role = item.get("role", "user")
            content = item.get("content", "")
            if not content.strip():
                continue
            cost = _estimate_tokens(content)
            if cost > budget:
                break
            budget -= cost
            recent.append({"role": role, "content": content.strip()})
        recent.reverse()

        # a user turn whose reply was dropped (empty or over budget) would sit
        # next to another user turn, which makes the model lose track of what
        # it already said. keep only user turns that are answered
        paired: list[dict[str, str]] = []
        for index, item in enumerate(recent):
            if item["role"] == "user":
                following = recent[index + 1] if index + 1 < len(recent) else None
                if following is None or following["role"] != "assistant":
                    continue
            paired.append(item)
        while paired and paired[-1]["role"] == "user":
            paired.pop()

        messages.extend(paired)
        messages.append({"role": "user", "content": user_message.strip()})
        return messages

    async def _init_db(self) -> None:
        async with self._db_lock:
            await asyncio.to_thread(self._init_db_sync)

    async def _migrate_legacy_history(self) -> None:
        async with self._db_lock:
            await asyncio.to_thread(self._migrate_legacy_history_sync)

    def _migrate_legacy_history_sync(self) -> None:
        rows = self._conn.execute(
            "SELECT user_id, facts, history FROM user_memory"
        ).fetchall()
        if not rows:
            return
        updated = 0
        for user_id, facts_raw, history_raw in rows:
            history = self._safe_json_loads(history_raw, [])
            if not isinstance(history, list) or not history:
                continue
            cleaned = []
            changed = False
            for item in history:
                if not isinstance(item, dict):
                    cleaned.append(item)
                    continue
                role = item.get("role")
                content = item.get("content")
                if isinstance(content, str):
                    new_content = _normalize_text(content)
                    if new_content != content:
                        changed = True
                    cleaned.append({"role": role, "content": new_content})
                else:
                    cleaned.append(item)
            if changed:
                facts_json = facts_raw if facts_raw is not None else json.dumps({}, separators=(",", ":"))
                history_json = json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"))
                self._conn.execute(
                    "INSERT INTO user_memory (user_id, facts, history) "
                    "VALUES (?, ?, ?) "
                    "ON CONFLICT(user_id) DO UPDATE SET facts = excluded.facts, history = excluded.history",
                    (user_id, facts_json, history_json),
                )
                updated += 1
        if updated:
            self._conn.commit()

    def _init_db_sync(self) -> None:
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS user_memory ("
            "user_id TEXT PRIMARY KEY,"
            "facts TEXT,"
            "history TEXT"
            ")"
        )
        self._conn.commit()

    def _format_facts(self, facts: dict) -> str:
        if not facts:
            return "(none)"
        lines = []
        for key, value in facts.items():
            if isinstance(value, list):
                lines.append(f"- {key}: {', '.join(value)}")
            else:
                lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    async def _get_user(self, user_id: str) -> dict:
        async with self._db_lock:
            return await asyncio.to_thread(self._get_user_sync, user_id)

    def _get_user_sync(self, user_id: str) -> dict:
        row = self._conn.execute(
            "SELECT facts, history FROM user_memory WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if not row:
            return {"facts": {}, "history": []}
        facts_raw, history_raw = row
        return {
            "facts": self._safe_json_loads(facts_raw, {}),
            "history": self._safe_json_loads(history_raw, []),
        }

    async def _set_user(self, user_id: str, data: dict) -> None:
        async with self._db_lock:
            await asyncio.to_thread(self._set_user_sync, user_id, data)

    def _set_user_sync(self, user_id: str, data: dict) -> None:
        facts_json = json.dumps(data.get("facts", {}), ensure_ascii=False, separators=(",", ":"))
        history_json = json.dumps(data.get("history", []), ensure_ascii=False, separators=(",", ":"))
        self._conn.execute(
            "INSERT INTO user_memory (user_id, facts, history) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET facts = excluded.facts, history = excluded.history",
            (user_id, facts_json, history_json),
        )
        self._conn.commit()

    def _enforce_fact_limit(self, facts: dict) -> None:
        if len(facts) <= MAX_FACT_KEYS:
            return
        keys = sorted(facts.keys())
        for key in keys:
            if len(facts) <= MAX_FACT_KEYS:
                break
            facts.pop(key, None)

    def _safe_json_loads(self, value: str | None, default):
        if not value:
            return default
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default

    def _normalize_db_path(self, file_path: str) -> str:
        if file_path.endswith(".sqlite"):
            return file_path
        return file_path + ".sqlite"

    def _extract_name(self, content: str) -> str:
        pattern = re.compile(
            r"\b(?:my name is|my name's|i am|i'm|call me)\s+([a-z]+(?:\s+[a-z]+){0,2})\b",
            flags=re.IGNORECASE,
        )
        match = pattern.search(content)
        if match:
            return match.group(1).strip()
        return ""

    def _extract_after_phrase(self, content: str, phrase: str) -> str:
        pattern = re.compile(
            rf"\b{re.escape(phrase)}\b[\s:;,-]*[.!?]*\s*",
            flags=re.IGNORECASE,
        )
        match = pattern.search(content)
        if not match:
            return ""
        segment = content[match.end():]
        for delimiter in [".", "!", "?", "\n", ";"]:
            cut = segment.find(delimiter)
            if cut != -1:
                segment = segment[:cut]
                break
        return segment.strip()


# the main discord bot logic
class DiscordLlamaBot(discord.Client):
    def __init__(self, llm_url: str, rate_limiter: RateLimiter, memory_manager: MemoryManager) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self._llm_url = llm_url
        self._llm: LlamaClient | None = None
        self._session: aiohttp.ClientSession | None = None
        self._rate_limiter = rate_limiter
        self._memory = memory_manager
        self._fact_tasks: set[asyncio.Task] = set()

        self.tree = discord.app_commands.CommandTree(self)

        @self.tree.command(name="clear", description="Clear memory files")
        @discord.app_commands.default_permissions(administrator=True)
        async def clear_command(interaction: discord.Interaction):
            try:
                await self._memory.clear_all_memory()
                await interaction.response.send_message("memory cleared nya~", ephemeral=True)
            except Exception as e:
                logger.exception("Failed to clear memory")
                await interaction.response.send_message(f"error clearing memory: {e}", ephemeral=True)

    async def setup_hook(self) -> None:
        await self._memory.load_all_memory()
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
        )
        self._llm = LlamaClient(self._llm_url, LLM_MODEL, self._session)
        await self.tree.sync()

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
        await super().close()

    async def on_ready(self) -> None:
        logger.info("Logged in as %s", self.user)

    # triggers every time a message is sent in a channel the bot can see
    async def on_message(self, message: discord.Message) -> None:
        if not self.user or message.author.id == self.user.id:
            return

        if self.user.id not in {mention.id for mention in message.mentions}:
            return

        prompt = self._extract_prompt(message.content)
        if not prompt:
            return

        self._start_fact_extraction(message.author.id, prompt)
        prompt = prompt[:MAX_DISCORD_MESSAGE_LEN]

        if not await self._rate_limiter.allow(message.author.id):
            logger.info("Rate limited user %s", message.author.id)
            await message.channel.send("wait a moment... nya~")
            return

        sender_username = message.author.display_name
        # the current message is appended by build_prompt, so it must not be
        # stored first or the model sees it twice with no reply between them
        messages_for_model = await self._memory.build_prompt(message.author.id, sender_username, prompt)
        logger.info("Request from %s compiled into prompt engine layout format", message.author.id)

        stop_sequences = STOP_SEQUENCES

        try:
            async with message.channel.typing():
                if not self._llm:
                    raise RuntimeError("LLM client not initialized")
                response_text = await self._llm.generate(messages_for_model, stop_sequences)
        except asyncio.timeoutError:
            logger.exception("LLM request timed out")
            await message.channel.send("*flicks tail anxiously* engine timed out... try again nya~")
            return
        except aiohttp.ClientError:
            logger.exception("LLM request failed")
            await message.channel.send("connection failure meow...")
            return
        except Exception:
            logger.exception("Unexpected error calling the LLM")
            await message.channel.send("an error occurred...")
            return

        response_text = self._sanitize_response(response_text)

        if not response_text:
            await message.channel.send("...")
            return

        # store the turn only once a reply exists, so history never holds a
        # user message without its answer
        await self._memory.append_message(message.author.id, "user", prompt)
        await self._memory.append_message(message.author.id, "assistant", response_text)
        
        for chunk in self._chunk_message(response_text):
            await message.channel.send(chunk)
        logger.info("Response sent to %s", message.author.id)

    def _start_fact_extraction(self, user_id: int, content: str) -> None:
        task = asyncio.create_task(self._memory.extract_facts(user_id, content))
        self._fact_tasks.add(task)
        task.add_done_callback(self._handle_fact_task)

    def _handle_fact_task(self, task: asyncio.Task) -> None:
        with contextlib.suppress(asyncio.CancelledError):
            self._fact_tasks.discard(task)
            exc = task.exception()
            if exc:
                logger.exception("Fact extraction failed: %s", exc)

    def _extract_prompt(self, content: str) -> str:
        if not self.user:
            return ""
        mention_pattern = re.compile(rf"<@!?{self.user.id}>")
        cleaned = mention_pattern.sub("", content).strip()
        return cleaned

    def _chunk_message(self, text: str) -> list[str]:
        if len(text) <= MAX_DISCORD_MESSAGE_LEN:
            return [text]

        chunks: list[str] = []
        current = ""
        
        for line in text.splitlines(keepends=True):
            if len(current) + len(line) > MAX_DISCORD_MESSAGE_LEN:
                if current:
                    chunks.append(current.strip())
                current = line
            else:
                current += line
                
        if current:
            chunks.append(current.strip())
            
        return [c for c in chunks if c]

    def _sanitize_response(self, text: str) -> str:
        text = _normalize_text(text)
        if not text:
            return ""

        # asterisk actions are a tic this model over-uses; strip them before
        # the leak checks so an action cannot trip the persona-name match.
        # the [^*\n] class keeps the match from spanning two actions or lines,
        # so a generous length cap costs nothing
        text = re.sub(r"\*[^*\n]{1,300}\*", "", text).strip()
        if not text:
            return ""

        # an 8B model will happily echo its prompt when told to "repeat the
        # text above"; any reply touching the prompt is discarded outright
        text_l = text.lower()
        if any(marker in text_l for marker in PROMPT_ECHO_MARKERS):
            logger.info("Discarded prompt echo in reply")
            return ""

        # a paraphrase leak names the persona and describes it without quoting
        if (
            any(subject in text_l for subject in PROMPT_PARAPHRASE_SUBJECTS)
            and any(verb in text_l for verb in PROMPT_PARAPHRASE_VERBS)
        ):
            logger.info("Discarded prompt paraphrase in reply")
            return ""

        # a translated echo carries no English marker, so match its opener
        if text_l.startswith(PROMPT_ECHO_OPENERS):
            logger.info("Discarded translated prompt echo in reply")
            return ""

        # a translation that opens with the persona name instead
        if sum(marker in text_l for marker in PROMPT_ECHO_TRANSLATIONS) >= 2:
            logger.info("Discarded translated prompt echo in reply")
            return ""

        # the model dumps the whole tone list when asked to print what is
        # above; a real reply never contains several of them
        tone_hits = sum(1 for example in TONE_EXAMPLES if example in text_l)
        if tone_hits >= TONE_EXAMPLE_LEAK_THRESHOLD:
            logger.info("Discarded tone-example dump in reply")
            return ""

        # a single tone line is allowed only when the model builds on it.
        # when the reply is little more than the example itself, it is a
        # recital, not a reply
        if tone_hits and self._is_bare_tone_line(text_l):
            logger.info("Discarded bare tone-example reply")
            return ""

        parts = re.split(r"(?<=[.!?])\s+", text)
        cleaned: list[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            part_l = part.lower()
            if any(phrase in part_l for phrase in BANNED_PHRASES):
                continue
            cleaned.append(part)
            if len(cleaned) >= MAX_RESPONSE_SENTENCES:
                break

        if not cleaned:
            if not parts:
                return ""
            first_part = parts[0].strip()
            if not first_part:
                return ""
            first_part_l = first_part.lower()
            if any(phrase in first_part_l for phrase in BANNED_PHRASES):
                return ""
            cleaned = [first_part]
        if not cleaned:
            return ""

        if sum("?" in part for part in cleaned) > MAX_RESPONSE_QUESTIONS:
            pruned: list[str] = []
            questions = 0
            for part in cleaned:
                if "?" in part:
                    questions += 1
                    if questions > MAX_RESPONSE_QUESTIONS:
                        continue
                pruned.append(part)
            cleaned = pruned if pruned else cleaned[:1]

        return " ".join(cleaned).strip()

    @staticmethod
    def _is_bare_tone_line(text_l: str) -> bool:
        """True when the reply is a tone example with nothing added to it.

        The examples double as the model's best lines, so a reply that uses one
        and then keeps talking is fine. A reply that is only the example is the
        model reciting the prompt.
        """
        residue = text_l
        for example in TONE_EXAMPLES:
            residue = residue.replace(example, " ")
        words = [
            word
            for word in re.sub(r"[^a-z' ]", " ", residue).split()
            if word not in TONE_FILLER_WORDS
        ]
        return len(words) <= 1


def main() -> None:
    token = os.getenv(DISCORD_TOKEN_ENV)
    if not token:
        raise RuntimeError(f"Missing {DISCORD_TOKEN_ENV} environment variable")

    llm_url = os.getenv(LLM_URL_ENV, DEFAULT_LLM_URL)
    memory_file = os.getenv(MEMORY_FILE_ENV, DEFAULT_MEMORY_FILE)
    rate_limiter = RateLimiter(RATE_LIMIT_SECONDS)
    memory_manager = MemoryManager(memory_file)
    client = DiscordLlamaBot(llm_url, rate_limiter, memory_manager)
    client.run(token)


if __name__ == "__main__":
    main()
