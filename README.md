# llamacpp-discord

discord bot that talks to a local llama.cpp server.

## what it does

- text generation via llama.cpp's OpenAI-compatible endpoint (llama-server or llama-swap in front of it)
- persistent memory: pulls facts out of conversations (names, preferences) into a local sqlite file
- keeps a rolling window of recent messages so replies have context
- rate limiting so one user can't hammer the endpoint

## setup

clone and enter a venv:

```bash
git clone https://github.com/senoxide/llamacpp-discord
cd llamacpp-discord
python3 -m venv .venv && source .venv/bin/activate
```

install deps:

```bash
pip install -r requirements.txt
```

configure:

```bash
cp .env.example .env
```

then edit `.env`:

```dotenv
DISCORD_TOKEN=your_discord_bot_token
LLM_URL=http://localhost:8080/v1/chat/completions
MEMORY_FILE=user_memory.sqlite
```

`.env` is git-ignored so the token stays out of the repo. shell variables
override `.env` if you'd rather export them yourself.

run:

```bash
python main.py
```

## notes

- python 3.11+ if running locally.
- there's a `docker-compose.example.yml` for containerized setups. rename it, set your token, check `LLM_URL` points at your llama.cpp host, then `docker compose up -d`.
