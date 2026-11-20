# Discord llama.cpp Bot

Locally-Hosted discord bot to connect to llama.cpp

## Features

- **Local AI Generation:** Connects directly to a local llama.cpp `v1/chat/completions` endpoint (llama-server or llama-swap). No external API keys required!
- **Persistent Memory:** Automatically extracts conversational facts (like names and user preferences) and saves them to a local `.sqlite` database.
- **Contextual History:** Keeps a rolling window of recent messages for natural, conversational flow.
- **Rate Limiting:** Built-in safeguards to prevent API spam.
- **Docker / CasaOS Ready:** Includes a `docker-compose.example.yml` configured for easy deployment on CasaOS or standard Docker environments.

## Prerequisites

- [Python 3.11+](https://www.python.org/downloads/) (if running locally)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` running, or [llama-swap](https://github.com/mostlygeek/llama-swap) fronting it.
- A model loaded and served over the OpenAI-compatible API.
- Set the served model name with `LLM_MODEL` in `main.py` (defaults to `Qwen 3.6`).
  The request sends `chat_template_kwargs: {"enable_thinking": false}`, which keeps
  Qwen-family models from spending the reply budget on a reasoning block. A model
  that ignores the flag still works; it just answers slower.
- A [Discord Bot Token](https://discord.com/developers/applications).

## Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/senoxide/llamacpp-discord.git
   cd llamacpp-discord
   ```

2. **Create a virtual environment (fish shell):**
   ```fish
   python3 -m venv .venv
   source .venv/bin/activate.fish
   ```
   Your prompt gains a `(.venv)` prefix. Every later `python`/`pip` call uses it.
   To leave the environment, run `deactivate`.

   > Not using fish? Use `source .venv/bin/activate` in bash/zsh, or
   > `.venv\Scripts\activate` in Windows PowerShell.

3. **Install dependencies:**
   ```fish
   pip install -r requirements.txt
   ```

4. **Configure your environment:**
   Copy the example file and fill in your token. `main.py` reads `.env`
   automatically, so the variables are available to `python main.py` without
   exporting them by hand.
   ```fish
   cp .env.example .env
   ```
   Then edit `.env`:
   ```dotenv
   DISCORD_TOKEN=your_discord_bot_token
   LLM_URL=http://localhost:8080/v1/chat/completions
   MEMORY_FILE=user_memory.sqlite
   ```
   `.env` is git-ignored, so your token never lands in the repository.

   > Prefer shell variables instead? They take precedence over `.env`, so you
   > can override a single value for the current session only:
   > ```fish
   > set -x DISCORD_TOKEN "your_discord_bot_token"
   > set -x LLM_URL "http://localhost:8080/v1/chat/completions"
   > set -x MEMORY_FILE "user_memory.sqlite"
   > ```
   > `set -x` lasts only for that terminal. Use `set -Ux` to persist it across
   > sessions, and `set -e` to unset it again.

5. **Run the bot:**
   ```fish
   python main.py
   ```

## Docker Setup

A `docker-compose.example.yml` is included for containerized environments.

1. Rename the example file to `docker-compose.yml`
2. Update the `DISCORD_TOKEN` environment variable with your actual bot token.
3. Ensure the mapped volume target has your `main.py` and `requirements.txt` available (the example uses `/DATA/AppData/discord_bot`).
4. Update `LLM_URL` if your llama.cpp server is not reachable via `host.docker.internal`.
5. Run using docker compose:
   ```bash
   docker compose up -d
   ```
