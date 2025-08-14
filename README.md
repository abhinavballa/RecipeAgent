# RecipeAgent – Real‑time Voice AI (Gordon Ramsay)

This project is a realtime voice agent that talks like Gordon Ramsay, answers cookbook questions using RAG over a local PDF, and can perform measurement conversions via tools. It uses LiveKit Cloud for realtime media, a Python worker for the agent, a FastAPI token server, and a React frontend.

## Architecture & End‑to‑End Flow

Components:
- Python Agent (LiveKit Agents): persona, tools, RAG, STT/TTS, VAD
- Token Server (FastAPI): securely issues LiveKit JWTs to the frontend
- React Frontend: connects to the room, starts audio, shows live transcript
- LiveKit Cloud: media SFU + job dispatch to the agent worker

Flow:
1) User clicks Start Call in the frontend → token server issues a JWT → frontend joins room on LiveKit Cloud.
2) LiveKit Cloud dispatches a job to the worker registered by the agent (`python3 agent.py start`).
3) Agent joins the room, performs STT (Deepgram), LLM reasoning (OpenAI), optional tools, and TTS (Cartesia or OpenAI TTS), then publishes audio and synchronized transcriptions.
4) Frontend plays audio and renders live transcripts via LiveKit text streams on topic `lk.transcription` (synchronized with speech by default). See LiveKit text/transcription docs: [Text and transcriptions](https://docs.livekit.io/agents/build/text/).

Repo layout (key paths):
- `agent.py` – agent logic (persona, tools, RAG, session config)
- `docs/Stealth Health - Meal Planning.pdf` – cookbook PDF
- `storage/cookbook_index/…` – on‑disk LlamaIndex (auto‑generated)
- `frontend/` – React client
- `backend/token_server.py` – FastAPI JWT token service

## How RAG Is Integrated

Framework: LlamaIndex.
- Documents: all `*.pdf` in `docs/` via `SimpleDirectoryReader`.
- Embeddings: OpenAI `text-embedding-3-small`.
- Chunking: `SentenceSplitter` with `chunk_size=1024`, `chunk_overlap=240`.
- Index: `VectorStoreIndex` persisted to `storage/cookbook_index/<embed>_cs<chunk>_co<overlap>_d<hash>/`.
- Versioning: directory name includes model id, chunk params, and a hash of PDF names + mtimes to prevent dimension mismatch or stale data.
- Query: tool `query_cookbook(question)` retrieves top nodes, synthesizes a concise answer (citations retained where available) and cleans output for TTS.

When the agent hears a cooking‑related question, `on_user_turn_completed` fetches short snippets and injects a brief “Cookbook context …” message before LLM generation to ground responses.

## Tools / Frameworks

- LiveKit Agents (Python) – worker lifecycle, session orchestration
- LiveKit Cloud – Selective Forwarding Unit, jobs & dispatch
- STT: Deepgram `nova-3`
- TTS: Cartesia `sonic-2` voice (or OpenAI TTS fallback)
- VAD: Silero
- LLM: OpenAI `gpt-4o-mini`
- RAG: LlamaIndex (see above)
- Frontend: React + `livekit-client` text streams for transcripts
- Backend: FastAPI token server

## Setup

Prerequisites:
- Python 3.11+
- Node.js 18+
- LiveKit Cloud project (URL + API key/secret)
- API keys: OpenAI, Deepgram, Cartesia (or use OpenAI TTS)

Install & run:
1) Token server
   ```bash
   cd backend
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   uvicorn token_server:app --host 0.0.0.0 --port 8000
   ```
2) Agent worker
   ```bash
   cd ..
   source venv/bin/activate  # or your environment
   python3 agent.py start
   ```
   Notes:
   - This registers a worker; when a user joins the room, a job is dispatched and the agent joins.
   - The first run builds the index into `storage/cookbook_index/...`.
3) Frontend
   ```bash
   cd frontend
   npm install
   npm start
   ```

Frontend actions:
- Click Start Call → audio unlock + join.
- You should hear the greeting and see transcript lines as you and the agent speak (we handle `lk.transcription` via text streams).

## Design Decisions & Assumptions

RAG:
- Vector store: LlamaIndex default `VectorStoreIndex` persisted locally (simple, no external DB).
- Chunking/embedding chosen for a single PDF cookbook; index versioned to avoid mismatches.
- Context injection favors short, TTS‑friendly snippets and includes page citations when available.

Agent design:
- Persona: Gordon Ramsay tough‑love; explicit instruction to use humorous, non‑discriminatory insults.
- Tools: `query_cookbook` (RAG), `convert_measurements` (unit conversions with basic density assumption 1 g/ml).
- Session: Deepgram STT → LLM → Tools → TTS; VAD via Silero; turn detection optional.
- Transcripts: published automatically by `AgentSession` and consumed via text streams topic `lk.transcription` on the frontend (per LiveKit docs: [Text and transcriptions](https://docs.livekit.io/agents/build/text/)).

Hosting assumptions:
- LiveKit Cloud is used for SFU/dispatch.
- Token server runs locally (port 8000) for dev; in prod, host behind HTTPS and configure CORS for your frontend origin.

Trade‑offs / limitations:
- Local file persistence means first load builds index; large PDFs may take longer.
- No external vector DB; for multi‑document or large corpora, consider a managed vector store.

## Operations & Troubleshooting

Common commands:
- Clear and recreate RAG storage (forces rebuild):
  ```bash
  rm -rf storage/cookbook_index && mkdir -p storage/cookbook_index
  ```

Known issues & fixes:
- Worker init timeout at startup
  - Symptom (logs): `worker failed ... TimeoutError` during inference runner initialization.
  - Meaning: a component (e.g., turn detector/VAD/provider init) exceeded the init window.
  - Fixes:
    - Increase init timeout:
      ```bash
      export LIVEKIT_AGENTS_INFERENCE_INIT_TIMEOUT_MS=30000
      ```
    - Disable heavy turn detection, rely on VAD.
    - Skip RAG pre-warm at startup; build index after `ctx.connect()`.

## Developing / Extending

- Implement hierarchical clustering to group recipes in the same section together and segment ingredients vs instructions more accurately.
- Add more tools: decorate with `@function_tool()` and include in the agent’s `tools=[...]` list.
- Tune grounding: adjust `similarity_top_k`, chunk sizes, and context injection limits.

## Additional Design Justifications & Tradeoffs

- LlamaIndex local persistence: simple dev UX; tradeoff is first-run build cost, more RAM usage and limited scalability without an external DB.
- Embedding model: `text-embedding-3-small` balances cost/latency vs accuracy; can upgrade if recall is insufficient.
- Chunking 1024/240: chosen for cookbook context density; adjust if retrieval becomes too generic (smaller) or too fragmented (larger/overlap).
- Context injection before reply: improves grounding vs. longer latency; capped length for TTS stability.


