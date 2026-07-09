# YouTube Video RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers natural-language questions about a YouTube video, grounded in that video's own transcript. When the transcript doesn't fully cover a question, the system falls back to Wikipedia for supplementary information — clearly labeling which parts of the answer came from which source. The system also supports multi-turn conversation, remembering earlier questions and answers within a session.

## Features

- Fetches and indexes a YouTube video's transcript into a per-video vector store
- Answers questions using semantic retrieval (MMR) over the transcript
- Decides, per question, whether the video content alone is sufficient to answer well
- Falls back to Wikipedia only when video context is insufficient, via an LLM-driven query rewrite step
- Maintains conversational memory so follow-up questions resolve correctly
- Exposes the pipeline via a FastAPI backend and a Streamlit chat frontend

## Architecture
YouTube video page ->
Extract video ID → Fetch transcript (youtube-transcript-api) -> Chunk transcript (RecursiveCharacterTextSplitter, 1000 chars / 200 overlap) -> Embed chunks (OpenAI text-embedding-3-small) → Store in Chroma
       (collection scoped per video_id — isolated namespace)

On a question:
MMR retriever (k=4) → video_context ->
Sufficiency check (structured LLM call, Pydantic schema: sufficient: bool, reason: str)
│
├── sufficient  → skip Wikipedia
└── insufficient → rewrite question into a search query → fetch Wikipedia summary
│
▼
Assemble prompt: video_context + wiki_context (if any) + chat_history
│
▼
LLM (gpt-4o-mini) generates answer, labeling any Wikipedia-derived content
│
▼
Answer returned + chat_history updated (capped at last 8 messages)
## Tech Stack

- **Orchestration:** LangChain (LCEL — `RunnableParallel`, `RunnableBranch`, `RunnableLambda`, `RunnablePassthrough`)
- **Vector store:** Chroma (per-video collections)
- **Embeddings / LLM:** OpenAI (`text-embedding-3-small`, `gpt-4o-mini`)
- **Transcript ingestion:** `youtube-transcript-api`
- **Supplementary retrieval:** `wikipedia-api`
- **Structured output:** Pydantic (`PydanticOutputParser`)
- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Config:** python-dotenv

## Project Structure
youtube-chatbot/
├── rag_pipeline.py   # Core RAG logic — ingestion, retrieval, conditional Wikipedia augmentation, memory
├── main.py           # FastAPI backend (/ingest, /ask endpoints)
├── streamlit_app.py  # Streamlit chat frontend
├── .env              # OPENAI_API_KEY (not committed — see below)
└── chroma_db/        # Persisted vector store (created at runtime, not committed)

## Setup

1. Clone the repo and create a virtual environment:
```bash
   git clone https://github.com/atharva0324/Youtube_chatbot.git
   cd Youtube_chatbot
   conda create -n tf_env python=3.10
   conda activate tf_env
```

2. Install dependencies:
```bash
   pip install langchain langchain-openai langchain-chroma langchain-text-splitters langchain-core \
               youtube-transcript-api wikipedia-api pydantic python-dotenv fastapi uvicorn streamlit requests
```

3. Create a `.env` file in the project root:
OPENAI_API_KEY=your_openai_api_key_here

4. Run the backend:
```bash
   uvicorn main:app --reload
```

5. In a separate terminal, run the frontend:
```bash
   streamlit run streamlit_app.py
```

6. Open the Streamlit app (usually `http://localhost:8501`), paste a YouTube video link, and start asking questions.

## Key Design Decisions

- **Per-video Chroma collections instead of one shared collection** — keeps retrieval scoped correctly, since a retriever built on `collection_name=video_id` cannot see another video's vectors. Also gives natural caching: if a video was already ingested, the app skips straight to retrieval.
- **MMR retrieval instead of plain similarity search** — reduces redundant, near-duplicate chunks in the context window.
- **Conditional (branch-based) Wikipedia retrieval instead of always-on** — a structured sufficiency check (Pydantic schema) decides whether outside information is actually needed, avoiding both wasted latency/cost and irrelevant context pollution on questions the video already answers well.
- **Query rewriting before hitting Wikipedia** — a raw conversational question is a poor Wikipedia search query, so an LLM call reduces it to a clean topic phrase first.
- **Source-labeling in the system prompt** — the model is explicitly instructed to prefix any Wikipedia-derived content with "Additional info (not from the video):" so a user always knows what's transcript-grounded vs. supplementary.

## Known Limitations

- Session state (`retrievers_by_video`, `chat_histories_by_video`) is held in-memory per process — it resets on server restart and isn't safe across multiple worker processes. A production version would use a shared store like Redis.
- Naive top-k retrieval isn't well-suited to whole-video summarization questions — a "summarize the video" request would need a separate full-transcript or map-reduce path.
- No timestamp-aware chunking yet, so answers can't currently link back to a specific point in the video (e.g. "this is mentioned at 4:32").

## Possible Next Steps

- Timestamp-aware chunking + clickable "jump to this point in the video" links
- Auto-generated chapter markers from the transcript
- Redis-backed session state for multi-user, multi-process deployment
- Chrome extension frontend (in addition to Streamlit) for in-page YouTube Q&A

## Author

Atharva Shinde — [GitHub](https://github.com/atharva0324) · [Portfolio](https://atharva0324.github.io)
