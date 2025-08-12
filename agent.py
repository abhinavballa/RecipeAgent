import os
from functools import lru_cache
from typing import Dict

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    AgentSession,
    Agent,
    RoomInputOptions,
    function_tool,
    ChatContext,
    ChatMessage,
)
from livekit.plugins import (
    openai,
    cartesia,
    deepgram,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# LlamaIndex (RAG) imports
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI

load_dotenv()

# ---------------------------
# RAG: Cookbook Index Helpers
# ---------------------------

def _project_path(*rel_parts: str) -> str:
    """Resolve an absolute path under the project directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), *rel_parts))


def _ensure_dirs(path: str) -> None:
    """Create directories for a given path if they do not exist."""
    os.makedirs(path, exist_ok=True)


def _init_llamaindex_settings() -> None:
    """
    Configure global LlamaIndex settings for embeddings and LLM.
    We use OpenAI for both, leveraging the OPENAI_API_KEY from the environment.
    """
    # Favor precision for chapter-specific facts
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
    Settings.llm = LlamaOpenAI(model="gpt-4o-mini")
    # Configure robust chunking for better retrieval granularity
    Settings.node_parser = SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=120,
        separator="\n",
    )


def _cookbook_pdf_path() -> str:
    """Return the absolute path to the cookbook PDF in `docs/`."""
    return _project_path("docs", "Stealth Health - Meal Planning.pdf")


def _cookbook_index_dir() -> str:
    """Directory where the VectorStoreIndex is persisted on disk."""
    return _project_path("storage", "cookbook_index")


def build_or_load_cookbook_index() -> VectorStoreIndex:
    """
    Build a VectorStoreIndex from the cookbook PDF if not already persisted; otherwise load it.
    The index is stored under `storage/cookbook_index`.
    """
    _init_llamaindex_settings()

    persist_dir = _cookbook_index_dir()
    _ensure_dirs(persist_dir)

    try:
        # Try to load an existing index first for fast startup
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
        return index
    except Exception:
        # Fall back to building from the source PDF
        pdf_path = _cookbook_pdf_path()
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(
                f"Cookbook PDF not found at '{pdf_path}'. Please ensure the file exists."
            )

        documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=persist_dir)
        return index


@lru_cache(maxsize=1)
def get_cookbook_query_engine():
    """
    Create (and cache) a LlamaIndex query engine for the cookbook.
    Using an LRU cache ensures we only build/load once per process.
    """
    index = build_or_load_cookbook_index()
    return index.as_query_engine(response_mode="compact")


def get_cookbook_retriever(top_k: int = 6):
    """Return a retriever for the cookbook index."""
    index = build_or_load_cookbook_index()
    return index.as_retriever(similarity_top_k=top_k)


def _extract_chapter_hint(text: str) -> str | None:
    """Detect a chapter hint like 'chapter 3' or 'chapter: Meal Prep Basics'."""
    t = text.lower()
    # Simple numeric match
    import re

    m = re.search(r"chapter\s*(\d+)", t)
    if m:
        return f"chapter {m.group(1)}"
    # Title-like hint: 'chapter: foo'
    m2 = re.search(r"chapter\s*[:\-]\s*([\w\s]+)", t)
    if m2:
        return f"chapter {m2.group(1).strip()}"
    return None


# --------------------------------
# Tool: Cookbook RAG Query (LLM-Fn)
# --------------------------------

@function_tool()
def query_cookbook(question: str) -> str:
    """
    High-precision RAG over the cookbook for chapter-specific facts.
    Returns a concise answer with citations to page numbers when available.
    """
    retriever = get_cookbook_retriever(top_k=8)
    nodes = retriever.retrieve(question)

    chapter_hint = _extract_chapter_hint(question)
    if chapter_hint:
        # Simple re-ranking: promote nodes that mention the chapter hint
        def score(n):
            content = (n.node.get_content(metadata=False) or "").lower()
            return (chapter_hint in content)

        nodes = sorted(nodes, key=lambda n: score(n), reverse=True)

    # Build concise context with citations
    context_chunks = []
    citations = []
    for n in nodes[:4]:
        content = (n.node.get_content(metadata=False) or "").strip()
        if not content:
            continue
        meta = n.node.metadata or {}
        page = meta.get("page_label") or meta.get("page")
        if page is not None:
            citations.append(str(page))
        # keep chunks short to avoid latency
        context_chunks.append(content[:800])

    combined_context = "\n---\n".join(context_chunks)
    cite_str = f" (pages {', '.join(dict.fromkeys(citations))})" if citations else ""

    # Synthesize a concise answer using the LLM anchored to retrieved context
    try:
        prompt = (
            "You are answering a question using the provided cookbook context. "
            "Cite page numbers in parentheses if present. Be precise and concise.\n\n"
            f"Question: {question}\n\n"
            f"Context{cite_str}:\n{combined_context}\n\n"
            "Answer:"
        )
        comp = Settings.llm.complete(prompt)
        text = getattr(comp, "text", None) or str(comp)
        return (text.strip() + (f" {cite_str}" if cite_str else "")).strip()
    except Exception:
        # Fallback: return the most relevant snippets with citations
        return f"Context{cite_str}: {combined_context}"


# --------------------------------------
# Tool: Measurement Conversion (LLM-Fn)
# --------------------------------------

def _normalize_unit(raw_unit: str) -> str:
    """Normalize unit strings to canonical tokens."""
    u = raw_unit.strip().lower().replace(".", "").replace(" ", "")
    aliases: Dict[str, str] = {
        # volume
        "cup": "cup",
        "cups": "cup",
        "tbsp": "tbsp",
        "tablespoon": "tbsp",
        "tablespoons": "tbsp",
        "tbs": "tbsp",
        "tsp": "tsp",
        "teaspoon": "tsp",
        "teaspoons": "tsp",
        "ml": "ml",
        "milliliter": "ml",
        "millilitre": "ml",
        "milliliters": "ml",
        "millilitres": "ml",
        "l": "l",
        "liter": "l",
        "litre": "l",
        "liters": "l",
        "litres": "l",
        "floz": "fl_oz",
        "fl_oz": "fl_oz",
        "fluidounce": "fl_oz",
        "fluidounces": "fl_oz",
        # mass
        "g": "g",
        "gram": "g",
        "grams": "g",
        "kg": "kg",
        "kilogram": "kg",
        "kilograms": "kg",
        "oz": "oz",  # mass ounce
        "ounce": "oz",
        "ounces": "oz",
        "lb": "lb",
        "lbs": "lb",
        "pound": "lb",
        "pounds": "lb",
    }
    return aliases.get(u, u)


VOLUME_TO_ML: Dict[str, float] = {
    "cup": 240.0,
    "tbsp": 15.0,
    "tsp": 5.0,
    "ml": 1.0,
    "l": 1000.0,
    "fl_oz": 29.5735,
}

MASS_TO_G: Dict[str, float] = {
    "g": 1.0,
    "kg": 1000.0,
    "oz": 28.3495,
    "lb": 453.592,
}


def _is_volume(unit: str) -> bool:
    return unit in VOLUME_TO_ML


def _is_mass(unit: str) -> bool:
    return unit in MASS_TO_G


@function_tool()
def convert_measurements(value: float, unit_from: str, unit_to: str) -> float:
    """
    Convert common cooking measurements.

    Supported volume units: cup, tbsp, tsp, ml, l, fl_oz
    Supported mass units: g, kg, oz, lb

    Converting between volume and mass assumes a density of 1 g/ml (water-like),
    which is an approximation. For ingredients like flour or oil, results may vary.
    """
    if value is None:
        raise ValueError("'value' must be provided")

    uf = _normalize_unit(unit_from)
    ut = _normalize_unit(unit_to)

    if uf == ut:
        return float(round(value, 4))

    # Volume → Volume
    if _is_volume(uf) and _is_volume(ut):
        ml_value = value * VOLUME_TO_ML[uf]
        result = ml_value / VOLUME_TO_ML[ut]
        return float(round(result, 4))

    # Mass → Mass
    if _is_mass(uf) and _is_mass(ut):
        g_value = value * MASS_TO_G[uf]
        result = g_value / MASS_TO_G[ut]
        return float(round(result, 4))

    # Cross-dimension using default density of 1 g/ml
    DENSITY_G_PER_ML = 1.0

    # Volume → Mass
    if _is_volume(uf) and _is_mass(ut):
        ml_value = value * VOLUME_TO_ML[uf]
        g_value = ml_value * DENSITY_G_PER_ML
        result = g_value / MASS_TO_G[ut]
        return float(round(result, 4))

    # Mass → Volume
    if _is_mass(uf) and _is_volume(ut):
        g_value = value * MASS_TO_G[uf]
        ml_value = g_value / DENSITY_G_PER_ML
        result = ml_value / VOLUME_TO_ML[ut]
        return float(round(result, 4))

    raise ValueError(
        f"Unsupported conversion from '{unit_from}' to '{unit_to}'."
    )


class ChefRamsay(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are an AI assistant persona of celebrity chef Gordon Ramsay. "
                "Your name is Chef Ramsay. You are a highly critical, impatient, "
                "and demanding mentor for home cooks. Your tone is often angry, "
                "sarcastic, and slightly frustrated, but your goal is to make the "
                "user a better cook. You use British slang and common Gordon Ramsay "
                "catchphrases. "
                "You will not tolerate mistakes or laziness from the user. "
                "When asked for a recipe or cooking advice, you will provide "
                "it in your characteristic tough-love style. "
                "Keep your responses direct and to the point. "
                "You have access to a meal prep cookbook and a measurement conversion tool. "
                "You must use these resources to answer questions and fulfill user requests. "
                "When the user asks about recipes, ingredients, or meal prep techniques, "
                "query the cookbook tool to retrieve precise information. "
                "When the user requests conversions (e.g., cups to grams), call the conversion tool. "
                "Always maintain your Gordon Ramsay persona in responses."
            ),
            # Register tools here per LiveKit's tool-call pattern
            tools=[query_cookbook, convert_measurements],
        )

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ) -> None:
        """
        Inject cookbook RAG context before LLM generation, following
        https://docs.livekit.io/agents/build/external-data/#add-context-during-conversation
        """
        # Get message text; accommodate versions where it's a property, not a method
        tc = getattr(new_message, "text_content", None)
        text = tc() if callable(tc) else tc
        if not text:
            return

        # Simple heuristic: only fetch RAG context if message seems like a question
        # or references core cooking topics to avoid unnecessary queries.
        lower = text.lower()
        triggers = [
            "recipe",
            "ingredient",
            "ingredients",
            "prep",
            "meal prep",
            "cook",
            "cooking",
            "how do i",
            "how to",
            "method",
            "technique",
            "calories",
            "macros",
            "substitute",
        ]
        should_query = "?" in lower or any(t in lower for t in triggers)
        if not should_query:
            return

        try:
            retriever = get_cookbook_retriever(top_k=5)
            nodes = retriever.retrieve(text)

            chapter_hint = _extract_chapter_hint(text)
            if chapter_hint:
                def score(n):
                    content = (n.node.get_content(metadata=False) or "").lower()
                    return (chapter_hint in content)
                nodes = sorted(nodes, key=lambda n: score(n), reverse=True)

            snippets = []
            pages = []
            for n in nodes[:3]:
                content = (n.node.get_content(metadata=False) or "").strip()
                if content:
                    snippets.append(content[:500])
                meta = n.node.metadata or {}
                page = meta.get("page_label") or meta.get("page")
                if page is not None:
                    pages.append(str(page))

            if snippets:
                cite = ", ".join(dict.fromkeys(pages))
                condensed = " \n---\n".join(snippets)
                turn_ctx.add_message(
                    role="assistant",
                    content=f"Cookbook context{(f' (pages {cite})' if cite else '')}: {condensed}",
                )
        except Exception as e:
            # Non-fatal: if RAG fails, continue without extra context
            print(f"[WARN] RAG lookup failed in on_user_turn_completed: {e}")


async def entrypoint(ctx: agents.JobContext):

    # Pre-warm the RAG engine so first query is snappy
    try:
        _ = get_cookbook_query_engine()
    except Exception as e:
        # If indexing fails at startup, proceed without RAG but log the issue.
        # The tool will raise a useful error if called later.
        print(f"[WARN] Failed to initialize cookbook index: {e}")

    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="en-US"), # Changed to 'en-US' for better English performance
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(model="sonic-2", voice="63ff761f-c1e8-414b-b969-d1833d1c870c"),
        vad=silero.VAD.load(),
        # turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=ChefRamsay(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

    # The initial greeting must also be in the Gordon Ramsay persona.
    # This sets the tone from the very beginning.
    await session.generate_reply(
        instructions=(
            "Greet the user in the persona of Chef Gordon Ramsay. "
            "Tell them you are here to make them a better cook and to "
            "make their meal prep less of a disaster. "
            "Be demanding and impatient, using phrases like 'don't waste my time'. "
            "Mention that you can pull from a cookbook and convert measurements on command."
        )
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))