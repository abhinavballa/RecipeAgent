import os
import hashlib
from glob import glob
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
from llama_index.core.schema import TextNode
import re
from typing import List
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


class RecipeAwareChunker(SentenceSplitter):
    """Enhanced chunker that detects recipe boundaries and classifies content."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def get_nodes_from_documents(self, documents, show_progress=False, **kwargs):
        """Override to add recipe-aware processing."""
        all_nodes = []
        
        for document in documents:
            text = document.get_content()
            recipe_chunks = self._split_by_recipes(text)
            
            for recipe_name, recipe_text in recipe_chunks:
                # Split recipe into smaller semantic chunks
                base_chunks = super().split_text(recipe_text)
                
                for chunk_text in base_chunks:
                    content_type = self._classify_content_type(chunk_text)
                    
                    # Create enhanced node with metadata
                    node = TextNode(
                        text=chunk_text,
                        metadata={
                            "recipe_name": recipe_name,
                            "content_type": content_type,
                            **document.metadata
                        }
                    )
                    all_nodes.append(node)
        
        return all_nodes
    
    def _split_by_recipes(self, text: str) -> List[tuple[str, str]]:
        """Split text into recipe sections and identify recipe names."""
        lines = text.split('\n')
        recipes = []
        current_recipe_name = "Unknown Recipe"
        current_recipe_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect recipe titles (common patterns)
            if self._is_recipe_title(line):
                # Save previous recipe if exists
                if current_recipe_lines:
                    recipes.append((current_recipe_name, '\n'.join(current_recipe_lines)))
                
                # Start new recipe
                current_recipe_name = self._clean_recipe_name(line)
                current_recipe_lines = [line]
            else:
                current_recipe_lines.append(line)
        
        # Don't forget the last recipe
        if current_recipe_lines:
            recipes.append((current_recipe_name, '\n'.join(current_recipe_lines)))
        
        return recipes
    
    def _is_recipe_title(self, line: str) -> bool:
        """Detect if a line is likely a recipe title."""
        line = line.strip()
        
        # Common recipe title patterns
        title_patterns = [
            r'^[A-Z][A-Za-z\s&-]+(?:Recipe|Meal Prep|Stir[- ]?Fry|Soup|Salad|Pasta|Chicken|Beef)(?:\s|$)',
            r'^[A-Z][A-Za-z\s&-]+(?:with|and|in|al|de)\s+[A-Z][A-Za-z\s&-]+$',
            r'^[A-Z][A-Za-z\s&-]+\s+(?:Bowl|Wrap|Sandwich|Burrito|Tacos?)$',
        ]
        
        # Check patterns
        for pattern in title_patterns:
            if re.match(pattern, line):
                return True
        
        # Additional heuristics
        if (len(line) < 50 and 
            len(line) > 10 and 
            line[0].isupper() and 
            not re.match(r'^\d', line) and  # Not a numbered instruction
            ':' not in line and  # Not an ingredient line
            '.' not in line[:20]):  # Not a sentence start
            return True
            
        return False
    
    def _clean_recipe_name(self, title_line: str) -> str:
        """Extract clean recipe name from title line."""
        # Remove common prefixes/suffixes
        name = title_line.strip()
        name = re.sub(r'^(Recipe:?\s*|Meal Prep:?\s*)', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s*(Recipe|Meal Prep)\s*$', '', name, flags=re.IGNORECASE)
        return name.strip()
    
    def _classify_content_type(self, text: str) -> str:
        """Classify chunk content as ingredients, instructions, or other."""
        text_lower = text.lower()
        
        # Ingredient patterns
        ingredient_patterns = [
            r'\b\d+\s*(?:cups?|tbsp|tablespoons?|tsp|teaspoons?|lbs?|pounds?|oz|ounces?|grams?|ml|liters?)\b',
            r'^\s*\d+(?:/\d+)?\s+',  # Starting with fractions or numbers
            r':\s*$',  # Lines ending with colon (ingredient headers)
        ]
        
        # Instruction patterns  
        instruction_patterns = [
            r'^\s*\d+\.',  # Numbered steps
            r'\b(?:preheat|heat|cook|bake|boil|simmer|sauté|fry|mix|stir|add|place|transfer|remove)\b',
            r'\b(?:minutes?|hours?|until|degrees?|°[CF])\b',
        ]
        
        # Count pattern matches
        ingredient_score = sum(1 for p in ingredient_patterns if re.search(p, text))
        instruction_score = sum(1 for p in instruction_patterns if re.search(p, text_lower))
        
        if ingredient_score > instruction_score:
            return "ingredients"
        elif instruction_score > 0:
            return "instructions"
        else:
            return "description"


def _init_llamaindex_settings() -> None:
    """
    Configure global LlamaIndex settings for embeddings and LLM.
    We use OpenAI for both, leveraging the OPENAI_API_KEY from the environment.
    """
    # Favor precision for chapter-specific facts
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
    Settings.llm = LlamaOpenAI(model="gpt-4o-mini")
    # Configure recipe-aware chunking
    Settings.node_parser = RecipeAwareChunker(
        chunk_size=1024,
        chunk_overlap=120,
        separator="\n",
    )


def _cookbook_pdf_paths() -> list[str]:
    """Return absolute paths to all PDF files in `docs/`."""
    docs_dir = _project_path("docs")
    pdfs = sorted(glob(os.path.join(docs_dir, "*.pdf")))
    return pdfs


def _cookbook_index_dir() -> str:
    """Base directory where indices are persisted on disk."""
    return _project_path("storage", "cookbook_index")


def build_or_load_cookbook_index() -> VectorStoreIndex:
    """
    Build a VectorStoreIndex from the cookbook PDF if not already persisted; otherwise load it.
    The index is stored under `storage/cookbook_index`.
    """
    _init_llamaindex_settings()

    # Version the index by embedding model and chunking config to avoid
    # shape mismatches when these settings change between runs.
    def _index_signature() -> str:
        embed = Settings.embed_model
        model_id = getattr(embed, "model", getattr(embed, "_model", "unknown"))
        chunker = Settings.node_parser
        chunk_size = getattr(chunker, "chunk_size", "na")
        chunk_overlap = getattr(chunker, "chunk_overlap", "na")
        # include data signature: filenames + mtimes
        pdf_paths = _cookbook_pdf_paths()
        data_sig_src = "|".join(
            f"{os.path.basename(p)}:{int(os.path.getmtime(p))}" for p in pdf_paths
        ) if pdf_paths else "no_docs"
        data_sig = hashlib.md5(data_sig_src.encode("utf-8")).hexdigest()[:12]
        return f"{model_id}_cs{chunk_size}_co{chunk_overlap}_d{data_sig}"

    persist_dir = _project_path("storage", "cookbook_index", _index_signature())
    _ensure_dirs(persist_dir)

    try:
        # Try to load an existing index first for fast startup
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
        return index
    except Exception:
        # Fall back to building from PDF(s)
        pdf_paths = _cookbook_pdf_paths()
        if not pdf_paths:
            raise FileNotFoundError(
                f"No PDF files found in '{_project_path('docs')}'. Place your cookbook PDF(s) there."
            )

        documents = SimpleDirectoryReader(input_files=pdf_paths).load_data()
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


# Removed hardcoded topic reranking to support arbitrary cookbooks


# --------------------------------
# Tool: Cookbook RAG Query (LLM-Fn)
# --------------------------------

@function_tool()
def get_recipe_details(recipe_name: str, content_type: str = "all") -> str:
    """
    Get specific parts of a recipe by name and content type.
    
    Args:
        recipe_name: Name of the recipe to find
        content_type: Type of content to return - "ingredients", "instructions", "description", or "all"
    
    Returns:
        Formatted recipe information with page citations when available
    """
    retriever = get_cookbook_retriever(top_k=10)
    
    # Search for the specific recipe
    search_query = f"recipe {recipe_name}"
    nodes = retriever.retrieve(search_query)
    
    # Filter nodes by recipe name and content type
    filtered_nodes = []
    for node in nodes:
        meta = node.node.metadata or {}
        node_recipe = meta.get("recipe_name", "").lower()
        node_content_type = meta.get("content_type", "")
        
        # Check if this node belongs to the requested recipe
        if recipe_name.lower() in node_recipe or node_recipe in recipe_name.lower():
            if content_type == "all" or node_content_type == content_type:
                filtered_nodes.append(node)
    
    if not filtered_nodes:
        return f"Could not find recipe '{recipe_name}' or content type '{content_type}' in the cookbook."
    
    # Build response organized by content type
    response_parts = []
    citations = []
    
    # Group by content type
    ingredients = []
    instructions = []
    descriptions = []
    
    for node in filtered_nodes[:6]:  # Limit to prevent TTS overload
        content = (node.node.get_content() or "").strip()
        if not content:
            continue
            
        meta = node.node.metadata or {}
        page = meta.get("page_label") or meta.get("page")
        if page:
            citations.append(str(page))
            
        node_type = meta.get("content_type", "description")
        clean_content = _clean_text_for_tts(content[:300])
        
        if node_type == "ingredients":
            ingredients.append(clean_content)
        elif node_type == "instructions":
            instructions.append(clean_content)
        else:
            descriptions.append(clean_content)
    
    # Format response
    if content_type == "ingredients" or (content_type == "all" and ingredients):
        response_parts.append(f"Ingredients: {'; '.join(ingredients)}")
    
    if content_type == "instructions" or (content_type == "all" and instructions):
        response_parts.append(f"Instructions: {'. '.join(instructions)}")
        
    if content_type == "description" or (content_type == "all" and descriptions):
        response_parts.append(f"Description: {'. '.join(descriptions)}")
    
    cite_str = f" (pages {', '.join(dict.fromkeys(citations))})" if citations else ""
    final_response = f"{recipe_name}: {' | '.join(response_parts)}{cite_str}"
    
    return final_response[:800]  # TTS-safe length


@function_tool()
def query_cookbook(question: str) -> str:
    """
    High-precision RAG over the cookbook for chapter-specific facts.
    Returns a concise answer with citations to page numbers when available.
    """
    retriever = get_cookbook_retriever(top_k=8)
    nodes = retriever.retrieve(question)

    # No topic reranking; support arbitrary PDFs

    # Build concise context with citations
    context_chunks = []
    citations = []
    for n in nodes[:4]:
        content = (n.node.get_content() or "").strip()
        if not content:
            continue
        meta = n.node.metadata or {}
        page = meta.get("page_label") or meta.get("page")
        if page is not None:
            citations.append(str(page))
        # keep chunks short to avoid latency
        context_chunks.append(content[:400])  # Reduced from 800 to 400

    combined_context = "\n---\n".join(context_chunks)
    cite_str = f" (pages {', '.join(dict.fromkeys(citations))})" if citations else ""

def _clean_text_for_tts(text: str) -> str:
    """Clean text to be TTS-friendly by removing problematic characters."""
    import re
    # Remove or replace problematic characters
    text = re.sub(r'[^\w\s\.,;:!?()-]', ' ', text)  # Keep basic punctuation
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = text.replace('&', 'and')  # Replace ampersands
    text = text.replace('°', ' degrees ')  # Replace degree symbols
    return text.strip()

    # Synthesize a concise answer using the LLM anchored to retrieved context
    try:
        prompt = (
            "You are answering a question using the provided cookbook context. "
            "Cite page numbers in parentheses if present. Be precise and concise. "
            "Keep your answer under 500 characters for voice synthesis.\n\n"
            f"Question: {question}\n\n"
            f"Context{cite_str}:\n{combined_context}\n\n"
            "Answer:"
        )
        comp = Settings.llm.complete(prompt)
        text = getattr(comp, "text", None) or str(comp)
        raw_response = (text.strip() + (f" {cite_str}" if cite_str else "")).strip()
        
        # Clean and truncate for TTS
        clean_response = _clean_text_for_tts(raw_response)
        return clean_response[:800]  # Hard limit for TTS compatibility
        
    except Exception:
        # Fallback: return cleaned and truncated snippets with citations
        fallback = f"Context{cite_str}: {combined_context}"
        clean_fallback = _clean_text_for_tts(fallback)
        return clean_fallback[:800]


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
                "Use get_recipe_details when users ask for specific parts of a recipe (ingredients, instructions). "
                "When the user requests conversions (e.g., cups to grams), call the conversion tool. "
                "Always maintain your Gordon Ramsay persona in responses."
            ),
            # Register tools here per LiveKit's tool-call pattern
            tools=[query_cookbook, get_recipe_details, convert_measurements],
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

            # No topic reranking; support arbitrary PDFs

            snippets = []
            pages = []
            for n in nodes[:3]:
                content = (n.node.get_content() or "").strip()
                if content:
                    snippets.append(content[:300])  # Reduced from 500 to 300
                meta = n.node.metadata or {}
                page = meta.get("page_label") or meta.get("page")
                if page is not None:
                    pages.append(str(page))

            if snippets:
                cite = ", ".join(dict.fromkeys(pages))
                condensed = " \n---\n".join(snippets)
                
                # Clean the context text for TTS compatibility
                def _clean_context_for_tts(text: str) -> str:
                    import re
                    text = re.sub(r'[^\w\s\.,;:!?()-]', ' ', text)
                    text = re.sub(r'\s+', ' ', text)
                    text = text.replace('&', 'and')
                    text = text.replace('°', ' degrees ')
                    return text.strip()
                
                clean_condensed = _clean_context_for_tts(condensed)
                context_msg = f"Cookbook context{(f' (pages {cite})' if cite else '')}: {clean_condensed}"
                
                # Truncate context message to avoid TTS issues
                turn_ctx.add_message(
                    role="assistant",
                    content=context_msg[:600],  # Hard limit for context injection
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