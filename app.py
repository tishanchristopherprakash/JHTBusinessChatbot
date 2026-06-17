import streamlit as st
import ollama
import os
from datetime import date as _date
from dotenv import load_dotenv

load_dotenv()

from sheets_checker import check_availability, find_available_slots
from intent_extractor import extract_booking_intent

# Import our engineering tools from LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

# ─────────────────────────────────────────────
# Configure UI Header
# ─────────────────────────────────────────────
st.set_page_config(page_title="RAG Business Bot", page_icon="🧠")
st.title("🧠 JHT RAG-Augmented Business Assistant")
st.caption("Using LangChain + ChromaDB + Local Qwen 2.5")

# ─────────────────────────────────────────────
# Step 1: Custom Ollama Embeddings
# ─────────────────────────────────────────────
class OllamaEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [ollama.embeddings(model="nomic-embed-text", prompt=t)["embedding"] for t in texts]

    def embed_query(self, text):
        return ollama.embeddings(model="nomic-embed-text", prompt=text)["embedding"]


# ─────────────────────────────────────────────
# Step 2: Load all .txt files from knowledge base folder
# ─────────────────────────────────────────────
def load_knowledge_files(folder: str = ".") -> list[Document]:
    """
    Loads all .txt files found in `folder`.
    Falls back to info.txt in the working directory if no folder exists.
    Returns a list of LangChain Documents with source metadata.
    """
    docs = []
    txt_files = [f for f in os.listdir(folder) if f.endswith(".txt")]

    if not txt_files:
        return docs

    for filename in txt_files:
        filepath = os.path.join(folder, filename)
        try:
            with open(filepath, "r", encoding="utf-8-sig", errors="ignore") as f:
                content = f.read().strip()
            if content:
                docs.append(Document(page_content=content, metadata={"source": filename}))
        except Exception as e:
            st.warning(f"Could not load {filename}: {e}")

    return docs


# ─────────────────────────────────────────────
# Step 3: Build & cache the vector store
# ─────────────────────────────────────────────
@st.cache_resource
def initialize_knowledge_base():
    documents = load_knowledge_files(".")

    if not documents:
        return None

    # FIX: Increased overlap to 150 to avoid cutting sentences at chunk boundaries
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    vector_store = Chroma.from_documents(chunks, OllamaEmbeddings())
    return vector_store


# ─────────────────────────────────────────────
# Step 4: Query rewriting for vague follow-ups
# ─────────────────────────────────────────────
def rewrite_query(conversation_history: list[dict], current_query: str) -> str:
    """
    Uses the LLM to rewrite a vague follow-up question into a standalone,
    self-contained question based on conversation history.
    This prevents 'tell me more' style queries from confusing the vector search.
    """
    if len(conversation_history) < 2:
        # No prior context — nothing to rewrite
        return current_query

    # Build a short summary of recent turns for the rewriter
    recent = ""
    for msg in conversation_history[-4:]:
        if msg["role"] in ("user", "assistant"):
            role = "User" if msg["role"] == "user" else "Assistant"
            recent += f"{role}: {msg['content']}\n"

    rewrite_prompt = f"""Given this conversation:
{recent}

Rewrite the following question as a fully self-contained, specific question 
that can be understood without any prior context. 
Return ONLY the rewritten question, nothing else.

Question to rewrite: {current_query}"""

    try:
        result = ollama.chat(
            model="qwen2.5:14b",
            messages=[{"role": "user", "content": rewrite_prompt}],
            options={"temperature": 0.0},
        )
        rewritten = result["message"]["content"].strip()
        return rewritten if rewritten else current_query
    except Exception:
        return current_query


# ─────────────────────────────────────────────
# Step 5: Initialise session state
# ─────────────────────────────────────────────
vector_db = initialize_knowledge_base()

# FIX: Warn user clearly if no knowledge base was found instead of silently failing
if vector_db is None:
    st.error(
        "⚠️ No knowledge base found. Please add one or more `.txt` files to the "
        "application directory and restart."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# ─────────────────────────────────────────────
# Sidebar: Schedule debug panel
# ─────────────────────────────────────────────
with st.sidebar:
    if st.button("🔍 Debug: Inspect Sheet"):
        from sheets_checker import load_sheet_data, find_date_rows, parse_time_slots
        sheet_id = os.environ.get("SHEET_ID", "")
        data = load_sheet_data(sheet_id)
        if data is None:
            st.error("Failed to load sheet — check API key and Sheet ID in .env")
        else:
            st.success(f"Sheet loaded. Rows: {len(data['grid_data'])}, Merges: {len(data['merges'])}")
            from sheets_checker import find_date_rows
            dates = find_date_rows(data["grid_data"])
            st.write("**Parsed date rows:**", dates if dates else "⚠️ None found")
            times = parse_time_slots(data)
            st.write("**Parsed time slots (first 10):**", dict(list(times.items())[:10]))
            if data["grid_data"]:
                row0_vals = [
                    data["grid_data"][0].get("values", [])[i].get("formattedValue", "")
                    if i < len(data["grid_data"][0].get("values", [])) else ""
                    for i in range(min(15, len(data["grid_data"][0].get("values", []))))
                ]
                st.write("**Row 0 raw cell values (first 15 cols):**", row0_vals)
            if len(data["grid_data"]) > 1:
                row1_vals = [
                    data["grid_data"][1].get("values", [])[i].get("formattedValue", "")
                    if i < len(data["grid_data"][1].get("values", [])) else ""
                    for i in range(min(15, len(data["grid_data"][1].get("values", []))))
                ]
                st.write("**Row 1 raw cell values (first 15 cols):**", row1_vals)

# ─────────────────────────────────────────────
# Step 6: Render conversation history
# ─────────────────────────────────────────────
for message in st.session_state.messages:
    if message["role"] not in ("system",):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            # Show source chunks that were used (stored alongside assistant messages)
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📄 View source context used"):
                    for i, chunk in enumerate(message["sources"], 1):
                        source_name = chunk.get("source", "unknown")
                        st.caption(f"**Chunk {i} — {source_name}**")
                        st.text(chunk["text"])


# ─────────────────────────────────────────────
# Step 7: Handle new user input
# ─────────────────────────────────────────────
if user_input := st.chat_input("Ask about company location, hours, or policies..."):

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # ── 7a: Detect greetings/chitchat — skip retrieval entirely ──
    GREETINGS = {"hello", "hi", "hey", "yo", "hiya", "good morning", "good afternoon",
                "good evening", "howdy", "greetings", "sup", "what's up", "heyy", "heyyy"}

    is_greeting = user_input.strip().lower().rstrip("!.,") in GREETINGS

    # ── 7a cont: Reply to greetings directly — no retrieval, no fallback prompt ──
    if is_greeting:
        greeting_reply = (
            "Hello! Welcome to JHT Cleaning Service. "
            "I can help you check appointment availability or answer questions about our services. "
            "What can I help you with today?"
        )
        with st.chat_message("assistant"):
            st.write(greeting_reply)
        st.session_state.messages.append(
            {"role": "assistant", "content": greeting_reply, "sources": []}
        )
        st.rerun()

    # ── 7b: Check for booking/availability intent before hitting the vector DB ──
    if not is_greeting:
        intent = extract_booking_intent(
            user_input,
            today=_date.today().isoformat(),
            conversation_history=st.session_state.messages[:-1],
        )
        if intent is not None:
            from datetime import datetime as _dt
            dur = intent["duration_hours"]
            friendly_dur = f"{int(dur)}-hour" if (dur and dur == int(dur)) else (f"{dur}-hour" if dur else None)
            try:
                friendly_date = _dt.strptime(intent["date"], "%Y-%m-%d").strftime("%B %-d")
            except Exception:
                friendly_date = intent["date"]

            if intent["start_time"] is None:
                # No specific time — list all available slots for that day
                lookup_dur = dur if dur else 0.5  # default to 30-min slots if no duration given
                slots = find_available_slots(intent["date"], lookup_dur)
                if isinstance(slots, str):  # error string
                    print(f"[DEBUG] find_available_slots returned: {slots}", flush=True)
                    if slots == "error:date_not_found":
                        bot_reply = (
                            f"I don't see {friendly_date} on our schedule yet — "
                            "it may not be open for booking. "
                            "Please contact us at +60172786498 or via WhatsApp to check."
                        )
                    else:
                        bot_reply = (
                            "I wasn't able to check the calendar right now. "
                            "Please contact us directly at +60172786498 or via WhatsApp to check availability."
                        )
                elif not slots:
                    suffix = f"for a {friendly_dur} clean" if friendly_dur else "on that day"
                    bot_reply = (
                        f"Sorry, there are no available slots on {friendly_date} {suffix}. "
                        "Would you like to try a different date?"
                    )
                else:
                    slots_str = ", ".join(slots)
                    suffix = f"for a {friendly_dur} clean" if friendly_dur else ""
                    bot_reply = (
                        f"On {friendly_date}, the available start times {suffix} are: {slots_str}. "
                        "Would any of these work for you?"
                    )
            else:
                result = check_availability(intent["date"], intent["start_time"], dur)
                print(f"[DEBUG] check_availability returned: {result}", flush=True)
                try:
                    friendly_time = _dt.strptime(intent["start_time"], "%H:%M").strftime("%-I:%M%p").lower()
                except Exception:
                    friendly_time = intent["start_time"]

                if result == "available":
                    bot_reply = (
                        f"Great news! {friendly_date} at {friendly_time} is available "
                        f"for a {friendly_dur} clean. "
                        "Please contact us to confirm your booking!"
                    )
                elif result == "taken":
                    bot_reply = (
                        f"Sorry, {friendly_date} at {friendly_time} is already booked. "
                        "Would you like to try a different date or time?"
                    )
                elif result == "error:date_not_found":
                    bot_reply = (
                        f"I don't see {friendly_date} on our schedule yet — "
                        "it may not be open for booking. "
                        "Please contact us at +60172786498 or via WhatsApp to check."
                    )
                else:
                    bot_reply = (
                        "I wasn't able to check the calendar right now. "
                        "Please contact us directly at +60172786498 or via WhatsApp to check availability."
                    )

            with st.chat_message("assistant"):
                st.write(bot_reply)
            st.session_state.messages.append(
                {"role": "assistant", "content": bot_reply, "sources": [], "booking_intent": intent}
            )
            st.rerun()

    # ── 7c: Rewrite vague follow-up questions before hitting the vector DB ──
    search_query = rewrite_query(st.session_state.messages[:-1], user_input)

    # ── 7c: Retrieve relevant chunks (skip if greeting) ──
    retrieved_chunks = []
    context_text = ""

    if vector_db and not is_greeting:
        search_results = vector_db.similarity_search(search_query, k=4)
        retrieved_chunks = [
            {"text": doc.page_content, "source": doc.metadata.get("source", "unknown")}
            for doc in search_results
        ]
        context_text = "\n\n---\n\n".join([c["text"] for c in retrieved_chunks])

    # ── 7c: Build system prompt ──
    if context_text:
        system_prompt = f"""You are a professional business support assistant.
Answer the user's question accurately using ONLY the context facts provided below.
If the context does not contain the answer, clearly state that you don't have that information.
Do not make up or infer facts not present in the context.

CONTEXT DATA FROM KNOWLEDGE BASE:
{context_text}
"""
    else:
        # FIX: Explicit no-knowledge-base fallback — don't let the LLM hallucinate
        system_prompt = """You are a professional business support assistant.
You currently have NO context data available to answer questions.
Politely inform the user that the knowledge base is unavailable and you cannot answer their question."""

    # ── 7d: Build LLM payload with recent conversation history ──
    llm_payload = [{"role": "system", "content": system_prompt}]
    for msg in st.session_state.messages[-6:]:
        if msg["role"] in ("user", "assistant"):
            llm_payload.append({"role": msg["role"], "content": msg["content"]})

    # ── 7e: Stream the response ──
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        response_stream = ollama.chat(
            model="qwen2.5:14b",
            messages=llm_payload,
            stream=True,
            options={"temperature": 0.0},
        )

        for chunk in response_stream:
            full_response += chunk["message"]["content"]
            response_placeholder.write(full_response + "▌")

        response_placeholder.write(full_response)

        # Show which source chunks were used for this answer
        if retrieved_chunks:
            with st.expander("📄 View source context used"):
                for i, chunk in enumerate(retrieved_chunks, 1):
                    st.caption(f"**Chunk {i} — {chunk['source']}**")
                    st.text(chunk["text"])

    # ── 7f: Persist to session state (including sources for history rendering) ──
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": retrieved_chunks,
    })
    st.rerun()