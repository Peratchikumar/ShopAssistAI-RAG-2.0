
# ============================================================
# SHOPASSISTAI RAG 2.0
# Advanced E-Commerce Customer Support Chatbot
# Flask Backend - Local Development and Render Deployment
# ============================================================

import os
import uuid
import time
import logging
import threading

from pathlib import Path
from functools import lru_cache
from datetime import timedelta

import chromadb

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    session
)

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

VECTORSTORE_DIR = Path(
    os.getenv(
        "VECTORSTORE_DIR",
        str(BASE_DIR / "vectorstore")
    )
)

if not VECTORSTORE_DIR.is_absolute():
    VECTORSTORE_DIR = BASE_DIR / VECTORSTORE_DIR

COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "shopassistai_knowledge_base"
)

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-6-luna"
)

TOP_K = int(os.getenv("TOP_K", "5"))

MAX_HISTORY_MESSAGES = int(
    os.getenv("MAX_HISTORY_MESSAGES", "6")
)

MAX_MESSAGE_LENGTH = 2000

MAX_OUTPUT_TOKENS = 700

load_dotenv(BASE_DIR / ".env")

# Read environment variables after loading .env.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    OPENAI_MODEL
)

FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY")

if not FLASK_SECRET_KEY:
    FLASK_SECRET_KEY = uuid.uuid4().hex + uuid.uuid4().hex

    logging.warning(
        "FLASK_SECRET_KEY is not configured. "
        "A temporary development key has been generated. "
        "Configure a permanent secret before deployment."
    )


# ============================================================
# 2. LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger("ShopAssistAI")


# ============================================================
# 3. INITIALIZE FLASK APPLICATION
# ============================================================

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static")
)

app.secret_key = FLASK_SECRET_KEY

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=(
        os.getenv("COOKIE_SECURE", "false").lower() == "true"
    ),
    PERMANENT_SESSION_LIFETIME=timedelta(hours=2),
    MAX_CONTENT_LENGTH=1 * 1024 * 1024
)


# ============================================================
# 4. CONVERSATION STORAGE
# ============================================================

# This is in-memory storage for local development and a
# single-process demonstration deployment.
#
# Conversations are lost when the application restarts.
# For multiple workers or production persistence, use Redis
# or a database.

CONVERSATIONS = {}

CONVERSATION_LOCK = threading.RLock()


def get_session_id():
    """Get or create the customer's session ID."""

    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())

    session.permanent = True

    return session["session_id"]


def get_conversation_history(session_id):
    """Return conversation history for a session."""

    with CONVERSATION_LOCK:
        return list(CONVERSATIONS.get(session_id, []))


def save_conversation_history(session_id, history):
    """Save the latest conversation messages."""

    with CONVERSATION_LOCK:
        CONVERSATIONS[session_id] = history[
            -MAX_HISTORY_MESSAGES:
        ]


def reset_conversation(session_id):
    """Clear conversation history."""

    with CONVERSATION_LOCK:
        CONVERSATIONS[session_id] = []


# ============================================================
# 5. LOAD RAG BACKEND COMPONENTS
# ============================================================

@lru_cache(maxsize=1)
def load_backend():
    """
    Initialize and cache:
    1. OpenAI client
    2. ChromaDB collection
    3. Sentence Transformer embedding model
    """

    logger.info("Initializing ShopAssistAI RAG backend...")

    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is missing. "
            "Add it to your .env file or deployment environment."
        )

    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(
            f"Vectorstore directory not found: {VECTORSTORE_DIR}"
        )

    # OpenAI client.
    openai_client = OpenAI(
        api_key=OPENAI_API_KEY,
        timeout=60.0,
        max_retries=2
    )

    # ChromaDB persistent client.
    chroma_client = chromadb.PersistentClient(
        path=str(VECTORSTORE_DIR)
    )

    # Load existing collection.
    collection = chroma_client.get_collection(
        name=COLLECTION_NAME
    )

    record_count = collection.count()

    if record_count == 0:
        raise ValueError(
            "The ChromaDB collection is empty. "
            "Run your notebook's vector database indexing step."
        )

    # Load the embedding model used during indexing.
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    logger.info(
        "RAG backend initialized successfully. "
        "Collection: %s | Records: %s",
        COLLECTION_NAME,
        record_count
    )

    return (
        openai_client,
        collection,
        embedding_model
    )


# ============================================================
# 6. RETRIEVE RELEVANT DOCUMENTS
# ============================================================

def retrieve_documents(question):
    """Retrieve relevant documents from ChromaDB."""

    (
        openai_client,
        collection,
        embedding_model
    ) = load_backend()

    record_count = collection.count()

    if record_count == 0:
        return []

    # Convert the question into an embedding.
    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    ).tolist()

    # Query ChromaDB.
    result = collection.query(
        query_embeddings=query_embedding,
        n_results=min(TOP_K, record_count),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    retrieved_texts = result.get("documents", [[]])[0] or []

    retrieved_metadata = result.get("metadatas", [[]])[0] or []

    retrieved_distances = result.get("distances", [[]])[0] or []

    documents = []

    for text, metadata, distance in zip(
        retrieved_texts,
        retrieved_metadata,
        retrieved_distances
    ):

        if not text:
            continue

        metadata = metadata or {}

        source = (
            metadata.get("source")
            or metadata.get("file_name")
            or metadata.get("filename")
            or "Unknown source"
        )

        page = metadata.get("page", "N/A")

        similarity_score = None

        if distance is not None:
            similarity_score = round(
                1.0 - float(distance),
                4
            )

        documents.append({
            "content": str(text),
            "source": str(source),
            "page": str(page),
            "score": similarity_score
        })

    logger.info(
        "Retrieved %s knowledge-base documents.",
        len(documents)
    )

    return documents


# ============================================================
# 7. GENERATE ANSWER USING OPENAI RESPONSES API
# ============================================================

def generate_answer(question, conversation_history):
    """Generate an answer using retrieved RAG context."""

    start_time = time.time()

    # Retrieve relevant documents.
    retrieved_documents = retrieve_documents(question)

    if not retrieved_documents:
        return {
            "answer": (
                "I could not find relevant information in "
                "the knowledge base. Please contact customer support."
            ),
            "sources": [],
            "response_time": round(time.time() - start_time, 2),
            "token_usage": {}
        }

    # Build knowledge-base context.
    context_parts = []

    for index, document in enumerate(
        retrieved_documents,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {index}
Document: {document["source"]}
Page: {document["page"]}

Content:
{document["content"]}
"""
        )

    context = "\n\n".join(context_parts)

    # Prepare recent conversation history.
    history_parts = []

    for message in conversation_history[-MAX_HISTORY_MESSAGES:]:

        role = message.get("role", "user").upper()

        content = message.get("content", "")

        history_parts.append(
            f"{role}: {content}"
        )

    history_text = "\n".join(history_parts)

    # System instructions.
    system_instructions = """
You are ShopAssistAI, an intelligent e-commerce
customer-support assistant.

Your responsibilities are to provide helpful, accurate,
polite and context-aware customer-support answers.

IMPORTANT RULES:

1. Use the retrieved knowledge-base context to answer questions.

2. Do not invent return policies, refund policies, prices,
   product details or shipping information.

3. Clearly explain when the knowledge base does not contain
   the requested information.

4. Never claim to check live order status or inventory.

5. Never claim to execute refunds, payments, cancellations
   or exchanges.

6. The knowledge-base documents contain synthetic demonstration
   content. Do not present them as verified real-store policies.

7. Use conversation history to understand follow-up questions.

8. Mention relevant source document names when appropriate.

9. Give clear, professional and easy-to-understand answers.

10. If information is insufficient, ask the customer to contact
    the appropriate support team instead of guessing.

11. Use readable paragraphs and bullet points where appropriate.
"""

    # Construct user prompt.
    user_prompt = f"""
CUSTOMER CONVERSATION HISTORY:

{history_text}

RETRIEVED KNOWLEDGE-BASE CONTEXT:

{context}

CURRENT CUSTOMER QUESTION:

{question}

Provide an accurate, helpful and context-aware response.
Use only the supplied knowledge-base information for store-specific facts.
"""

    # Get OpenAI client.
    openai_client, collection, embedding_model = load_backend()

    logger.info(
        "Generating answer using model: %s",
        OPENAI_MODEL
    )

    # OpenAI Responses API.
    response = openai_client.responses.create(
        model=OPENAI_MODEL,
        instructions=system_instructions,
        input=user_prompt,
        max_output_tokens=MAX_OUTPUT_TOKENS
    )

    answer = (response.output_text or "").strip()

    if not answer:
        answer = (
            "I could not generate a response. "
            "Please try asking your question again."
        )

    # Extract token usage safely.
    usage = getattr(response, "usage", None)

    token_usage = {}

    if usage is not None:
        token_usage = {
            "input_tokens": getattr(
                usage,
                "input_tokens",
                0
            ),
            "output_tokens": getattr(
                usage,
                "output_tokens",
                0
            ),
            "total_tokens": getattr(
                usage,
                "total_tokens",
                0
            )
        }

    response_time = round(
        time.time() - start_time,
        2
    )

    # Prepare source information.
    sources = [
        {
            "source": document["source"],
            "page": document["page"],
            "score": document["score"]
        }
        for document in retrieved_documents
    ]

    return {
        "answer": answer,
        "sources": sources,
        "response_time": response_time,
        "token_usage": token_usage
    }


# ============================================================
# 8. HOME PAGE
# ============================================================

@app.route("/", methods=["GET"])
def home():
    """Render the chatbot homepage."""

    get_session_id()

    return render_template("index.html")


# ============================================================
# 9. CHAT API
# ============================================================

@app.route("/api/chat", methods=["POST"])
def chat():
    """Receive a customer question and return an AI response."""

    try:

        data = request.get_json(silent=True) or {}

        question = data.get("message", "")

        # Validate message type.
        if not isinstance(question, str):

            return jsonify({
                "status": "error",
                "message": "The message must be text."
            }), 400

        question = question.strip()

        # Validate empty messages.
        if not question:

            return jsonify({
                "status": "error",
                "message": "Please enter a question."
            }), 400

        # Validate message length.
        if len(question) > MAX_MESSAGE_LENGTH:

            return jsonify({
                "status": "error",
                "message": (
                    f"Your question must be under "
                    f"{MAX_MESSAGE_LENGTH} characters."
                )
            }), 400

        # Get customer session.
        session_id = get_session_id()

        # Retrieve conversation history.
        conversation_history = get_conversation_history(
            session_id
        )

        logger.info(
            "Processing chat request for session: %s",
            session_id
        )

        # Generate RAG response.
        result = generate_answer(
            question,
            conversation_history
        )

        answer = result["answer"]

        # Update conversation history.
        updated_history = conversation_history + [
            {
                "role": "user",
                "content": question
            },
            {
                "role": "assistant",
                "content": answer
            }
        ]

        save_conversation_history(
            session_id,
            updated_history
        )

        # Always return lowercase status.
        return jsonify({
            "status": "success",
            "session_id": session_id,
            "answer": answer,
            "sources": result["sources"],
            "response_time": result["response_time"],
            "token_usage": result["token_usage"],
            "conversation_message_count": len(
                CONVERSATIONS.get(session_id, [])
            )
        }), 200

    except Exception as error:

        logger.exception(
            "Chat request failed: %s",
            error
        )

        return jsonify({
            "status": "error",
            "message": (
                "The chatbot could not process your request. "
                "Please check the server logs and try again."
            )
        }), 500


# ============================================================
# 10. RESET CONVERSATION API
# ============================================================

@app.route("/api/reset", methods=["POST"])
def reset_chat():
    """Clear the current customer's conversation history."""

    try:

        session_id = get_session_id()

        reset_conversation(session_id)

        logger.info(
            "Conversation reset for session: %s",
            session_id
        )

        return jsonify({
            "status": "success",
            "session_id": session_id,
            "message": "Conversation history reset successfully."
        }), 200

    except Exception as error:

        logger.exception(
            "Conversation reset failed: %s",
            error
        )

        return jsonify({
            "status": "error",
            "message": "Unable to reset the conversation."
        }), 500


# ============================================================
# 11. HEALTH CHECK API
# ============================================================

@app.route("/health", methods=["GET"])
def health_check():
    """Basic application health check."""

    return jsonify({
        "status": "success",
        "application": "ShopAssistAI RAG 2.0",
        "message": "Flask application is running."
    }), 200


# ============================================================
# 12. APPLICATION ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def page_not_found(error):

    return jsonify({
        "status": "error",
        "message": "The requested route was not found."
    }), 404


@app.errorhandler(413)
def request_too_large(error):

    return jsonify({
        "status": "error",
        "message": "The submitted request is too large."
    }), 413


@app.errorhandler(500)
def internal_server_error(error):

    logger.exception(
        "Unhandled application error: %s",
        error
    )

    return jsonify({
        "status": "error",
        "message": "An internal server error occurred."
    }), 500


# ============================================================
# 13. RUN APPLICATION LOCALLY
# ============================================================

if __name__ == "__main__":

    port = int(os.getenv("PORT", "5000"))

    logger.info(
        "Starting ShopAssistAI RAG 2.0 on port %s",
        port
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True
    )
