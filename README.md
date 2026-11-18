# ShopAssistAI RAG 2.0 – Advanced E-Commerce Customer Support Chatbot

An AI-powered e-commerce customer support chatbot built using **Retrieval-Augmented Generation (RAG)**. It uses a knowledge base of product information and customer support documents to generate context-aware responses to customer queries.

The project combines **ChromaDB, Sentence Transformers, hybrid retrieval optimization, OpenAI API, Flask, and RAGAS evaluation** to deliver an intelligent customer support experience.

---

## Project Overview

ShopAssistAI RAG 2.0 is designed to help customers find information about:

- Product information and pricing
- Return and exchange policies
- Refund procedures and timelines
- Shipping and delivery information
- Payment methods
- Warranty guidance
- Order management
- Order cancellation
- Customer complaints and support escalation
- Privacy and customer information protection

The chatbot retrieves relevant information from its knowledge base before generating an answer using a Large Language Model (LLM).

**Note:** The included store documents and product catalogue contain synthetic demonstration data. They do not represent verified policies, live inventory, or real customer orders.

---

## Key Features

- **Retrieval-Augmented Generation (RAG):** Retrieves relevant knowledge-base content to support responses.
- **Hybrid Retrieval Optimization:** Combines semantic similarity and keyword relevance to improve document retrieval.
- **Vector Database:** Uses ChromaDB for persistent vector storage and similarity search.
- **Semantic Embeddings:** Uses Sentence Transformers to convert text into numerical embeddings.
- **Context-Aware Responses:** Uses retrieved documents and conversation history when generating answers.
- **Conversational Interface:** Provides a web-based chatbot interface using Flask, HTML, CSS, and JavaScript.
- **Source Transparency:** Displays knowledge-base source documents associated with chatbot responses.
- **RAGAS Evaluation:** Evaluates answer faithfulness, answer relevancy, context precision, and context recall.
- **Conversation Management:** Supports starting and resetting conversations.
- **Deployment Ready:** Includes the application configuration required for deployment on Render.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python 3.11 |
| LLM API | OpenAI API |
| RAG Framework | Custom RAG pipeline |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Database | ChromaDB |
| Retrieval | Semantic and keyword-based hybrid retrieval |
| Evaluation | RAGAS |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Data Processing | Pandas, NumPy |
| PDF Processing | PyPDF |
| Environment Management | Python virtual environment, python-dotenv |
| Deployment | Render |
| Version Control | Git and GitHub |

---

## System Architecture

```text
              USER
                |
                v
       FLASK WEB APPLICATION
                |
                v
        USER QUESTION INPUT
                |
                v
       QUERY PROCESSING
                |
                v
       HYBRID RETRIEVAL
         /           \
        v             v
  SEMANTIC SEARCH  KEYWORD SEARCH
        \             /
         \           /
                |
                v
       RETRIEVAL RANKING
                |
                v
          CHROMADB
                |
                v
      RELEVANT DOCUMENTS
                |
                v
       CONTEXT GENERATION
                |
                v
          OPENAI API
                |
                v
       GENERATED RESPONSE
                |
                v
       SOURCE REFERENCES
                |
                v
       CHATBOT INTERFACE
                |
                v
              USER
```

### RAG Workflow

1. Collect and load knowledge-base documents.
2. Extract text from PDF and CSV files.
3. Split document content into manageable chunks.
4. Generate embeddings using Sentence Transformers.
5. Store document chunks and embeddings in ChromaDB.
6. Accept customer questions through the Flask application.
7. Retrieve relevant documents using hybrid retrieval.
8. Rank retrieved documents according to relevance.
9. Construct the context using the retrieved information.
10. Send the question and context to the OpenAI API.
11. Generate a context-aware response.
12. Display the answer and its knowledge-base sources.

---

## Project Structure

```text
ShopAssistAI-RAG-2.0/
│
├── data/
│   ├── 01_FAQs.pdf
│   ├── 02_Return_Policy.pdf
│   ├── 03_Refund_Policy.pdf
│   ├── 04_Shipping_Policy.pdf
│   ├── 05_Payment_Policy.pdf
│   ├── 06_Warranty_Guidance.pdf
│   ├── 07_Order_Management_Guide.pdf
│   ├── 08_Cancellation_Policy.pdf
│   ├── 09_Exchange_Policy.pdf
│   ├── 10_Privacy_Policy.pdf
│   ├── 11_Customer_Support_Escalation_Guide.pdf
│   └── 12_Product_Catalogue.csv
│
├── evaluation/
│   ├── ragas_baseline_results.csv
│   └── retrieval_evaluation.csv
│
├── outputs/
│   └── .gitkeep
│
├── static/
│   ├── style.css
│   └── script.js
│
├── templates/
│   └── index.html
│
├── vectorstore/
│   └── ChromaDB persistent database files
│
├── app.py
├── requirements.txt
├── Procfile
├── .gitignore
│
└── ShopAssistAI RAG 2.0 – Advanced E-Commerce Customer Support Chatbot.ipynb
```

The vector database is generated from the knowledge-base documents. Its availability and initialization must be configured correctly in the deployment environment.

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Peratchikumar/ShopAssistAI-RAG-2.0.git
```

### 2. Navigate to the Project Directory

```bash
cd ShopAssistAI-RAG-2.0
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure Environment Variables

Create a `.env` file in the project root directory.

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-5.6-luna
FLASK_SECRET_KEY=replace_with_a_secure_random_secret
COOKIE_SECURE=false
```

Replace the placeholder values with your actual credentials and configuration.

The configured OpenAI model identifier must be available to your API account. Never publish your API key or secret values on GitHub.

### 7. Run the Flask Application

```bash
python app.py
```

Open the following address in your browser:

```text
http://127.0.0.1:5000/
```

---

## Model and Vector Database

### Embedding Model

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embedding model converts knowledge-base text and customer queries into vector representations for semantic retrieval.

### Vector Database

```text
ChromaDB
```

ChromaDB stores document embeddings, text chunks, and metadata to support similarity-based retrieval.

### Language Model

```text
OpenAI API
```

The OpenAI API generates responses using the retrieved knowledge-base context and conversation history.

---

## Retrieval Optimization

The project uses a hybrid retrieval approach to improve the relevance of retrieved documents.

The retrieval process combines:

- Semantic similarity scoring
- Keyword relevance scoring
- Candidate document retrieval
- Final relevance ranking
- Top-K document selection

The objective is to retrieve the most relevant knowledge-base content before sending context to the language model.

Retrieval quality should be measured using a separate evaluation dataset and compared against the baseline.

---

## RAGAS Evaluation

RAGAS is used to evaluate the quality of the RAG pipeline.

The evaluation includes four important metrics:

| Metric | Purpose |
|---|---|
| Faithfulness | Measures whether the generated answer is supported by the retrieved context. |
| Answer Relevancy | Measures how relevant the generated answer is to the user's question. |
| Context Precision | Measures whether relevant information is ranked appropriately in the retrieved context. |
| Context Recall | Measures whether the retrieved context contains the information needed to answer the question. |

### Baseline Evaluation

The initial baseline evaluation produced the following results:

| Evaluation Metric | Baseline Score |
|---|---:|
| Faithfulness | 93.89% |
| Answer Relevancy | 60.95% |
| Context Precision | 80.58% |
| Context Recall | 100% |

These are the recorded baseline results for the project's initial evaluation run. They are not guaranteed to represent the current optimized system.

### Evaluation Files

```text
evaluation/
├── ragas_baseline_results.csv
└── retrieval_evaluation.csv
```

The retrieval evaluation file records the expected source documents and the documents retrieved by the retrieval pipeline.

---

## Flask API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Loads the chatbot interface. |
| `/api/chat` | POST | Processes a customer question and returns a chatbot response. |
| `/api/reset` | POST | Resets the current conversation. |
| `/health` | GET | Provides an application health-check endpoint. |

### Example Chat Request

```json
{
  "message": "What is your return policy?"
}
```

The API response format depends on the Flask backend implementation.

---

## Deployment – Render

The application is designed for deployment using Render.

### Deployment Configuration

| Setting | Value |
|---|---|
| Service Type | Web Service |
| Runtime | Python |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `gunicorn --workers 1 --threads 2 --timeout 300 app:app` |
| Health Check Path | `/health` |

### Environment Variables

Configure the following variables in the Render dashboard:

```text
OPENAI_API_KEY
OPENAI_MODEL
FLASK_SECRET_KEY
COOKIE_SECURE
```

Use `COOKIE_SECURE=true` when serving the application over HTTPS.

**Important deployment considerations:**

- Do not upload `.env` or expose API credentials.
- Ensure the ChromaDB database is available before accepting chatbot requests.
- If the vectorstore is excluded from Git, provide a deployment-time process to rebuild it from the files in `data/`, or configure suitable persistent storage.
- Render's ordinary application filesystem is not persistent across all redeployments and restarts.
- Use a persistent disk or a reliable database strategy if the application requires persistent vector storage.
- The current in-memory conversation storage is suitable for a demonstration, not reliable multi-worker or long-term production sessions.
- Test the deployed health endpoint and chatbot API before making the application publicly available.

---

## Limitations and Future Improvements

### Current Limitations

- The knowledge base contains synthetic demonstration information.
- Live order tracking is not connected to an external order-management system.
- Real payment, refund, cancellation, and exchange operations are not integrated.
- Product availability is not synchronized with a live inventory system.
- In-memory conversation storage does not provide durable session persistence.
- Evaluation results depend on the evaluation dataset and model configuration.

### Future Improvements

- Integrate live e-commerce product catalogues.
- Connect secure order-management APIs.
- Implement customer authentication.
- Add persistent conversation memory.
- Improve hybrid retrieval and reranking.
- Expand the RAG evaluation dataset.
- Introduce automated regression testing.
- Add monitoring, logging, and rate limiting.
- Implement production-grade error handling and security controls.

---

## Security Considerations

- Store API keys in environment variables.
- Never commit `.env` files or credentials.
- Use a secure Flask secret key.
- Use HTTPS in production.
- Validate incoming API requests.
- Apply rate limiting and abuse protection before public release.
- Do not collect or expose sensitive customer information unnecessarily.
- Do not claim to execute refunds, cancellations, or order changes without verified external integrations.

---

## Disclaimer

ShopAssistAI RAG 2.0 is an educational and demonstration project showcasing Retrieval-Augmented Generation and AI-powered customer support.

The supplied policies and product information are synthetic demonstration content. They must not be treated as verified real-store policies, legal advice, live product availability, or actual order information.

---

## Author

**K. Peratchi Kumar**

GitHub: [@Peratchikumar](https://github.com/Peratchikumar)

---

## Acknowledgements

Built using open-source Python libraries, ChromaDB, Sentence Transformers, RAGAS, Flask, and the OpenAI API.

**ShopAssistAI RAG 2.0 – Intelligent E-Commerce Customer Support Powered by Retrieval-Augmented Generation.**