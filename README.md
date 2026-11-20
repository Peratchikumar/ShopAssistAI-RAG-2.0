# 🛍️ ShopAssistAI RAG 2.0
## Advanced E-Commerce Customer Support Chatbot

An AI-powered e-commerce customer support chatbot built using **Retrieval-Augmented Generation (RAG)**. It retrieves relevant information from PDF documents and a product catalogue to generate accurate, contextual, and helpful responses to customer queries.

---

## 📌 Project Overview

ShopAssistAI RAG 2.0 is an intelligent customer support solution designed to answer common e-commerce questions related to:

- Frequently Asked Questions (FAQs)
- Product information and pricing
- Return and refund policies
- Shipping and delivery
- Payment methods
- Warranty guidance
- Order management
- Order cancellation
- Product exchanges
- Privacy policies
- Customer support escalation

The application combines semantic search, vector embeddings, ChromaDB, and OpenAI language models to provide context-aware customer support.

## 🎯 Key Features

- **Knowledge Base Management:** Loads multiple PDF documents and CSV product catalogues.
- **Document Processing:** Extracts, cleans, and splits documents into smaller text chunks.
- **Text Embeddings:** Generates vector embeddings using OpenAI's embedding model.
- **Vector Database:** Stores and retrieves document embeddings using ChromaDB.
- **Semantic Search:** Retrieves relevant information based on the meaning of customer queries.
- **Optimized Retrieval:** Uses keyword-based scoring and product-focused retrieval optimization.
- **Hybrid Retrieval:** Routes product-related questions through optimized retrieval and other questions through semantic retrieval.
- **AI Response Generation:** Generates contextual answers using the OpenAI Responses API.
- **Conversation Memory:** Maintains conversation history and supports follow-up questions.
- **RAG Evaluation:** Evaluates retrieval and response quality using RAGAS.
- **Web Application:** Flask-based application with HTML, CSS, and JavaScript.
- **Cloud Deployment:** Designed for deployment on Render.com.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Backend Framework | Flask |
| Frontend | HTML, CSS, JavaScript |
| Large Language Model | OpenAI API |
| Embedding Model | text-embedding-3-small |
| Vector Database | ChromaDB |
| PDF Processing | PyPDF |
| Data Processing | Pandas, NumPy |
| Progress Monitoring | tqdm |
| RAG Evaluation | RAGAS |
| Environment Management | python-dotenv |
| Production Server | Gunicorn |
| Deployment Platform | Render.com |
| Version Control | Git and GitHub |

## 🧠 RAG Architecture

The application follows a Retrieval-Augmented Generation workflow.

1. **Document Loading:** Loads PDF knowledge-base documents and product information from CSV.
2. **Text Preprocessing:** Extracts and cleans document content.
3. **Text Chunking:** Splits documents into smaller chunks.
4. **Embedding Generation:** Converts text chunks into numerical vectors.
5. **Vector Storage:** Stores embeddings and metadata in ChromaDB.
6. **Query Processing:** Receives and processes customer questions.
7. **Document Retrieval:** Retrieves relevant knowledge-base chunks using semantic search and optimized retrieval.
8. **Context Preparation:** Combines retrieved information with the customer query and conversation history.
9. **Response Generation:** Uses the OpenAI model to generate a contextual response.
10. **Customer Interaction:** Returns the generated answer through the chatbot interface.

## ⚙️ RAG Configuration

| Parameter | Configuration |
|---|---|
| Chunk Size | 180 words |
| Chunk Overlap | 35 words |
| Embedding Model | text-embedding-3-small |
| Embedding Dimensions | 384 |
| Default Retrieval Top-K | 4 |
| Vector Database | ChromaDB |
| Retrieval Strategy | Semantic + Optimized + Hybrid |
| Generation API | OpenAI Responses API |

## 📂 Project Structure

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
│   └── .gitkeep
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
├── app.py
├── requirements.txt
├── Procfile
├── .gitignore
├── README.md
│
└── ShopAssistAI RAG 2.0 – Advanced E-Commerce Customer Support Chatbot.ipynb
```

## 🚀 Installation and Setup

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Navigate to the Project Directory

```bash
cd ShopAssistAI-RAG-2.0
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows:**

```bash
venv\Scripts\activate
```

**Mac/Linux:**

```bash
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Configure Environment Variables

Create a `.env` file in the project root.

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=your_supported_openai_model
CHROMA_PATH=./chroma_db
```

Replace the placeholder values with your actual credentials and supported model ID.

**Important:** Never upload your `.env` file or expose your API key on GitHub.

### Step 6: Run the Flask Application

```bash
python app.py
```

Open your browser:

```text
http://127.0.0.1:5000
```

---

## 📊 RAGAS Evaluation

The project evaluates RAG performance using four evaluation metrics.

| Metric | Purpose |
|---|---|
| Faithfulness | Measures whether generated answers are supported by retrieved context. |
| Answer Relevancy | Measures how relevant the answer is to the customer's question. |
| Context Precision | Measures the relevance of retrieved context. |
| Context Recall | Measures whether the retrieved context contains the information needed to answer the question. |

The evaluation workflow compares:

1. Baseline Retrieval
2. Optimized Retrieval
3. Hybrid Retrieval

Evaluation results are analyzed and saved for performance comparison.

## ☁️ Deployment on Render.com

The Flask application is configured for production deployment using Gunicorn.

### Deployment Configuration

| Setting | Value |
|---|---|
| Platform | Render.com |
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `gunicorn app:app` |
| Health Check Path | `/health` |

### Environment Variables

Configure the following variables in Render:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=your_supported_openai_model
CHROMA_PATH=/var/data/vectorstore
```

### Persistent Storage

Configure a Render Persistent Disk to store the ChromaDB database.

- Mount Path: `/var/data`
- Database Path: `/var/data/vectorstore`

This helps preserve the vector database across application restarts and redeployments.

**Note:** The deployment application must initialize or load the ChromaDB collection from this persistent location. The PDF and CSV knowledge-base files must also be included in the deployed project.

## 🔐 Security Considerations

- Store API keys in environment variables.
- Never expose confidential credentials in source code.
- Add `.env` to `.gitignore`.
- Validate incoming customer queries.
- Avoid exposing internal system prompts.
- Use persistent storage for production vector data.

## 🔮 Future Enhancements

- Advanced conversational memory.
- User authentication and customer profiles.
- Order tracking integration.
- Live inventory integration.
- Multi-language customer support.
- Human agent escalation.
- Analytics dashboard.
- Improved hybrid retrieval and reranking.
- Production-ready persistent conversation storage.

## 👨‍💻 Author

**K. Peratchi Kumar**

AI & Machine Learning Enthusiast

## 📄 License

This project is developed for educational and portfolio purposes.

---

**ShopAssistAI RAG 2.0 – Intelligent, Context-Aware E-Commerce Customer Support Powered by Retrieval-Augmented Generation.**