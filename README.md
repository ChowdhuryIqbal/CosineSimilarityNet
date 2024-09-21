# CosineSimilarityNet

**CosineSimilarityNet** is an AI-driven query system that provides real-time responses by matching user questions to predefined knowledge using advanced text embeddings and cosine similarity. It leverages OpenAI’s powerful language models deployed via Azure for seamless and scalable performance.

## Features

- **Real-time Query Response**: Users can input queries and receive immediate answers.
- **Cosine Similarity Matching**: Uses cosine similarity to find the most relevant question match from a predefined template.
- **Text Embeddings**: Employs OpenAI's `text-embedding-ada-002` model to generate embeddings of both questions and queries.
- **Custom GPT-4 Responses**: When no match is found, the system generates a response using GPT-4 for a more dynamic experience.
- **Scalability**: Built with Azure OpenAI for scalable, enterprise-level performance.
- **Interactive UI**: Uses Streamlit for a simple and user-friendly interface.

## Tech Stack

- **Azure OpenAI**: Provides language models and embedding generation.
- **Cosine Similarity**: Measures the similarity between user queries and predefined questions.
- **Streamlit**: Front-end interface for user interactions.
- **scikit-learn**: For computing cosine similarity.
- **Python**: Primary programming language for back-end logic.

## Installation
`pip install -r requirements.txt`
- .env file contains:
```bash
EMBEDDING_MODEL_NAME=
AZURE_OPENAI_KEY=
AZURE_OPENAI_ENDPOINT=
```
## Run in console
`streamlit run main.py`

### Prerequisites

Ensure you have the following tools installed:

- Python 3.8+
- pip
- Azure OpenAI access
- API keys stored in an `.env` file

### Clone the Repository

```bash
git clone https://github.com/ChowdhuryIqbal/cosinesimilaritynet.git
cd cosinesimilaritynet
