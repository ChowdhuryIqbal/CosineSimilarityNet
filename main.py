import os
import streamlit as st
from openai import AzureOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load environment variables
# load_dotenv()

# Template questions for the agent
RESPONSE_TEMPLATE = {
    "What does the Quantum Data Verification Agent (Q-DVA) do?":
    "Q-DVA autonomously verifies the integrity of massive datasets in real-time, reducing human error, and ensuring seamless data synchronization across distributed systems.",
    "What does the HyperClaims Processing Module (HCM) do?":
    "HCM accelerates the processing of complex claims and transactions using advanced algorithms, cutting down manual intervention and ensuring precision in high-volume workflows.",
    "How does the Nano-Payment Reconciliation Unit (NPRU) work?":
    "NPRU automates payment reconciliation across microtransactions, ensuring near-instant validation and settlement in complex financial ecosystems.",
    "Tell me about NebulaTech's AI Modules.":
    "NebulaTech offers a range of AI-driven automation modules that redefine how enterprises manage data, transactions, and payments. These include Quantum Data Verification (Q-DVA), HyperClaims Processing (HCM), and Nano-Payment Reconciliation (NPRU), among others.",
    "What are the benefits of using NebulaTech's AI modules?":
    "NebulaTech’s AI modules drastically lower operational overhead, boost processing speeds, and minimize errors in handling massive data and financial transactions, enabling companies to scale effortlessly."
}

# Similarity threshold; cosine similarity spans from -1 to +1
similarity_threshold = 0.8

# Azure OpenAI client initialized
oai_client = AzureOpenAI(api_key=os.getenv('AZURE_OPENAI_KEY'),
                         api_version="2023-05-15",
                         azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'))


# Generate OpenAI text-embedding-ada-002 embeddings; dimension 1536
def generate_embedding(input_text):

    # oai_client.embeddings.create(model="text-embedding-ada-002", input=chunk)
    embedding_object = oai_client.embeddings.create(
        model=os.getenv('EMBEDDING_MODEL_NAME'), input=input_text)

    embedding_vector = embedding_object.dict()['data'][0]['embedding']
    return embedding_vector


# Measuring similarity between question and user query embedding with cosine similarity measure
def similarity_measure(question_embeddings, user_query_embedding, user_query):
    # Varialbe to keep track of the similarity measure, for cosine similarity -1 is the minimum
    least_similar = -1

    # Keeping track of the most similar question
    most_similar_question = None

    # Iterate through the dictionairy of question embedding to measure similarity
    for question, question_embedding in question_embeddings.items():
        similarity_measure = cosine_similarity([question_embedding],
                                               [user_query_embedding])

        # Find the most similar question based on the user query
        if similarity_measure >= least_similar:
            most_similar_measure = similarity_measure
            most_similar_question = question

    # Return the best matched question
    if most_similar_measure > similarity_threshold:
        return RESPONSE_TEMPLATE[most_similar_question]
    else:
        # If threshold check fails return a generic response
        generic_response = oai_client.chat.completions.create(
            model='gpt-4',
            temperature=0.3,
            messages=[{
                "role":
                "system",
                "content":
                "You are a helpful agent for patient's queries. Do not answer outside of the context provided."
            }, {
                "role":
                "user",
                "content":
                f"Context:\n{RESPONSE_TEMPLATE}\n\nQuestion: {user_query}"
            }])
        return generic_response.choices[0].message.content


# Getting embedding of the user query for the similarity measure
def process_user_query(question_embeddings, user_query):
    user_query_embedding = generate_embedding(user_query)

    return similarity_measure(question_embeddings, user_query_embedding,
                              user_query)


def main():
    st.title("Thoughtful AI Support Agent")
    st.write(
        "Welcome! How can I assist you today with information about Thoughtful AI's Agents?"
    )

    user_query = st.text_input("User Query:")
    submit_button = st.button("Submit Query")

    question_embeddings = {}

    for question in RESPONSE_TEMPLATE.keys():
        question_embeddings[question] = generate_embedding(question)

    if user_query or submit_button:
        st.write(process_user_query(question_embeddings, user_query))


if __name__ == "__main__":
    main()
