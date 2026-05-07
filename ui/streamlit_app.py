import streamlit as st
import requests

# API_URL = "http://api:8000/ask"
# API_URL = "http://13.39.84.130:8000/ask"
API_URL = "http://rag-alb-291336896.eu-west-3.elb.amazonaws.com:8000/ask"


st.title("🤖 RAG Assistant AWS")

question = st.text_input("Pose ta question ici...")

if st.button("Envoyer"):

    response = requests.post(API_URL, json={"question": question})

    if response.status_code == 200:
        data = response.json()

        st.subheader("🧠 Réponse")
        st.write(data["answer"])

        st.subheader("📚 Contexte")
        st.write(data["context"])
    else:
        st.error("Erreur API")
