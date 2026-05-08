import boto3
import json
import os

from dotenv import load_dotenv

from boto3 import Session

from opensearchpy import (
    OpenSearch,
    RequestsHttpConnection,
    AWSV4SignerAuth
)

# =========================
# LOAD ENV
# =========================
load_dotenv()

# =========================
# CONFIG
# =========================
region = os.getenv("AWS_REGION", "eu-west-3")
host = os.getenv("OPENSEARCH_HOST")

# =========================
# AUTH AWS (AUTO REFRESH ECS)
# =========================
credentials = Session().get_credentials()

auth = AWSV4SignerAuth(
    credentials,
    region,
    "es"
)

# =========================
# CLIENT OPENSEARCH
# =========================
client = OpenSearch(
    hosts=[{"host": host, "port": 443}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

# =========================
# CLIENT BEDROCK
# =========================
bedrock = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1"
)

# =========================
# EMBEDDING
# =========================
def get_embedding(text):

    try:

        response = bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v1",
            body=json.dumps({
                "inputText": text
            })
        )

        result = json.loads(
            response["body"].read()
        )

        return result["embedding"]

    except Exception as e:

        print(f"❌ Embedding error: {e}")

        return None


# =========================
# RERANK
# =========================
def rerank(query, docs):

    return sorted(
        docs,
        key=lambda d: (
            sum(
                word in d.lower()
                for word in query.lower().split()
            ),
            -len(d)
        ),
        reverse=True
    )


# =========================
# SEARCH
# =========================
def search_docs(query, k=2):

    query_vector = get_embedding(query)

    if query_vector is None:
        return "Aucun contexte trouvé"

    search_query = {
        "size": k,
        "query": {
            "bool": {
                "should": [
                    {
                        "knn": {
                            "embedding": {
                                "vector": query_vector,
                                "k": k
                            }
                        }
                    },
                    {
                        "match": {
                            "text": {
                                "query": query,
                                "boost": 0.3
                            }
                        }
                    }
                ]
            }
        }
    }

    try:

        response = client.search(
            index="rag-index",
            body=search_query
        )

        hits = response["hits"]["hits"]

        docs = [
            hit["_source"]["text"]
            for hit in hits
        ]

        # =========================
        # FILTRAGE SEMANTIQUE
        # =========================
        docs = [
            d for d in docs
            if any(
                word in d.lower()
                for word in query.lower().split()
            )
        ]

        # =========================
        # RERANKING
        # =========================
        docs = rerank(query, docs)

        return "\n\n".join(docs[:2])

    except Exception as e:

        print(f"❌ OpenSearch error: {e}")

        return "Erreur lors de la recherche documentaire"


# =========================
# GENERATION
# =========================
def generate_answer(question, context):

    try:

        response = bedrock.invoke_model(
            modelId="amazon.nova-lite-v1:0",
            body=json.dumps({
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": f"""
Tu es un assistant expert en conformité bancaire.

Réponds uniquement avec les informations du contexte.

Instructions :
- Réponse claire et concise
- Appuie ta réponse sur les éléments du contexte
- N'invente rien
- Si l'information est absente → "Je ne sais pas"

Contexte:
{context}

Question:
{question}

Réponse:
"""
                            }
                        ]
                    }
                ],
                "inferenceConfig": {
                    "maxTokens": 300,
                    "temperature": 0.3
                }
            })
        )

        result = json.loads(
            response["body"].read()
        )

        return result["output"]["message"]["content"][0]["text"]

    except Exception as e:

        print(f"❌ Nova generation error: {e}")

        return f"Fallback:\n{context[:500]}"


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    question = input("🔎 Pose ta question : ")

    context = search_docs(question)

    print("\n📚 Contexte trouvé:\n")
    print(context)

    answer = generate_answer(question, context)

    print("\n🤖 Réponse:\n")
    print(answer)