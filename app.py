from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import os
import time
import requests
from pydantic import BaseModel
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# -------------------------------------------------------------------
# Load env
# -------------------------------------------------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

if not PINECONE_API_KEY or not PINECONE_INDEX:
    raise RuntimeError("Missing Pinecone environment variables")

# -------------------------------------------------------------------
# Init Pinecone
# -------------------------------------------------------------------
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

existing_indexes = [i["name"] for i in pinecone_client.list_indexes()]

# ⚠️ LOCAL EMBEDDING DIMENSION = 384
EMBED_DIM = 384

if PINECONE_INDEX not in existing_indexes:
    pinecone_client.create_index(
        name=PINECONE_INDEX,
        dimension=EMBED_DIM,
        metric="cosine",
        spec={
            "serverless": {
                "cloud": "aws",
                "region": "us-east-1"
            }
        }
    )
    time.sleep(10)

index = pinecone_client.Index(PINECONE_INDEX)

# -------------------------------------------------------------------
# Local embedding model (FREE, NO API)
# -------------------------------------------------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------
app = FastAPI(title="Local Embedding RAG API")

@app.get("/")
def home():
    return {"status": "Server running (local embeddings, 384-dim)"}

# -------------------------------------------------------------------
# Models
# -------------------------------------------------------------------
class IngestRequest(BaseModel):
    id: str
    text: str

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

# -------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------
def create_embedding(text: str):
    if not text or not text.strip():
        raise ValueError("Text is empty")
    return embedding_model.encode(text).tolist()

# -------------------------------------------------------------------
# Ingest
# -------------------------------------------------------------------
@app.post("/ingest")
def ingest_data(payload: IngestRequest):
    try:
        vector = create_embedding(payload.text)

        index.upsert([
            {
                "id": payload.id,
                "values": vector,
                "metadata": {
                    "text": payload.text
                }
            }
        ])

        return {
            "status": "stored",
            "id": payload.id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------------------------
# Query
# -------------------------------------------------------------------
@app.post("/query")
def query_data(payload: QueryRequest):
    try:
        if not payload.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        query_vector = create_embedding(payload.query)

        result = index.query(
            vector=query_vector,
            top_k=payload.top_k,
            include_metadata=True
        )

        matches = []
        for match in result.matches:
            matches.append({
                "id": match.id,
                "score": match.score,
                "text": match.metadata.get("text", "")
            })

        return {
            "query": payload.query,
            "matches": matches
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/media-generate")
def generate_image(payload: dict):
    """
    Expected payload:
    {
      "type": "image",
      "prompt": "NEET students studying online",
      "orientation": "landscape"
    }
    """

    media_type = payload.get("type")
    prompt = payload.get("prompt")

    if media_type != "image":
        return {"error": "Only image generation supported"}

    if not prompt:
        return {"error": "prompt is required"}

    headers = {
        "Authorization": os.getenv("PEXELS_API_KEY")
    }

    params = {
        "query": prompt,
        "per_page": 1,
        "orientation": payload.get("orientation", "landscape")
    }

    response = requests.get(
        "https://api.pexels.com/v1/search",
        headers=headers,
        params=params,
        timeout=10
    )

    if response.status_code != 200:
        return {"error": "Failed to fetch image from Pexels"}

    data = response.json()

    if not data.get("photos"):
        return {"error": "No images found"}

    photo = data["photos"][0]

    return {
        "media_type": "image",
        "image_url": photo["src"]["large"],
        "photographer": photo["photographer"],
        "source": "pexels"
    }


