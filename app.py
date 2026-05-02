from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from prompt import SYSTEM_PROMPT
from startup import build_db_if_needed
build_db_if_needed()

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./geeta_db")
collection = client.get_collection("shlokas")
import os
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []

@app.get("/")
def home():
    return FileResponse("index.html")

@app.post("/chat")
def chat(req: ChatRequest):
    import traceback
    try:
        # Step 1: Find matching shlokas
        query_embedding = model.encode(req.message).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        shlokas_context = ""
        for i, meta in enumerate(results["metadatas"][0]):
            shlokas_context += f"""
Shloka {i+1}: {meta['id']}
Sanskrit: {meta['sanskrit']}
Translation: {meta['translation_english']}
Themes: {meta['themes']}
Neuroscience: {meta['neuroscience_link']}
"""

        # Step 2: Build messages with full history
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        print(f"History received: {len(req.history)} messages")

        # Add conversation history (last 6 messages to save tokens)
        for msg in req.history[-6:]:
            messages.append({"role": msg.role, "content": msg.content})

        # Add current message with shloka context
        user_prompt = f"""The person says: "{req.message}"

Relevant Bhagavad Gita shlokas:
{shlokas_context}

Respond as GeetaAI using the most fitting shloka."""

        messages.append({"role": "user", "content": user_prompt})

        # Step 3: Try big model first, fall back to smaller
        response = None
        for model_name in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
            try:
                response = groq_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=500
                )
                break
            except Exception:
                continue

        if response is None:
            return {"response": "GeetaAI is resting for now. Please try again in a few minutes. 🙏"}

        return {"response": response.choices[0].message.content}

    except Exception as e:
        error_details = traceback.format_exc()
        print("FULL ERROR:\n", error_details)
        return {"response": f"Something went wrong: {str(e)}"}