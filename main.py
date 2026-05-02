from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import chromadb
import hashlib
import math
import os
from groq import Groq
from prompt import SYSTEM_PROMPT
from startup import build_db_if_needed

build_db_if_needed()

app = FastAPI()
client = chromadb.PersistentClient(path="./geeta_db")
collection = client.get_collection("shlokas")
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def get_simple_embedding(text, size=384):
    words = text.lower().split()
    vector = [0.0] * size
    for i, word in enumerate(words):
        hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
        for j in range(min(5, size)):
            idx = (hash_val + i * 7 + j * 13) % size
            vector[idx] += 1.0 / (i + 1)
    magnitude = math.sqrt(sum(x*x for x in vector)) or 1.0
    return [x / magnitude for x in vector]

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
        query_embedding = get_simple_embedding(req.message)
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

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in req.history[-6:]:
            messages.append({"role": msg.role, "content": msg.content})

        user_prompt = f"""The person says: "{req.message}"

Relevant Bhagavad Gita shlokas:
{shlokas_context}

Respond as GeetaAI using the most fitting shloka."""

        messages.append({"role": "user", "content": user_prompt})

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
            return {"response": "GeetaAI is resting. Please try again in a few minutes. 🙏"}

        return {"response": response.choices[0].message.content}

    except Exception as e:
        print("ERROR:", traceback.format_exc())
        return {"response": f"Something went wrong: {str(e)}"}