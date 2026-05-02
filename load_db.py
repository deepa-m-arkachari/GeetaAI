import json
import chromadb
from sentence_transformers import SentenceTransformer

print("Loading model for text embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading your gita_dataset.json...")
with open("gita_dataset.json", encoding="utf-8") as f:
    dataset = json.load(f)

client = chromadb.PersistentClient(path="./geeta_db")
try:
    client.delete_collection("shlokas")
except:
    pass
collection = client.create_collection("shlokas")

print(f"Loading {len(dataset)} shlokas into ChromaDB...")

for i, s in enumerate(dataset):
    search_text = (
        s.get("translation_english", "") + " " +
        " ".join(s.get("themes", [])) + " " +
        " ".join(s.get("life_scenarios", [])) + " " +
        s.get("neuroscience_link", "")
    )
    embedding = model.encode(search_text).tolist()
    collection.add(
        documents=[search_text],
        embeddings=[embedding],
        metadatas=[{
            "id": s["id"],
            "chapter": str(s["chapter"]),
            "verse": str(s["verse"]),
            "sanskrit": s.get("sanskrit", "")[:300],
            "translation_english": s.get("translation_english", "")[:500],
            "themes": ", ".join(s.get("themes", [])),
            "neuroscience_link": s.get("neuroscience_link", ""),
            "life_scenarios": ", ".join(s.get("life_scenarios", []))
        }],
        ids=[s["id"]]
    )
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(dataset)} loaded...")

print(f"\nDone! ChromaDB is ready at ./geeta_db")