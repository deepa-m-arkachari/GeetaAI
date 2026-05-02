import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

def build_db_if_needed():
    db_path = "./geeta_db"
    
    # Check if DB already exists and has data
    if os.path.exists(db_path):
        try:
            client = chromadb.PersistentClient(path=db_path)
            collection = client.get_collection("shlokas")
            count = collection.count()
            if count > 0:
                print(f"ChromaDB already exists with {count} shlokas. Skipping rebuild.")
                return
        except:
            pass
    
    print("Building ChromaDB from gita_dataset.json...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    with open("gita_dataset.json", encoding="utf-8") as f:
        dataset = json.load(f)
    
    client = chromadb.PersistentClient(path=db_path)
    try:
        client.delete_collection("shlokas")
    except:
        pass
    collection = client.create_collection("shlokas")
    
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
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(dataset)} loaded...")
    
    print(f"ChromaDB built successfully with {len(dataset)} shlokas!")

if __name__ == "__main__":
    build_db_if_needed()