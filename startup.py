import os
import json
import chromadb

def get_simple_embedding(text, size=384):
    """Lightweight embedding without sentence-transformers"""
    import hashlib
    import math
    words = text.lower().split()
    vector = [0.0] * size
    for i, word in enumerate(words):
        hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
        for j in range(min(5, size)):
            idx = (hash_val + i * 7 + j * 13) % size
            vector[idx] += 1.0 / (i + 1)
    magnitude = math.sqrt(sum(x*x for x in vector)) or 1.0
    return [x / magnitude for x in vector]

def build_db_if_needed():
    db_path = "./geeta_db"
    if os.path.exists(db_path):
        try:
            client = chromadb.PersistentClient(path=db_path)
            collection = client.get_collection("shlokas")
            if collection.count() > 0:
                print(f"ChromaDB already exists. Skipping rebuild.")
                return
        except:
            pass

    print("Building ChromaDB from gita_dataset.json...")
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
        embedding = get_simple_embedding(search_text)
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

    print(f"ChromaDB built with {len(dataset)} shlokas!")

if __name__ == "__main__":
    build_db_if_needed()