import json

print("Loading files...")
with open("gita/data/verse.json", encoding="utf-8") as f:
    verses = json.load(f)

with open("gita/data/translation.json", encoding="utf-8") as f:
    translations = json.load(f)

with open("gita/data/commentary.json", encoding="utf-8") as f:
    commentaries = json.load(f)

# Group translations by verse_id
print("Grouping translations...")
trans_by_verse = {}
for t in translations:
    vid = t["verse_id"]
    if vid not in trans_by_verse:
        trans_by_verse[vid] = {}
    lang = t.get("lang", "unknown")
    # Keep one English and one Hindi per verse
    if lang == "english" and "english" not in trans_by_verse[vid]:
        trans_by_verse[vid]["english"] = t["description"].strip()
    elif lang == "hindi" and "hindi" not in trans_by_verse[vid]:
        trans_by_verse[vid]["hindi"] = t["description"].strip()

# Group commentaries by verse_id
print("Grouping commentaries...")
comm_by_verse = {}
for c in commentaries:
    vid = c["verse_id"]
    if vid not in comm_by_verse:
        comm_by_verse[vid] = []
    comm_by_verse[vid].append({
        "author": c.get("authorName", ""),
        "text": c.get("description", "").strip()
    })

# Merge everything into one dataset
print("Merging into GeetaAI dataset...")
dataset = []
for verse in verses:
    vid = verse["id"]
    chapter = verse["chapter_number"]
    verse_num = verse["verse_number"]

    entry = {
        "id": f"BG_{chapter}_{verse_num}",
        "chapter": chapter,
        "verse": verse_num,
        "sanskrit": verse.get("text", "").strip(),
        "transliteration": verse.get("transliteration", "").strip(),
        "word_meanings": verse.get("word_meanings", "").strip(),
        "translation_english": trans_by_verse.get(vid, {}).get("english", ""),
        "translation_hindi": trans_by_verse.get(vid, {}).get("hindi", ""),
        "commentaries": comm_by_verse.get(vid, []),

        # --- YOU WILL FILL THESE IN NEXT STEP ---
        "themes": [],
        "life_scenarios": [],
        "neuroscience_link": "",
        "emotion_intensity": ""
    }
    dataset.append(entry)

# Save
with open("gita_dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"\nDone! {len(dataset)} shlokas saved to gita_dataset.json")

# Preview first entry
print("\nPreview of BG 1.1:")
print(json.dumps(dataset[0], indent=2, ensure_ascii=False)[:800])