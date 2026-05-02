import json
import time
from groq import Groq

# ── CONFIG ──────────────────────────────────────────────
import os
API_KEY = os.environ.get("GROQ_API_KEY", "")   # replace this
# ────────────────────────────────────────────────────────

client = Groq(api_key=API_KEY)

def tag_shloka(shloka):
    prompt = f"""You are building GeetaAI — a compassionate mental health chatbot grounded in Bhagavad Gita wisdom.

Analyze this shloka and return ONLY a valid JSON object. No explanation, no markdown, no code fences.

Shloka ID: {shloka['id']}
Sanskrit: {shloka['sanskrit']}
English Translation: {shloka['translation_english']}

Return exactly this JSON structure:
{{
  "themes": ["4-8 emotional themes from: grief, anxiety, fear, duty, attachment, anger, loneliness, purpose, failure, ego, forgiveness, self_worth, overthinking, depression, identity, comparison, career_failure, relationship_pain, family_conflict, existential_emptiness"],
  "life_scenarios": ["4-6 real modern life situations this shloka applies to, e.g. lost my job, going through breakup, feeling like a failure, parents fighting, scared of death"],
  "neuroscience_link": "1-2 sentences connecting this shloka to modern brain science — mention cortisol, amygdala, dopamine, neuroplasticity, rumination, default mode network where relevant",
  "emotion_intensity": "low or medium or high"
}}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=600
    )

    raw = response.choices[0].message.content.strip()

    # Clean markdown fences if model adds them
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    return json.loads(raw)


# Load dataset
print("Loading gita_dataset.json...")
with open("gita_dataset.json", encoding="utf-8") as f:
    dataset = json.load(f)

total = len(dataset)
tagged = 0
errors = 0

print(f"Found {total} shlokas. Starting tagging with Groq (LLaMA 3.3 70B)...\n")

for i, shloka in enumerate(dataset):

    # Skip if already tagged
    if shloka.get("themes") and len(shloka["themes"]) > 0:
        print(f"  [{i+1}/{total}] {shloka['id']} — already tagged, skipping")
        continue

    try:
        print(f"  [{i+1}/{total}] Tagging {shloka['id']}...", end=" ", flush=True)
        tags = tag_shloka(shloka)

        dataset[i]["themes"]           = tags.get("themes", [])
        dataset[i]["life_scenarios"]   = tags.get("life_scenarios", [])
        dataset[i]["neuroscience_link"]= tags.get("neuroscience_link", "")
        dataset[i]["emotion_intensity"]= tags.get("emotion_intensity", "")

        tagged += 1
        print(f"✓  {tags.get('themes', [])[:3]}")

        # Save every 20 shlokas
        if tagged % 20 == 0:
            with open("gita_dataset.json", "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            print(f"\n  💾 Progress saved — {tagged} tagged so far\n")

        time.sleep(0.2)   # Groq is fast, tiny pause to respect rate limits

    except Exception as e:
        errors += 1
        print(f"✗  Error on {shloka['id']}: {e}")
        with open("gita_dataset.json", "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        time.sleep(3)     # wait a bit longer if there's an error

# Final save
with open("gita_dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"\n✅ Complete! Tagged: {tagged} | Errors: {errors} | Total: {total}")
print("gita_dataset.json is now fully tagged and ready for GeetaAI!")