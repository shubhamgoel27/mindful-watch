# Unrot Me - Project Changes Summary

**Latest Update:** January 13, 2026  
**Focus:** Hybrid scoring system, diversity filter, UI rebrand

---

## 1. App Rebrand

Renamed from MindfulWatch to **Unrot Me** with Gen-Z focused messaging.

| Element | Before | After |
|---------|--------|-------|
| Name | MindfulWatch | **Unrot Me** |
| Icon | 🧘 | **🧠** |
| Tagline | "Your personalized guide..." | **"Doomscrolling lost. You won."** |

---

## 2. Hybrid Scoring System

Replaced simple weighted average with geometric mean + bonuses + diversity penalty.

### Formula

```python
# Geometric mean (both mood and profile must be good)
base_score = sqrt(profile_score * mood_score)

# Bonus for exceptional matches (capped at 15% each)
mood_bonus = max(0, mood_score - 0.4) * 0.15
profile_bonus = max(0, profile_score - 0.4) * 0.15

final_score = base_score + mood_bonus + profile_bonus
```

### Diversity Penalty

Items >85% similar to already-selected results get ×0.7 penalty.

---

## 3. Embedding Model

Using `BAAI/bge-small-en-v1.5` (~130MB, 384 dims) for cloud compatibility.

---

## 4. ChromaDB Optimizations

```python
HNSW_CONFIG = {
    "hnsw:M": 16,
    "hnsw:construction_ef": 100,
    "hnsw:search_ef": 50,
    "hnsw:space": "cosine",
}
```

---

## 5. Video Duration Filtering

| Filter | Value |
|--------|-------|
| Min duration | 5 minutes |
| Max duration | 120 minutes |
| Skip keywords | "10 hour", "sleep music", "white noise" |

---

## 6. Power User Admin Panel

Password-protected admin panel (`power_user_27` + `ADMIN_PASSWORD`):

| Tab | Features |
|-----|----------|
| Clear Data | Clear DB, user data |
| Seed Content | Quick seed + full seed |
| Fetch Videos | YouTube search |
| Fetch Movies | TMDB search + discover |
| View Content | Database samples |

---

## 7. Tinder-Style Onboarding

- Centered image cards with max-height
- Like / Skip / Nope buttons
- Progress indicator
- Min 5 ratings required

---

## Key Files

| File | Purpose |
|------|---------|
| `utils.py` | Hybrid scoring, diversity filter, embedding model |
| `app.py` | Admin panel, onboarding, rebranded UI |
| `seed_database.py` | Full seeding |
| `config.py` | Secret handling |

---

## Usage

```bash
# Run app
uv run streamlit run app.py

# Full database seed
uv run python seed_database.py

# View stats
uv run python db_stats.py stats
```

---

## Secrets Required

| Secret | Purpose |
|--------|---------|
| `TMDB_API_KEY` | Movie data |
| `YOUTUBE_API_KEY` | Video metadata |
| `ADMIN_PASSWORD` | Power user access |
