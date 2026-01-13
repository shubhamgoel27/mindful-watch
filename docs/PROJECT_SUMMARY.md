# MindfulWatch - Project Changes Summary

**Latest Update:** January 12, 2026  
**Focus:** Content curation, embedding model upgrade, admin panel, security

---

## 1. Embedding Model Upgrade

Switched from `all-MiniLM-L6-v2` (22M params) to `Qwen/Qwen3-Embedding-0.6B` (600M params).

| Aspect | Before | After |
|--------|--------|-------|
| Model | MiniLM-L6-v2 | Qwen3-Embedding-0.6B |
| Embedding Dims | 384 | 1024 |
| Quality | Basic | Multilingual, better semantic |

---

## 2. ChromaDB Optimizations

HNSW index tuning for faster queries and efficient storage:

```python
HNSW_CONFIG = {
    "hnsw:M": 16,                  # Balanced connectivity
    "hnsw:construction_ef": 100,   # Quality index
    "hnsw:search_ef": 50,          # 5x faster queries
    "hnsw:space": "cosine",        # Text embeddings
}
```

---

## 3. Video Duration Filtering

Content curation to skip shorts and ambient content:

| Filter | Value |
|--------|-------|
| Minimum duration | 5 minutes |
| Maximum duration | 120 minutes |
| Skip keywords | "10 hour", "sleep music", "white noise", etc. |

---

## 4. Expanded Content Library

Target: 500+ movies, 10,000+ videos

| Content | Sources |
|---------|---------|
| Western movies | TMDB genres, top-rated, popular |
| Indian movies | Hindi, Tamil, Telugu (Bollywood/Tollywood) |
| Videos | 500+ curated YouTube queries |

---

## 5. Power User Admin Panel

Password-protected admin panel for `power_user_27`:

| Tab | Features |
|-----|----------|
| Clear Data | Clear DB, clear user data |
| Seed Content | Quick seed + full seed button |
| Fetch Videos | YouTube search by query |
| Fetch Movies | TMDB search + discover by genre/year/language |
| View Content | Sample database content |

**Security:** Password stored in Streamlit Secrets (`ADMIN_PASSWORD`), never in code.

---

## 6. Tinder-Style Onboarding

Card-by-card swipe interface for preferences:
- Like / Skip / Nope buttons
- Minimum 5 ratings required
- Progress indicator

---

## Key Files

| File | Purpose |
|------|---------|
| `utils.py` | Qwen3 model, HNSW config, video filtering |
| `app.py` | Admin panel, Tinder onboarding, modern UI |
| `seed_database.py` | Full seeding with Bollywood support |
| `config.py` | ADMIN_PASSWORD secret handling |

---

## Usage

```bash
# Run app
uv run streamlit run app.py

# Full database seed (local)
uv run python seed_database.py

# Clear and reset database
uv run python db_stats.py clear_all

# View stats
uv run python db_stats.py stats
```

---

## Secrets Required

| Secret | Purpose |
|--------|---------|
| `TMDB_API_KEY` | Movie data |
| `YOUTUBE_API_KEY` | Video metadata (optional with yt-dlp) |
| `ADMIN_PASSWORD` | Power user access |
