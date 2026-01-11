# MindfulWatch - Project Changes Summary

**Session:** January 8-10, 2026  
**Focus:** YouTube quota bypass, search improvements, personalization, cloud deployment

---

## 1. YouTube API Quota Bypass

### Problem
YouTube Data API quota exceeded (403 error), limiting video recommendations.

### Solution
Integrated **yt-dlp** as a quota-free fallback that scrapes YouTube directly.

### Files Changed
- `utils.py`: Added `search_youtube_ytdlp()` function
- `pyproject.toml`: Added `yt-dlp>=2024.0.0` dependency

### Fallback Order
```
YouTube API → yt-dlp → Vector DB cache → Static content
```

---

## 2. Fixed YouTube Search Query Issues

### Problem
Search queries were too long (244+ chars) by concatenating all liked titles + keywords.

### Solution
- Use only **top 3-5 keywords** (not 20+)
- Limit query length to 100 chars
- Removed redundant "deep dive video essay analysis" suffix

### Before
```
"Avatar: Fire and Ash The Advanced Science Research Center... philosophy avatar fire ash advanced science research center deep dive video essay analysis"
```

### After
```
"philosophy avatar fire"
```

---

## 3. Embedding Model Singleton

### Problem
Model was reloading on every `cache_content_to_db()` call because `@st.cache_resource` only works in Streamlit context.

### Solution
Module-level singleton pattern that works in both Streamlit and standalone scripts.

```python
_embedding_model = None

def load_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model
```

---

## 4. Expanded Vector Database

### Problem
Original DB only had ~89 items (26 movies, 63 videos).

### Solution
Created comprehensive seeding script with 280+ high-value queries.

### New Files
| File | Purpose |
|------|---------|
| `seed_database.py` | Full seeding script (280 queries) |
| `db_stats.py` | Database statistics & management tool |
| `seed_config.py` | Essential queries for quick startup (50 queries) |

### Results
| Metric | Before | After |
|--------|--------|-------|
| Total items | 89 | 3,495 |
| Movies | 26 | 641 |
| Videos | 63 | 2,854 |

### Query Categories
- Science & Physics
- Mathematics
- Biology & Nature
- History (25+ queries)
- Philosophy & Psychology
- Technology & Engineering
- Economics & Finance
- Video Essays (quality channels like Kurzgesagt, Veritasium, 3Blue1Brown)
- Arts & Culture
- Medicine & Health
- Geography & Earth Science

---

## 5. Two-Stage Retrieval + Reranking

### Problem
Single-query approach with all keywords jumbled together produced poor results.

### Solution
Industry-standard two-stage system following RAG best practices.

### Architecture
```
┌─────────────────────────────────────────────────┐
│              STAGE 1: RETRIEVAL                 │
│  • Use mood as primary search query             │
│  • Fallback to top 3 keywords                   │
│  • Multi-channel: yt-dlp + API + Vector DB      │
│  → Output: ~100 candidates                      │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│              STAGE 2: RERANKING                 │
│  • Profile similarity (40%): user preferences   │
│  • Mood similarity (60%): current intent        │
│  → Output: Top 50 ranked results                │
└─────────────────────────────────────────────────┘
```

### New Functions
- `create_user_profile_embedding()` - Semantic embedding from liked content
- `rerank_by_profile_and_mood()` - Two-factor personalized reranking

### Match Reasons
| Before | After |
|--------|-------|
| `52% Match to 'your preferences'` | `76% match (80% mood, 70% profile)` |

---

## 6. Improved Onboarding

### Problem
Onboarding was static (always same movies from API).

### Solution
- Prioritize cached DB content with **random sampling**
- Balanced mix: 15 movies + 15 videos randomly selected
- Uses yt-dlp instead of YouTube API (avoids quota issues)

---

## 7. Automatic Cloud Seeding

### Problem
Streamlit Cloud has ephemeral storage - local ChromaDB doesn't persist.

### Solution
Automatic seeding on app startup using `@st.cache_resource`.

### How It Works
1. App checks if DB has < 200 items
2. If sparse, seeds automatically:
   - ~80 movies from 8 genres
   - ~150 videos from 30 queries
3. Shows "🌱 Seeding database with content..." spinner
4. Only runs **once per deployment**

---

## File Summary

| File | Changes |
|------|---------|
| `utils.py` | yt-dlp integration, two-stage reranking, user profile embedding, singleton model loading |
| `app.py` | Startup seeding, improved onboarding |
| `seed_database.py` | **NEW** - Comprehensive seeding script |
| `db_stats.py` | **NEW** - Database management tool |
| `seed_config.py` | **NEW** - Essential queries for cloud |
| `pyproject.toml` | Added yt-dlp dependency |

---

## Commits

1. `0c7941b` - Add yt-dlp as fallback for YouTube search
2. `25f09ab` - Add database seeding script
3. `fa15229` - Fix embedding model reloading
4. `b972f1d` - Fix onboarding variety, increase results to 50
5. `edb31cf` - Implement two-stage retrieval + reranking
6. `7f23de6` - Add automatic database seeding on Streamlit Cloud startup

---

## Usage

### Full Database Seed (Local)
```bash
uv run python seed_database.py
```

### View Database Stats
```bash
uv run python db_stats.py stats
```

### Run App
```bash
uv run streamlit run app.py
```
