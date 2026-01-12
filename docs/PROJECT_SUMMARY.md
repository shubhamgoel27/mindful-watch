# MindfulWatch - Project Changes Summary

**Session:** January 8-11, 2026  
**Focus:** YouTube quota bypass, search improvements, personalization, cloud deployment, UI modernization

---

## 1. YouTube API Quota Bypass

### Problem
YouTube Data API quota exceeded (403 error), limiting video recommendations.

### Solution
Integrated **yt-dlp** as a quota-free fallback that scrapes YouTube directly.

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

---

## 3. Embedding Model Singleton

### Problem
Model was reloading on every `cache_content_to_db()` call.

### Solution
Module-level singleton pattern that works in both Streamlit and standalone scripts.

---

## 4. Expanded Vector Database

| Metric | Before | After |
|--------|--------|-------|
| Total items | 89 | 3,495 |
| Movies | 26 | 641 |
| Videos | 63 | 2,854 |

---

## 5. Two-Stage Retrieval + Reranking

Industry-standard two-stage system:
- **STAGE 1: RETRIEVAL** - Mood as primary query, multi-channel search
- **STAGE 2: RERANKING** - Profile (40%) + Mood (60%) similarity scoring

---

## 6. Automatic Cloud Seeding

Automatic seeding on app startup using `@st.cache_resource`. Seeds if DB has < 200 items.

---

## 7. UI Modernization

Netflix/Spotify-inspired dark theme with:

| Element | Before | After |
|---------|--------|-------|
| Background | Light gray | Dark gradient (#0d1117) |
| Cards | Basic borders | Glassmorphism + glow |
| Buttons | Default | Gradient (purple→blue) |
| Typography | System fonts | Inter (Google Fonts) |
| Hover states | None | Lift + glow animation |

---

## File Summary

| File | Changes |
|------|---------|
| `utils.py` | yt-dlp, two-stage reranking, user profile embedding |
| `app.py` | Startup seeding, modern dark theme CSS |
| `seed_database.py` | **NEW** - Comprehensive seeding script |
| `db_stats.py` | **NEW** - Database management tool |
| `seed_config.py` | **NEW** - Essential queries for cloud |

---

## Usage

```bash
# Run app
uv run streamlit run app.py

# Full database seed
uv run python seed_database.py

# View database stats
uv run python db_stats.py stats
```
