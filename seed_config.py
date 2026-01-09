"""
Essential queries for quick database seeding on Streamlit Cloud startup.

This file contains a curated list of ~50 high-value search queries
designed to quickly populate the database with quality content.
These are used for fast startup seeding (vs the full 280+ queries in seed_database.py).
"""

# Essential queries for quick seeding (~50 high-value queries)
# These cover core categories and will seed ~500 videos in ~2-3 minutes
ESSENTIAL_VIDEO_QUERIES = [
    # === SCIENCE (10 queries) ===
    "Kurzgesagt science",
    "Veritasium science explained",
    "physics documentary",
    "space exploration documentary",
    "biology nature documentary",
    "chemistry explained",
    "astronomy universe documentary",
    "quantum mechanics explained",
    "climate science documentary",
    "medical science documentary",
    
    # === HISTORY (8 queries) ===
    "history documentary",
    "ancient civilizations documentary",
    "World War documentary",
    "empire rise and fall documentary",
    "historical figures biography",
    "archaeology discoveries",
    "medieval history documentary",
    "cold war documentary",
    
    # === PHILOSOPHY & PSYCHOLOGY (6 queries) ===
    "philosophy explained",
    "stoicism philosophy",
    "psychology documentary",
    "consciousness explained",
    "ethics moral philosophy",
    "existentialism explained",
    
    # === TECHNOLOGY (6 queries) ===
    "technology explained documentary",
    "artificial intelligence documentary",
    "engineering how it works",
    "computer science explained",
    "future technology documentary",
    "innovation documentary",
    
    # === VIDEO ESSAYS (10 queries) ===
    "video essay analysis",
    "film analysis video essay",
    "deep dive documentary",
    "investigative journalism",
    "explained documentary",
    "3Blue1Brown math",
    "Tom Scott interesting places",
    "Wendover Productions",
    "Johnny Harris documentary",
    "Real Engineering",
    
    # === ARTS & CULTURE (5 queries) ===
    "art history documentary",
    "music theory explained",
    "architecture documentary",
    "literature analysis",
    "cultural documentary",
    
    # === NATURE & ENVIRONMENT (5 queries) ===
    "nature documentary wildlife",
    "ocean documentary marine life",
    "planet earth nature",
    "environmental documentary",
    "animal behavior documentary",
]

# TMDB genres to seed (subset for quick seeding)
ESSENTIAL_TMDB_GENRES = {
    28: "Action",
    35: "Comedy",
    18: "Drama",
    878: "Science Fiction",
    99: "Documentary",
    53: "Thriller",
    16: "Animation",
    36: "History",
}

# Minimum DB size before triggering seeding
MIN_DB_SIZE = 200

# Target DB size after seeding
TARGET_DB_SIZE = 1000
