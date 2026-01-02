import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "YOUR_TMDB_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "YOUR_YOUTUBE_KEY")

# Mapping for Watch Providers (US Region IDs approx.)
PROVIDER_MAP = {
    "Netflix": 8,
    "Prime Video": 9,
    "Disney+": 337,
    "Hulu": 15
}

# Focus Mode Genres for TMDB (Drama, Documentary, History)
FOCUS_GENRES = [18, 99, 36] 

# User Data Storage
USER_DATA_FILE = "user_data.json"