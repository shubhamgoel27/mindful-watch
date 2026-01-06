import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def get_secret(key, default):
    """Retrieves secret from streamlit secrets (cloud) or environment variables (local)."""
    try:
        # Check Streamlit Secrets first
        if key in st.secrets:
            return st.secrets[key]
    except:
        # Fallback if st.secrets is not available (e.g., during tests)
        pass
    return os.getenv(key, default)

# API Keys
TMDB_API_KEY = get_secret("TMDB_API_KEY", "YOUR_TMDB_KEY")
YOUTUBE_API_KEY = get_secret("YOUTUBE_API_KEY", "YOUR_YOUTUBE_KEY")

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