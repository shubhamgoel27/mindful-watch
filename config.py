import os
import streamlit as st
from dotenv import load_dotenv
from logging_config import logger

load_dotenv()

def get_secret(key, default):
    """
    Retrieves secret from Streamlit Secrets (cloud) or environment variables (local).
    Logs the source for debugging.
    """
    value = None
    source = None
    
    # Check Streamlit Secrets first (for Cloud deployment)
    try:
        if hasattr(st, 'secrets') and key in st.secrets:
            value = st.secrets[key]
            source = "Streamlit Secrets"
    except FileNotFoundError:
        # .streamlit/secrets.toml doesn't exist (local dev without secrets file)
        logger.debug(f"Streamlit secrets file not found, checking env for {key}")
    except Exception as e:
        # Log unexpected errors but continue to env fallback
        logger.warning(f"Error accessing Streamlit Secrets for {key}: {type(e).__name__}: {e}")
    
    # Fallback to environment variables
    if value is None:
        value = os.getenv(key, default)
        if value and value != default:
            source = "Environment Variable"
        else:
            source = "Default/Missing"
    
    # Log result (mask actual key values for security)
    if value and value != default:
        logger.info(f"Config: {key} loaded from {source} (value present)")
    else:
        logger.warning(f"Config: {key} not found - using default ({source})")
    
    return value

# API Keys
TMDB_API_KEY = get_secret("TMDB_API_KEY", "YOUR_TMDB_KEY")
YOUTUBE_API_KEY = get_secret("YOUTUBE_API_KEY", "YOUR_YOUTUBE_KEY")

# Log API key status at startup
logger.info(f"API Status: TMDB={'✓' if TMDB_API_KEY != 'YOUR_TMDB_KEY' else '✗'}, YouTube={'✓' if YOUTUBE_API_KEY != 'YOUR_YOUTUBE_KEY' else '✗'}")

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