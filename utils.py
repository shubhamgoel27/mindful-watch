import random
import requests
import json
import os
import traceback
import re
import subprocess
from collections import Counter
import chromadb
from chromadb.utils import embedding_functions
from googleapiclient.discovery import build
from tmdbv3api import TMDb, Movie, Discover, Person, Configuration
import isodate
import config
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from logging_config import logger

# Try to import yt-dlp for quota-free YouTube search
try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    logger.warning("yt-dlp not installed. Fallback YouTube search unavailable.")

# --- Lazy TMDB Setup ---
def get_tmdb():
    """Returns a configured TMDB instance."""
    t = TMDb()
    t.api_key = config.TMDB_API_KEY
    t.language = 'en'
    return t

# --- Model Loading (Singleton) ---
# Use a module-level singleton to avoid reloading the model on every call
# This works both in Streamlit (with st.cache_resource) and standalone scripts

_embedding_model = None

def load_embedding_model():
    """
    Loads the lightweight SentenceTransformer model.
    Uses a singleton pattern to load only once per process.
    """
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model (first time)...")
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Embedding model loaded successfully")
    return _embedding_model

# --- Score Normalization ---
def normalize_score(raw_score, min_raw=0.0, max_raw=0.35):
    """
    Normalize cosine similarity to a more meaningful display range (50-100%).
    Cosine similarity for sentence embeddings typically ranges 0.0-0.4 for reasonable matches.
    """
    clamped = max(min_raw, min(raw_score, max_raw))
    normalized = (clamped - min_raw) / (max_raw - min_raw)
    return int(50 + normalized * 50)  # Maps to 50-100%

# --- Keyword Extraction ---
# Common stopwords to filter out
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
    'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
    'this', 'that', 'these', 'those', 'it', 'its', 'he', 'she', 'they', 'we', 'you',
    'who', 'which', 'what', 'when', 'where', 'how', 'why', 'all', 'each', 'every',
    'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'not', 'only',
    'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there',
    'about', 'after', 'before', 'between', 'into', 'through', 'during', 'above',
    'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'their', 'them', 'his', 'her', 'him', 'your', 'our', 'my', 'can', 'get'
}

def extract_user_keywords(liked_titles, onboarding_content, top_k=3):
    """
    Extract TOP keywords only (not all) - quality over quantity.
    Used for simple keyword-based search queries.
    
    Args:
        liked_titles: List of titles the user liked during onboarding
        onboarding_content: List of content dicts with title, overview/description
        top_k: Number of top keywords to extract (default: 3 for focused queries)
    
    Returns:
        String of top keywords joined by spaces (max ~3-5 words)
    """
    if not liked_titles or not onboarding_content:
        return ""
    
    # Build a lookup for content by title
    content_by_title = {item.get('title', ''): item for item in onboarding_content}
    
    # Collect all text from liked content
    all_text = []
    for title in liked_titles:
        if title in content_by_title:
            item = content_by_title[title]
            text = f"{item.get('overview', '')} {item.get('description', '')}"
            all_text.append(text)
    
    if not all_text:
        return ""
    
    combined_text = " ".join(all_text).lower()
    
    # Extract words (alphanumeric, 3+ chars)
    words = re.findall(r'\b[a-z]{3,}\b', combined_text)
    
    # Filter stopwords
    filtered_words = [w for w in words if w not in STOPWORDS]
    
    # Count frequencies
    word_counts = Counter(filtered_words)
    
    # Get top keywords - limit to top_k for focused search
    top_keywords = [word for word, count in word_counts.most_common(top_k)]
    
    logger.debug(f"Extracted {len(top_keywords)} keywords from {len(liked_titles)} liked items: {top_keywords}")
    return " ".join(top_keywords)


def create_user_profile_embedding(liked_titles, onboarding_content):
    """
    Create a semantic embedding representing user preferences.
    This is used for RERANKING (not for keyword search).
    
    Combines the descriptions of all liked content into a single
    embedding that captures the user's overall interests.
    
    Returns:
        numpy array: User profile embedding vector, or None if no data
    """
    if not liked_titles or not onboarding_content:
        return None
    
    # Build a lookup for content by title
    content_by_title = {item.get('title', ''): item for item in onboarding_content}
    
    # Collect descriptions from liked content
    texts = []
    for title in liked_titles:
        if title in content_by_title:
            item = content_by_title[title]
            desc = item.get('overview', '') or item.get('description', '')
            if desc:
                texts.append(desc)
    
    if not texts:
        return None
    
    # Combine all descriptions into one text for embedding
    combined = " ".join(texts)
    
    # Truncate if too long (embedding models have limits)
    max_chars = 2000
    if len(combined) > max_chars:
        combined = combined[:max_chars]
    
    model = load_embedding_model()
    embedding = model.encode([combined])[0]
    
    logger.debug(f"Created user profile embedding from {len(texts)} liked items")
    return embedding

# --- Vector Database Setup (ChromaDB) ---
@st.cache_resource
def get_vector_collection():
    """Initializes and returns the persistent ChromaDB collection."""
    logger.info("Initializing ChromaDB collection...")
    try:
        client = chromadb.PersistentClient(path="./mindful_watch_db")
        
        # We will use the same model for Chroma's embedding function
        # Or let Chroma use its default (which is also all-MiniLM-L6-v2 usually)
        # But to be safe and offline-capable, let's use our local SentenceTransformer logic via a custom function or just pass raw embeddings.
        # For simplicity in this demo, we'll generate embeddings manually using our cached model and pass them to Chroma.
        
        collection = client.get_or_create_collection(name="mindful_content")
        logger.info(f"ChromaDB collection initialized successfully")
        return collection
    except Exception as e:
        logger.error(f"ChromaDB Init Error: {type(e).__name__}: {e}")
        logger.debug(traceback.format_exc())
        return None

def cache_content_to_db(items):
    """
    Takes a list of content items (dicts), generates embeddings, and saves to ChromaDB.
    Expected item keys: id, title, overview/description, type, poster_path, etc.
    """
    collection = get_vector_collection()
    if not collection or not items:
        return

    ids = []
    documents = []
    metadatas = []
    embeddings = []

    model = load_embedding_model()

    for item in items:
        # Unique ID string
        item_id = str(item.get('id', ''))
        # Fix mock IDs collision if re-generating
        if not item_id or item_id in ids: 
            continue
            
        ids.append(item_id)
        
        # Text to embed: Title + Overview/Description
        text = f"{item.get('title', '')}. {item.get('overview', '') or item.get('description', '')}"
        documents.append(text)
        
        # Metadata (flat dict)
        meta = {
            "title": item.get('title', 'Unknown'),
            "type": item.get('type', 'unknown'),
            "poster_path": item.get('poster_path', '') or item.get('thumbnail', ''),
            "runtime": str(item.get('runtime', '')),
            "vote_average": str(item.get('vote_average', '')),
            "video_id": item.get('video_id', ''),
            "duration": item.get('duration', ''),
            "overview": (item.get('overview', '') or item.get('description', ''))[:1000] # Truncate for metadata
        }
        # Clean None values for Chroma
        meta = {k: v for k, v in meta.items() if v is not None}
        metadatas.append(meta)

    # Generate embeddings batch
    if documents:
        embeddings = model.encode(documents).tolist()
        
        try:
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            logger.info(f"Cached {len(ids)} items to Vector DB.")
        except Exception as e:
            logger.error(f"Error caching to ChromaDB: {type(e).__name__}: {e}")
            logger.debug(traceback.format_exc())

def query_vector_db(query_text, n_results=10, where_filter=None):
    """
    Searches the Vector DB.
    where_filter example: {"type": "movie"}
    """
    collection = get_vector_collection()
    if not collection:
        return []

    model = load_embedding_model()
    query_vec = model.encode([query_text]).tolist()

    try:
        results = collection.query(
            query_embeddings=query_vec,
            n_results=n_results,
            where=where_filter
        )
        
        # Parse results back to our app's list-of-dicts format
        parsed_results = []
        if results['ids']:
            for i in range(len(results['ids'][0])):
                meta = results['metadatas'][0][i]
                doc_id = results['ids'][0][i]
                dist = results['distances'][0][i]
                
                # Reconstruct item
                item = {
                    "id": doc_id,
                    "title": meta.get('title'),
                    "type": meta.get('type'),
                    "overview": meta.get('overview'),
                    "description": meta.get('overview'), # Map back
                    "poster_path": meta.get('poster_path'),
                    "thumbnail": meta.get('poster_path'), # Map back
                    "runtime": meta.get('runtime'),
                    "duration": meta.get('duration'),
                    "video_id": meta.get('video_id'),
                    "vote_average": float(meta.get('vote_average', 0) or 0),
                    "match_reason": f"<b>{int((1-dist)*100)}% Match</b> (Cached)" # Rough approx logic
                }
                parsed_results.append(item)
                
        return parsed_results
        
    except Exception as e:
        logger.error(f"Vector Query Error: {type(e).__name__}: {e}")
        logger.debug(traceback.format_exc())
        return []

def get_random_cached_content(limit=20):
    """Returns random items from the DB to populate onboarding/fallback."""
    collection = get_vector_collection()
    if not collection: return []
    
    # Chroma doesn't have "random", but we can query with a generic term or get first N
    # Since we can't easily iterate all, we'll just get the first N (peek)
    try:
        # peek returns the first items in the DB
        # To get "random", better to fetch a larger batch and shuffle in python
        res = collection.get(limit=100)
        
        items = []
        if res['ids']:
            for i in range(len(res['ids'])):
                meta = res['metadatas'][i]
                item = {
                    "id": res['ids'][i],
                    "title": meta.get('title'),
                    "type": meta.get('type'),
                    "poster_path": meta.get('poster_path'),
                    "overview": meta.get('overview')
                }
                items.append(item)
        
        random.shuffle(items)
        return items[:limit]
    except:
        return []

# --- yt-dlp YouTube Search (Quota-Free Fallback) ---
def search_youtube_ytdlp(query, max_results=15):
    """
    Search YouTube using yt-dlp (no API quota).
    This is a fallback when the YouTube Data API quota is exceeded.
    """
    if not YT_DLP_AVAILABLE:
        logger.warning("yt-dlp not available for fallback search")
        return []
    
    logger.info(f"yt-dlp: Searching YouTube for '{query}'")
    
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,  # Don't download, just get metadata
            'skip_download': True,
            'default_search': f'ytsearch{max_results}',  # Limit results
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # ytsearch:query returns search results
            result = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
            
            if not result or 'entries' not in result:
                logger.warning("yt-dlp: No results found")
                return []
            
            videos = []
            for entry in result['entries']:
                if not entry:
                    continue
                    
                video_id = entry.get('id', '')
                title = entry.get('title', 'Unknown')
                description = entry.get('description', '') or ''
                
                # Get best thumbnail
                thumbnails = entry.get('thumbnails', [])
                thumb_url = ''
                if thumbnails:
                    # Prefer high quality thumbnail
                    for t in reversed(thumbnails):
                        if t.get('url'):
                            thumb_url = t['url']
                            break
                if not thumb_url:
                    thumb_url = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
                
                # Get duration
                duration_seconds = entry.get('duration')
                if duration_seconds:
                    mins = int(duration_seconds // 60)
                    duration = f"{mins} mins" if mins > 0 else "< 1 min"
                else:
                    duration = "Unknown"
                
                videos.append({
                    "id": video_id,
                    "video_id": video_id,
                    "title": title,
                    "description": description,
                    "overview": description,
                    "thumbnail": thumb_url,
                    "poster_path": thumb_url,
                    "type": "video",
                    "duration": duration,
                    "match_reason": "yt-dlp Search"
                })
            
            logger.info(f"yt-dlp: Found {len(videos)} videos")
            return videos
            
    except Exception as e:
        logger.error(f"yt-dlp search error: {type(e).__name__}: {e}")
        logger.debug(traceback.format_exc())
        return []

def load_user_data():
    if not os.path.exists(config.USER_DATA_FILE):
        return {}
    try:
        with open(config.USER_DATA_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_user_data(data):
    with open(config.USER_DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# --- Static Pool (Fallback) ---
def get_static_content_pool():
    # ... (Same as before, but ensure 'type' is set)
    # I will compress this for brevity but keep the logic
    content = []
    
    movies = [
        ("The Matrix", "Sci-Fi"), ("Inception", "Sci-Fi"), ("Interstellar", "Sci-Fi"),
        ("The Godfather", "Crime"), ("Pulp Fiction", "Crime"), ("Goodfellas", "Crime"),
        ("Spirited Away", "Animation"), ("The Lion King", "Animation"), ("Toy Story", "Animation"),
        ("Parasite", "Thriller"), ("Get Out", "Thriller"), ("Seven", "Thriller"),
        ("The Notebook", "Romance"), ("La La Land", "Romance"), ("Titanic", "Romance"),
        ("Gladiator", "Action"), ("Mad Max: Fury Road", "Action"), ("Die Hard", "Action"),
        ("Superbad", "Comedy"), ("The Hangover", "Comedy"), ("Mean Girls", "Comedy"),
        ("Schindler's List", "History"), ("Oppenheimer", "History"), ("1917", "History"),
        ("Planet Earth", "Documentary"), ("My Octopus Teacher", "Documentary")
    ]
    videos = [
        ("Kurzgesagt: The Egg", "Philosophy"), ("Veritasium: The Speed of Light", "Science"),
        ("Vsauce: What is Reality?", "Science"), ("TED: Do schools kill creativity?", "Education"),
        ("Daily Dose of Internet", "Fun"), ("Lofi Hip Hop Radio", "Music"),
        ("History of the World", "History"), ("How It's Made: Glass", "Engineering"),
        ("Primitive Technology", "Outdoors"), ("Binging with Babish", "Cooking"),
        ("Gordon Ramsay Cooking", "Cooking"), ("Bob Ross Painting", "Art"),
        ("Tom Scott: The Red Zone", "Travel"), ("GeoWizard: Straight Line", "Adventure")
    ]
    
    for i, (t, g) in enumerate(movies):
        color = f"{random.randint(50, 200):02x}{random.randint(50, 200):02x}{random.randint(50, 200):02x}"
        content.append({
            "id": f"mock_m_{i}", "title": t, "type": "movie",
            "poster_path": f"https://via.placeholder.com/300x450/{color}/FFFFFF?text={t.replace(' ', '+')}",
            "overview": f"A classic {g} movie.", "vote_average": 8.0, "runtime": 120,
            "match_reason": "Popular Classic" 
        })
    for i, (t, c) in enumerate(videos):
        color = f"{random.randint(50, 200):02x}{random.randint(50, 200):02x}{random.randint(50, 200):02x}"
        content.append({
            "id": f"mock_v_{i}", "title": t, "type": "video", "video_id": f"mock_vid_{i}",
            "poster_path": f"https://via.placeholder.com/400x225/{color}/FFFFFF?text={t.replace(' ', '+')}",
            "thumbnail": f"https://via.placeholder.com/400x225/{color}/FFFFFF?text={t.replace(' ', '+')}",
            "description": f"Video about {c}.", "duration": "20 mins", "overview": f"Video about {c}.",
            "match_reason": "Curated Pick"
        })
        
    # Ensure this static pool is ALSO cached to DB on first load!
    # We'll do that lazily in get_onboarding_content
    return content

# --- Direct TMDB API Functions (using requests instead of wrapper) ---

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p"

def fetch_tmdb_discover(params=None, max_pages=2):
    """
    Fetch movies from TMDB Discover API using direct requests.
    Returns list of movie dicts with proper structure.
    """
    if not config.TMDB_API_KEY or config.TMDB_API_KEY == "YOUR_TMDB_KEY":
        logger.warning("TMDB API key not configured")
        return []
    
    all_movies = []
    base_params = {
        "api_key": config.TMDB_API_KEY,
        "language": "en-US",
        "sort_by": "popularity.desc",
        "include_adult": "false",
        "include_video": "false",
    }
    
    if params:
        base_params.update(params)
    
    for page in range(1, max_pages + 1):
        base_params["page"] = page
        
        try:
            url = f"{TMDB_BASE_URL}/discover/movie"
            logger.debug(f"TMDB Request: {url} with params: {base_params}")
            
            response = requests.get(url, params=base_params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = data.get("results", [])
            
            logger.debug(f"TMDB page {page}: got {len(results)} movies")
            
            for movie in results:
                # Only include movies with required data
                movie_id = movie.get("id")
                title = movie.get("title", "")
                
                if not movie_id or not title:
                    logger.warning(f"TMDB: Skipping invalid movie - id={movie_id}, title={title}")
                    continue
                
                # Build full poster URL
                poster_path = movie.get("poster_path")
                full_poster_url = f"{TMDB_IMAGE_BASE}/w500{poster_path}" if poster_path else "https://via.placeholder.com/300x450?text=No+Poster"
                
                all_movies.append({
                    "id": movie_id,
                    "title": title,
                    "overview": movie.get("overview", ""),
                    "poster_path": full_poster_url,
                    "release_date": movie.get("release_date", "N/A"),
                    "vote_average": movie.get("vote_average", 0),
                    "popularity": movie.get("popularity", 0),
                    "type": "movie"
                })
                
        except requests.RequestException as e:
            logger.error(f"TMDB API request failed (page {page}): {e}")
            break
        except Exception as e:
            logger.error(f"TMDB parsing error (page {page}): {e}")
            break
    
    logger.info(f"TMDB: Fetched {len(all_movies)} valid movies total")
    return all_movies

def search_tmdb_person(name):
    """Search for a person (actor/director) on TMDB and return their ID."""
    if not config.TMDB_API_KEY or config.TMDB_API_KEY == "YOUR_TMDB_KEY":
        return None
    
    try:
        url = f"{TMDB_BASE_URL}/search/person"
        params = {
            "api_key": config.TMDB_API_KEY,
            "query": name,
            "language": "en-US"
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = data.get("results", [])
        
        if results:
            person_id = results[0].get("id")
            logger.debug(f"Found person '{name}' with ID: {person_id}")
            return person_id
        
    except Exception as e:
        logger.warning(f"TMDB person search error for '{name}': {e}")
    
    return None

def safe_get(item, key, default=None):
    """
    Safely gets a value from a dict or object.
    Handles TMDB's AsObj type which is dict-like but not isinstance(dict).
    """
    # First, try dict-style access (covers dict and dict-like objects like AsObj)
    if hasattr(item, 'get'):
        result = item.get(key, default)
        if result is not None:
            return result
    
    # Fallback to attribute access
    if hasattr(item, key):
        result = getattr(item, key, default)
        # Filter out methods (e.g., str.title on a string object)
        if callable(result) and not isinstance(result, (str, int, float, bool, list, dict)):
            return default
        return result
    
    return default

def get_onboarding_content():
    """
    Fetches mixed content for onboarding.
    PRIORITY: Cached DB content (randomized) -> Fresh API content -> Static fallback.
    This ensures variety across sessions by sampling randomly from the full database.
    """
    logger.info("Fetching onboarding content...")
    
    # 1. FIRST: Try to get random content from the cached database (most variety)
    cached = get_random_cached_content(limit=60)  # Get more for better mix
    if len(cached) >= 20:
        # We have enough cached content - use it for variety
        # Ensure good mix of movies and videos
        movies = [c for c in cached if c.get('type') == 'movie']
        videos = [c for c in cached if c.get('type') == 'video']
        
        # Take balanced mix
        random.shuffle(movies)
        random.shuffle(videos)
        balanced = movies[:15] + videos[:15]  # 15 of each
        random.shuffle(balanced)
        
        logger.info(f"Onboarding: Using {len(balanced)} cached items ({len(movies[:15])} movies, {len(videos[:15])} videos)")
        return balanced
    
    # 2. If cache is sparse, fetch fresh content from APIs
    all_content = []
    seen_ids = set()
    
    if config.TMDB_API_KEY and config.TMDB_API_KEY != "YOUR_TMDB_KEY":
        logger.info("TMDB API key present, fetching movies for onboarding...")
        genres = [28, 35, 18, 878, 99]  # Action, Comedy, Drama, Sci-Fi, Documentary
        
        for g_id in genres:
            movies = fetch_tmdb_discover(
                params={"with_genres": g_id, "vote_count.gte": 500},
                max_pages=1
            )
            
            for movie in movies[:6]:
                mid_str = str(movie["id"])
                if mid_str in seen_ids:
                    continue
                
                seen_ids.add(mid_str)
                all_content.append({
                    "id": mid_str,
                    "title": movie["title"],
                    "type": "movie",
                    "poster_path": movie["poster_path"],
                    "overview": movie.get("overview", ""),
                    "vote_average": movie.get("vote_average", 0),
                    "runtime": 0
                })
        
        logger.info(f"TMDB: Successfully fetched {len([c for c in all_content if c['type']=='movie'])} movies")

    # Try yt-dlp for videos (more reliable than YouTube API)
    if YT_DLP_AVAILABLE:
        logger.info("Fetching videos via yt-dlp for onboarding...")
        topics = [
            "deep dive video essay",
            "full history documentary",
            "philosophy analysis",
            "science documentary",
            "cinema critique"
        ]
        for topic in topics:
            videos = search_youtube_ytdlp(topic, max_results=5)
            for v in videos:
                vid_id = v.get('video_id') or v.get('id')
                if vid_id and vid_id not in seen_ids:
                    seen_ids.add(vid_id)
                    all_content.append(v)
        logger.info(f"yt-dlp: Successfully fetched {len([c for c in all_content if c['type']=='video'])} videos")

    # Cache whatever we got
    if all_content:
        logger.info(f"Onboarding: Got {len(all_content)} items from APIs, caching...")
        cache_content_to_db(all_content)
        random.shuffle(all_content)
        return all_content

    # 3. Last resort: Static Pool
    logger.warning("Onboarding: Using static fallback pool")
    static = get_static_content_pool()
    cache_content_to_db(static)
    random.shuffle(static)
    return static

def search_person_id(name):
    """Search for a person (actor/director) ID on TMDB."""
    try:
        if config.TMDB_API_KEY == "YOUR_TMDB_KEY": 
            return None
        search = Person()
        res = search.search(name)
        if res: 
            logger.debug(f"Found person ID for '{name}': {res[0].id}")
            return res[0].id
    except Exception as e:
        logger.warning(f"Error searching person '{name}': {type(e).__name__}: {e}")
        return None
    return None

def fetch_movie_recommendations(subscriptions, watched_movies, actors, directors, max_time, focus_mode, mood, liked_content=None):
    """
    Hybrid Search: API -> Cache -> Vector Search fallback.
    liked_content: tuple of (liked_titles, onboarding_content) for query enrichment
    """
    # Build enriched query from user's liked content
    enriched_query = mood
    if liked_content and mood:
        liked_titles, onboarding_content = liked_content
        keywords = extract_user_keywords(liked_titles, onboarding_content)
        if keywords:
            enriched_query = f"{mood} {keywords}"
            logger.info(f"Enriched query: '{enriched_query}'")
    
    logger.info(f"Fetching movie recommendations - mood: '{mood}', max_time: {max_time}, focus_mode: {focus_mode}")
    api_results = []
    error_msg = None
    
    # A. Try Live API using direct requests
    if config.TMDB_API_KEY and config.TMDB_API_KEY != "YOUR_TMDB_KEY":
        logger.info("TMDB: Starting movie discovery...")
        
        # Build params
        params = {
            "vote_average.gte": 6.5,
            "with_runtime.lte": max_time,
        }
        
        # Provider filter
        provider_ids = [str(config.PROVIDER_MAP[s]) for s in subscriptions if s in config.PROVIDER_MAP]
        if provider_ids:
            params["with_watch_providers"] = "|".join(provider_ids)
            params["watch_region"] = "US"
        
        # Actor/Director filter using new search function
        people_ids = []
        if actors:
            for a in actors.split(","):
                pid = search_tmdb_person(a.strip())
                if pid:
                    people_ids.append(str(pid))
        if directors:
            for d in directors.split(","):
                pid = search_tmdb_person(d.strip())
                if pid:
                    people_ids.append(str(pid))
        if people_ids:
            params["with_people"] = "|".join(people_ids)
        
        # Focus mode genres
        if focus_mode:
            params["with_genres"] = ",".join([str(g) for g in config.FOCUS_GENRES])
        
        # Fetch movies using direct API
        movies = fetch_tmdb_discover(params=params, max_pages=2)
        
        # Filter out watched movies
        watched_list = [w.strip().lower() for w in watched_movies.split(",") if w.strip()]
        
        for movie in movies:
            if movie["title"].lower() not in watched_list:
                movie["match_reason"] = "Filtered Match"
                movie["runtime"] = "N/A"  # Discover doesn't return runtime
                api_results.append(movie)
        
        logger.info(f"TMDB: Extracted {len(api_results)} valid movies after filtering")
    
    if not api_results:
        error_msg = "TMDB API returned no results"

    # B. Cache fresh results if any
    if api_results:
        logger.info(f"TMDB: Got {len(api_results)} movie results, caching...")
        cache_content_to_db(api_results)
    
    # C. If API failed or yielded 0 results, query Vector DB (Semantic Fallback)
    candidates = api_results
    if not candidates and mood:
        # Search DB for movies
        logger.info(f"TMDB: No API results, falling back to vector search for mood '{mood}'")
        candidates = query_vector_db(mood, n_results=50, where_filter={"type": "movie"})
        if candidates:
            logger.info(f"TMDB: Found {len(candidates)} cached movies via vector search")
            error_msg = "Showing similar cached movies (API unavailable/empty)." if error_msg else "Showing semantically similar movies from cache."
    
    # If still empty, use Static Pool
    if not candidates:
        logger.warning("TMDB: Both API and cache empty, using static fallback")
        pool = [x for x in get_static_content_pool() if x['type'] == 'movie']
        # Can also vector search the static pool if mood exists
        if mood:
            candidates = query_vector_db(mood, n_results=50, where_filter={"type": "movie"}) or pool[:50]
        else:
            candidates = pool[:5]
        if not error_msg: error_msg = "No matches found. Showing popular fallback content."

    # D. Final Re-ranking (if we have candidates and mood)
    # Even if they came from API, we re-rank them here (unless they came from vector search already)
    # The `query_vector_db` already returns sorted by distance, but `api_results` are sorted by popularity.
    if mood and api_results:
        logger.info(f"Re-ranking {len(candidates)} movies by semantic similarity to '{enriched_query}'")
        # Use our manual re-ranker for the fresh API batch
        candidates = semantic_rerank(candidates, enriched_query, text_key='overview', top_k=50)
        for item in candidates:
            score = item.get('similarity_score', 0)
            display_pct = normalize_score(score)
            item['match_reason'] = f"<b>{display_pct}% Match</b> to '{mood}'"

    logger.info(f"Returning {len(candidates[:50])} movie recommendations")
    return candidates[:50], error_msg

def fetch_video_recommendations(mood, max_time_mins=40, liked_content=None):
    """
    Two-stage retrieval + reranking for personalized video recommendations.
    
    Stage 1: RETRIEVAL (high recall)
    - Use mood as primary search query (clearest user intent signal)
    - Or use top 3-5 keywords if no mood
    - Multi-channel: yt-dlp + vector DB for diversity
    
    Stage 2: RERANKING (high precision)  
    - Two-factor scoring: profile similarity + mood similarity
    - Transparent match reasons showing both factors
    """
    logger.info(f"Fetching video recommendations. Mood: '{mood}'")
    
    # === PREPARE SIGNALS ===
    search_query = ""
    user_profile_embedding = None
    liked_titles = []
    onboarding_content = []
    
    if liked_content:
        liked_titles, onboarding_content = liked_content
        # Create user profile embedding for reranking (captures overall preferences)
        user_profile_embedding = create_user_profile_embedding(liked_titles, onboarding_content)
        # Extract only top 3 keywords for focused search
        search_keywords = extract_user_keywords(liked_titles, onboarding_content, top_k=3)
        logger.debug(f"User profile: {len(liked_titles)} liked items, keywords: '{search_keywords}'")
    else:
        search_keywords = ""
    
    # === STAGE 1: RETRIEVAL ===
    # Priority: mood > keywords > generic fallback
    # Use mood as primary search query - it's the clearest signal of current intent
    if mood:
        search_query = mood  # Use mood directly (e.g., "relaxing", "learn science")
        logger.info(f"Stage 1: Searching by mood: '{search_query}'")
    elif search_keywords:
        search_query = search_keywords  # Fallback to extracted keywords
        logger.info(f"Stage 1: Searching by keywords: '{search_query}'")
    else:
        search_query = "documentary video essay"  # Generic fallback
        logger.info(f"Stage 1: Using generic fallback query: '{search_query}'")
    
    candidates = []
    error_msg = None
    
    # Try yt-dlp first (no quota limits)
    if YT_DLP_AVAILABLE:
        logger.info(f"yt-dlp: Searching for '{search_query}'")
        yt_results = search_youtube_ytdlp(search_query, max_results=30)
        if yt_results:
            candidates.extend(yt_results)
            logger.info(f"yt-dlp: Got {len(yt_results)} results")
    
    # Try YouTube API as backup
    if len(candidates) < 20 and config.YOUTUBE_API_KEY != "YOUR_YOUTUBE_KEY":
        logger.info("YouTube API: Supplementing with API results...")
        try:
            youtube = build('youtube', 'v3', developerKey=config.YOUTUBE_API_KEY)
            res = youtube.search().list(
                q=search_query, part='id,snippet', 
                maxResults=20, type='video', videoDuration='long'
            ).execute()
            
            for item in res.get('items', []):
                thumbs = item['snippet']['thumbnails']
                thumb_url = thumbs.get('high', thumbs.get('medium', thumbs.get('default')))['url']
                candidates.append({
                    "title": item['snippet']['title'], 
                    "description": item['snippet']['description'],
                    "thumbnail": thumb_url, "poster_path": thumb_url,
                    "video_id": item['id']['videoId'], "id": item['id']['videoId'],
                    "type": "video", "duration": "20 mins",
                    "overview": item['snippet']['description']
                })
            logger.info(f"YouTube API: Added {len(res.get('items', []))} results")
        except Exception as e:
            logger.warning(f"YouTube API failed: {e}")
    
    # Also get from vector DB for diversity (using profile embedding if available)
    if user_profile_embedding is not None:
        logger.info("Vector DB: Searching by user profile embedding...")
        db_results = query_vector_db(search_query, n_results=30, where_filter={"type": "video"})
        if db_results:
            candidates.extend(db_results)
            logger.info(f"Vector DB: Added {len(db_results)} cached videos")
    
    # Deduplicate by ID
    seen = set()
    unique_candidates = []
    for c in candidates:
        cid = c.get('video_id') or c.get('id')
        if cid and cid not in seen:
            seen.add(cid)
            unique_candidates.append(c)
    
    logger.info(f"Stage 1 complete: {len(unique_candidates)} unique candidates")
    
    # Cache new content
    if unique_candidates:
        cache_content_to_db(unique_candidates)
    
    # Fallback to static pool if nothing found
    if not unique_candidates:
        logger.warning("No candidates found, using static fallback")
        pool = [x for x in get_static_content_pool() if x['type'] == 'video']
        unique_candidates = pool[:50]
        error_msg = "Showing curated content."
    
    # === STAGE 2: RERANKING ===
    logger.info(f"Stage 2: Reranking {len(unique_candidates)} candidates...")
    
    # Use two-factor reranking
    results = rerank_by_profile_and_mood(
        items=unique_candidates,
        user_profile_embedding=user_profile_embedding,
        mood=mood,
        top_k=50
    )
    
    logger.info(f"Returning {len(results)} video recommendations")
    return results, error_msg

# Reuse existing re-ranker from previous step
def semantic_rerank(items, query_text, text_key='overview', top_k=10):
    if not query_text or not items: return items[:top_k]
    try:
        model = load_embedding_model()
        q_vec = model.encode([query_text])
        i_vecs = model.encode([item.get(text_key, "") or "" for item in items])
        scores = cosine_similarity(q_vec, i_vecs)[0]
        scored = []
        for i, item in enumerate(items):
            c = item.copy()
            c['similarity_score'] = float(scores[i])
            scored.append(c)
        scored.sort(key=lambda x: x['similarity_score'], reverse=True)
        return scored[:top_k]
    except: return items[:top_k]


def rerank_by_profile_and_mood(items, user_profile_embedding, mood, top_k=50):
    """
    Two-factor reranking for personalized recommendations.
    
    Combines two signals:
    1. Profile similarity - how well the item matches user's overall preferences
    2. Mood similarity - how well the item matches user's current intent
    
    Args:
        items: List of candidate items to rerank
        user_profile_embedding: numpy array representing user preferences
        mood: string representing current user intent/mood
        top_k: number of top results to return
    
    Returns:
        List of items reranked by combined score with match_reason populated
    """
    if not items:
        return []
    
    model = load_embedding_model()
    
    # Get text for each item
    item_texts = []
    for item in items:
        text = item.get('description', '') or item.get('overview', '') or item.get('title', '')
        item_texts.append(text)
    
    # Encode all items
    item_embeddings = model.encode(item_texts)
    
    # Calculate profile similarity (if we have user profile)
    profile_scores = np.zeros(len(items))
    if user_profile_embedding is not None:
        profile_scores = cosine_similarity([user_profile_embedding], item_embeddings)[0]
        # Normalize to 0-1 range (cosine similarity can be negative)
        profile_scores = np.clip(profile_scores, 0, 1)
    
    # Calculate mood similarity (if mood provided)
    mood_scores = np.zeros(len(items))
    if mood:
        mood_embedding = model.encode([mood])[0]
        mood_scores = cosine_similarity([mood_embedding], item_embeddings)[0]
        # Normalize to 0-1 range
        mood_scores = np.clip(mood_scores, 0, 1)
    
    # Combine scores with appropriate weights
    # If mood provided: weight mood higher (60%) as it represents current intent
    # If no mood: use only profile scores
    if mood and user_profile_embedding is not None:
        alpha = 0.4  # profile weight
        beta = 0.6   # mood weight
        final_scores = alpha * profile_scores + beta * mood_scores
        score_type = "combined"
    elif mood:
        final_scores = mood_scores
        score_type = "mood"
    else:
        final_scores = profile_scores
        score_type = "profile"
    
    # Create scored items
    scored_items = list(zip(items, final_scores, profile_scores, mood_scores))
    scored_items.sort(key=lambda x: x[1], reverse=True)
    
    # Build results with transparent match reasons
    results = []
    for item, final, profile, mood_s in scored_items[:top_k]:
        item = item.copy()
        
        # Convert to percentage (scale up since cosine scores are typically 0.1-0.5)
        final_pct = min(99, int(final * 200))  # Scale up for display
        profile_pct = min(99, int(profile * 200))
        mood_pct = min(99, int(mood_s * 200))
        
        # Create transparent match reason
        if score_type == "combined":
            item['match_reason'] = f"<b>{final_pct}%</b> match ({mood_pct}% mood, {profile_pct}% profile)"
        elif score_type == "mood":
            item['match_reason'] = f"<b>{final_pct}%</b> match to '{mood}'"
        else:
            item['match_reason'] = f"<b>{final_pct}%</b> profile match"
        
        item['similarity_score'] = float(final)
        results.append(item)
    
    logger.info(f"Reranked {len(items)} items -> top {len(results)} by {score_type}")
    return results