import random
import requests
import json
import os
import traceback
import re
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

# --- Lazy TMDB Setup ---
def get_tmdb():
    """Returns a configured TMDB instance."""
    t = TMDb()
    t.api_key = config.TMDB_API_KEY
    t.language = 'en'
    return t

# --- Model Loading (Cached) ---
@st.cache_resource
def load_embedding_model():
    """Loads the lightweight SentenceTransformer model."""
    # We use this for manual encoding if needed, though Chroma handles it too.
    # To keep it consistent across the app, we'll use this model instance.
    return SentenceTransformer('all-MiniLM-L6-v2')

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

def extract_user_keywords(liked_titles, onboarding_content, top_k=8):
    """
    Extract top keywords from user's liked content to enrich search queries.
    
    Args:
        liked_titles: List of titles the user liked during onboarding
        onboarding_content: List of content dicts with title, overview/description
        top_k: Number of top keywords to extract
    
    Returns:
        String of top keywords joined by spaces
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
    
    # Get top keywords
    top_keywords = [word for word, count in word_counts.most_common(top_k)]
    
    logger.debug(f"Extracted keywords from {len(liked_titles)} liked items: {top_keywords}")
    return " ".join(top_keywords)

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

# --- User Data Management ---
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

# --- Fetch Logic ---

def init_tmdb():
    """Initializes global TMDB settings from config."""
    t = TMDb()
    t.api_key = config.TMDB_API_KEY
    t.language = 'en'
    return t

def safe_get(item, key, default=None):
    """Safely gets a value from a dict or object."""
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)

def get_onboarding_content():
    """Fetches mixed content. Tries API -> Caches -> Returns. If API fails, returns Cached/Static."""
    logger.info("Fetching onboarding content...")
    all_content = []
    seen_ids = set()
    
    # 1. Fetch from APIs if keys exist
    if config.TMDB_API_KEY and config.TMDB_API_KEY != "YOUR_TMDB_KEY":
        logger.info("TMDB API key present, fetching movies for onboarding...")
        try:
            init_tmdb()
            discover = Discover()
            genres = [28, 35, 18, 878, 99]
            for g_id in genres:
                logger.debug(f"TMDB: Discovering movies for genre {g_id}")
                res = discover.discover_movies({'sort_by': 'popularity.desc', 'with_genres': g_id, 'vote_count.gte': 500, 'page': 1})
                
                # Handle raw dict response
                logger.debug(f"TMDB response type: {type(res).__name__}")
                if isinstance(res, dict) and 'results' in res: 
                    res = res['results']
                else:
                    # Convert AsObj/Iterable to list to support slicing
                    res = list(res)
                
                logger.debug(f"TMDB: Got {len(res)} results for genre {g_id}")
                
                # Iterate and extract safely
                for m in res[:6]:
                    mid = safe_get(m, 'id')
                    mid_str = str(mid)
                    
                    if mid_str in seen_ids:
                        continue
                        
                    poster_path = safe_get(m, 'poster_path')
                    title = safe_get(m, 'title')
                    
                    img = f"https://image.tmdb.org/t/p/w300{poster_path}" if poster_path else "https://via.placeholder.com/300x450?text=No+Image"
                    
                    if mid and title: # Ensure essential fields exist
                        seen_ids.add(mid_str)
                        all_content.append({
                            "id": mid_str, "title": title, "type": "movie",
                            "poster_path": img, "overview": safe_get(m, 'overview'),
                            "vote_average": safe_get(m, 'vote_average'), "runtime": safe_get(m, 'runtime', 0)
                        })
            logger.info(f"TMDB: Successfully fetched {len([c for c in all_content if c['type']=='movie'])} movies")
        except Exception as e:
            logger.error(f"TMDB Onboarding Error: {type(e).__name__}: {e}")
            logger.debug(traceback.format_exc())

    if config.YOUTUBE_API_KEY != "YOUR_YOUTUBE_KEY":
        logger.info("YouTube API key present, fetching videos for onboarding...")
        try:
            youtube = build('youtube', 'v3', developerKey=config.YOUTUBE_API_KEY)
            # Use high-signal keywords to filter out "slop"
            topics = [
                "deep dive video essay", 
                "full history documentary", 
                "philosophy analysis", 
                "advanced science visualised", 
                "cinema critique video essay",
                "investigative journalism documentary"
            ]
            for topic in topics:
                logger.debug(f"YouTube: Searching for '{topic}'")
                res = youtube.search().list(q=topic, part='id,snippet', maxResults=6, type='video', videoDuration='medium').execute()
                items_found = res.get('items', [])
                logger.debug(f"YouTube: Got {len(items_found)} results for '{topic}'")
                for item in items_found:
                    vid_id = item['id']['videoId']
                    if vid_id in seen_ids:
                        continue
                        
                    thumbs = item['snippet']['thumbnails']
                    thumb_url = thumbs.get('high', thumbs.get('medium', thumbs.get('default')))['url']
                    
                    seen_ids.add(vid_id)
                    all_content.append({
                        "id": vid_id, "title": item['snippet']['title'], "type": "video",
                        "poster_path": thumb_url, "thumbnail": thumb_url, "video_id": vid_id,
                        "overview": item['snippet']['description'], "description": item['snippet']['description'],
                        "duration": "20 mins" # Approx
                    })
            logger.info(f"YouTube: Successfully fetched {len([c for c in all_content if c['type']=='video'])} videos")
        except Exception as e:
            logger.error(f"YouTube Onboarding Error: {type(e).__name__}: {e}")
            logger.debug(traceback.format_exc())

    # 2. If we got content, Cache It!
    if all_content:
        logger.info(f"Onboarding: Got {len(all_content)} items from APIs, caching...")
        cache_content_to_db(all_content)
        random.shuffle(all_content)
        return all_content

    # 3. If API failed or keys missing, fetch from Cache/Static
    logger.warning("Onboarding: No API content, trying cached content...")
    cached = get_random_cached_content(limit=30)
    if len(cached) > 5:
        logger.info(f"Onboarding: Using {len(cached)} cached items")
        return cached
    
    # 4. Fallback to Static Pool (and cache it for next time)
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
    
    # A. Try Live API (Recall)
    if config.TMDB_API_KEY and config.TMDB_API_KEY != "YOUR_TMDB_KEY":
        logger.info("TMDB: Starting movie discovery...")
        try:
            init_tmdb()
            discover = Discover()
            kwargs = {
                'sort_by': 'popularity.desc', 'vote_average.gte': 6.5,
                'with_runtime.lte': max_time, 'page': 1
            }
            logger.debug(f"TMDB: Base query params: {kwargs}")
            # ... Filters ...
            provider_ids = [str(config.PROVIDER_MAP[s]) for s in subscriptions if s in config.PROVIDER_MAP]
            if provider_ids:
                kwargs['with_watch_providers'] = '|'.join(provider_ids); kwargs['watch_region'] = 'US'
            
            people_ids = []
            if actors: 
                for a in actors.split(','): 
                    pid = search_person_id(a.strip())
                    if pid: people_ids.append(str(pid))
            if directors:
                for d in directors.split(','):
                    pid = search_person_id(d.strip())
                    if pid: people_ids.append(str(pid))
            if people_ids: kwargs['with_people'] = '|'.join(people_ids)
            if focus_mode: kwargs['with_genres'] = ','.join([str(g) for g in config.FOCUS_GENRES])

            # Fetch 2 pages
            for page in [1, 2]:
                kwargs['page'] = page
                res = discover.discover_movies(kwargs)
                if isinstance(res, dict) and 'results' in res: 
                    res = res['results']
                else:
                    res = list(res)
                
                watched_list = [w.strip().lower() for w in watched_movies.split(',')]
                
                for m in res:
                    title_str = str(safe_get(m, 'title', ''))
                    if title_str.lower() not in watched_list:
                        poster_path = safe_get(m, 'poster_path')
                        img_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else "https://via.placeholder.com/300x450?text=No+Poster"
                        
                        api_results.append({
                            "title": title_str, 
                            "overview": str(safe_get(m, 'overview', '')),
                            "release_date": safe_get(m, 'release_date', 'N/A'),
                            "vote_average": safe_get(m, 'vote_average', 0),
                            "runtime": safe_get(m, 'runtime', 'N/A'),
                            "poster_path": img_url, 
                            "id": safe_get(m, 'id', 0),
                            "type": "movie", 
                            "match_reason": "Filtered Match"
                        })
        except Exception as e:
            logger.error(f"TMDB API Error: {type(e).__name__}: {e}")
            logger.debug(traceback.format_exc())
            error_msg = f"TMDB API Error: {str(e)}"

    # B. Cache fresh results if any
    if api_results:
        logger.info(f"TMDB: Got {len(api_results)} movie results, caching...")
        cache_content_to_db(api_results)
    
    # C. If API failed or yielded 0 results, query Vector DB (Semantic Fallback)
    candidates = api_results
    if not candidates and mood:
        # Search DB for movies
        logger.info(f"TMDB: No API results, falling back to vector search for mood '{mood}'")
        candidates = query_vector_db(mood, n_results=15, where_filter={"type": "movie"})
        if candidates:
            logger.info(f"TMDB: Found {len(candidates)} cached movies via vector search")
            error_msg = "Showing similar cached movies (API unavailable/empty)." if error_msg else "Showing semantically similar movies from cache."
    
    # If still empty, use Static Pool
    if not candidates:
        logger.warning("TMDB: Both API and cache empty, using static fallback")
        pool = [x for x in get_static_content_pool() if x['type'] == 'movie']
        # Can also vector search the static pool if mood exists
        if mood:
            candidates = query_vector_db(mood, n_results=5, where_filter={"type": "movie"}) or pool[:5]
        else:
            candidates = pool[:5]
        if not error_msg: error_msg = "No matches found. Showing popular fallback content."

    # D. Final Re-ranking (if we have candidates and mood)
    # Even if they came from API, we re-rank them here (unless they came from vector search already)
    # The `query_vector_db` already returns sorted by distance, but `api_results` are sorted by popularity.
    if mood and api_results:
        logger.info(f"Re-ranking {len(candidates)} movies by semantic similarity to '{enriched_query}'")
        # Use our manual re-ranker for the fresh API batch
        candidates = semantic_rerank(candidates, enriched_query, text_key='overview', top_k=10)
        for item in candidates:
            score = item.get('similarity_score', 0)
            display_pct = normalize_score(score)
            item['match_reason'] = f"<b>{display_pct}% Match</b> to '{mood}'"

    logger.info(f"Returning {len(candidates[:10])} movie recommendations")
    return candidates[:10], error_msg

def fetch_video_recommendations(mood, max_time_mins=40, liked_content=None):
    # Build enriched query from user's liked content
    enriched_query = mood
    if liked_content and mood:
        liked_titles, onboarding_content = liked_content
        keywords = extract_user_keywords(liked_titles, onboarding_content)
        if keywords:
            enriched_query = f"{mood} {keywords}"
            logger.info(f"Video enriched query: '{enriched_query}'")
    
    logger.info(f"Fetching video recommendations for mood: '{mood}'")
    api_results = []
    error_msg = None

    if config.YOUTUBE_API_KEY != "YOUR_YOUTUBE_KEY":
        logger.info("YouTube: Starting video search...")
        try:
            youtube = build('youtube', 'v3', developerKey=config.YOUTUBE_API_KEY)
            # Query for substantial content
            search_query = f"{mood} deep dive video essay analysis documentary"
            logger.debug(f"YouTube: Search query: '{search_query}'")
            res = youtube.search().list(q=search_query, part='id,snippet', maxResults=15, type='video', videoDuration='long').execute()
            
            items_found = res.get('items', [])
            logger.info(f"YouTube: Got {len(items_found)} results")
            for item in items_found:
                thumbs = item['snippet']['thumbnails']
                thumb_url = thumbs.get('high', thumbs.get('medium', thumbs.get('default')))['url']
                api_results.append({
                    "title": item['snippet']['title'], "description": item['snippet']['description'],
                    "thumbnail": thumb_url, "poster_path": thumb_url, # standardize
                    "video_id": item['id']['videoId'], "id": item['id']['videoId'],
                    "type": "video", "duration": "20 mins", "match_reason": "Keyword Match",
                    "overview": item['snippet']['description']
                })
        except Exception as e:
            logger.error(f"YouTube API Error: {type(e).__name__}: {e}")
            logger.debug(traceback.format_exc())
            error_msg = f"YouTube API Error: {str(e)}"

    if api_results:
        logger.info(f"YouTube: Caching {len(api_results)} video results")
        cache_content_to_db(api_results)
    
    candidates = api_results
    if not candidates and mood:
        logger.info(f"YouTube: No API results, falling back to vector search for mood '{mood}'")
        candidates = query_vector_db(mood, n_results=10, where_filter={"type": "video"})
        if candidates: 
            logger.info(f"YouTube: Found {len(candidates)} cached videos")
            error_msg = "Showing cached videos."

    if not candidates:
        logger.warning("YouTube: Both API and cache empty, using static fallback")
        pool = [x for x in get_static_content_pool() if x['type'] == 'video']
        candidates = pool[:5]

    if mood and api_results:
        logger.info(f"Re-ranking {len(candidates)} videos by semantic similarity")
        candidates = semantic_rerank(candidates, enriched_query, 'description', top_k=5)
        for item in candidates:
            score = item.get('similarity_score', 0)
            display_pct = normalize_score(score)
            item['match_reason'] = f"<b>{display_pct}% Match</b> to '{mood}'"

    logger.info(f"Returning {len(candidates)} video recommendations")
    return candidates, error_msg

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