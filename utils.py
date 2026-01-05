import random
import requests
import json
import os
import traceback
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

# --- Setup TMDB ---
tmdb = TMDb()
tmdb.api_key = config.TMDB_API_KEY
tmdb.language = 'en'
tmdb.debug = True

# --- Model Loading (Cached) ---
@st.cache_resource
def load_embedding_model():
    """Loads the lightweight SentenceTransformer model."""
    # We use this for manual encoding if needed, though Chroma handles it too.
    # To keep it consistent across the app, we'll use this model instance.
    return SentenceTransformer('all-MiniLM-L6-v2')

# --- Vector Database Setup (ChromaDB) ---
@st.cache_resource
def get_vector_collection():
    """Initializes and returns the persistent ChromaDB collection."""
    try:
        client = chromadb.PersistentClient(path="./mindful_watch_db")
        
        # We will use the same model for Chroma's embedding function
        # Or let Chroma use its default (which is also all-MiniLM-L6-v2 usually)
        # But to be safe and offline-capable, let's use our local SentenceTransformer logic via a custom function or just pass raw embeddings.
        # For simplicity in this demo, we'll generate embeddings manually using our cached model and pass them to Chroma.
        
        collection = client.get_or_create_collection(name="mindful_content")
        return collection
    except Exception as e:
        print(f"ChromaDB Init Error: {e}")
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
            print(f"Cached {len(ids)} items to Vector DB.")
        except Exception as e:
            print(f"Error caching to ChromaDB: {e}")

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
        print(f"Vector Query Error: {e}")
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
            "poster_path": f"https://placehold.co/300x450/{color}/FFF?text={t.replace(' ', '+')}",
            "overview": f"A classic {g} movie.", "vote_average": 8.0, "runtime": 120,
            "match_reason": "Popular Classic" 
        })
    for i, (t, c) in enumerate(videos):
        color = f"{random.randint(50, 200):02x}{random.randint(50, 200):02x}{random.randint(50, 200):02x}"
        content.append({
            "id": f"mock_v_{i}", "title": t, "type": "video", "video_id": f"mock_vid_{i}",
            "poster_path": f"https://placehold.co/400x225/{color}/FFF?text={t.replace(' ', '+')}",
            "thumbnail": f"https://placehold.co/400x225/{color}/FFF?text={t.replace(' ', '+')}",
            "description": f"Video about {c}.", "duration": "20 mins", "overview": f"Video about {c}.",
            "match_reason": "Curated Pick"
        })
        
    # Ensure this static pool is ALSO cached to DB on first load!
    # We'll do that lazily in get_onboarding_content
    return content

# --- Fetch Logic ---

def safe_get(item, key, default=None):
    """Safely gets a value from a dict or object."""
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)

def get_onboarding_content():
    """Fetches mixed content. Tries API -> Caches -> Returns. If API fails, returns Cached/Static."""
    all_content = []
    seen_ids = set()
    
    # 1. Fetch from APIs if keys exist
    if config.TMDB_API_KEY != "YOUR_TMDB_KEY":
        try:
            discover = Discover()
            genres = [28, 35, 18, 878, 99]
            for g_id in genres:
                res = discover.discover_movies({'sort_by': 'popularity.desc', 'with_genres': g_id, 'vote_count.gte': 500, 'page': 1})
                
                # Handle raw dict response
                if isinstance(res, dict) and 'results' in res: 
                    res = res['results']
                else:
                    # Convert AsObj/Iterable to list to support slicing
                    res = list(res)
                
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
        except Exception as e:
            print(f"TMDB Onboarding Error: {e}")
            traceback.print_exc()

    if config.YOUTUBE_API_KEY != "YOUR_YOUTUBE_KEY":
        # ... fetch youtube ...
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
                res = youtube.search().list(q=topic, part='id,snippet', maxResults=6, type='video', videoDuration='medium').execute()
                for item in res.get('items', []):
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
        except Exception as e:
            print(f"YouTube Onboarding Error: {e}")

    # 2. If we got content, Cache It!
    if all_content:
        cache_content_to_db(all_content)
        random.shuffle(all_content)
        return all_content

    # 3. If API failed or keys missing, fetch from Cache/Static
    cached = get_random_cached_content(limit=30)
    if len(cached) > 5:
        return cached
    
    # 4. Fallback to Static Pool (and cache it for next time)
    static = get_static_content_pool()
    cache_content_to_db(static)
    random.shuffle(static)
    return static

def search_person_id(name):
    # ... (same as before)
    try:
        if config.TMDB_API_KEY == "YOUR_TMDB_KEY": return None
        search = Person()
        res = search.search(name)
        if res: return res[0].id
    except: return None
    return None

def fetch_movie_recommendations(subscriptions, watched_movies, actors, directors, max_time, focus_mode, mood):
    """
    Hybrid Search: API -> Cache -> Vector Search fallback.
    """
    api_results = []
    error_msg = None
    
    # A. Try Live API (Recall)
    if config.TMDB_API_KEY != "YOUR_TMDB_KEY":
        try:
            discover = Discover()
            kwargs = {
                'sort_by': 'popularity.desc', 'vote_average.gte': 6.5,
                'with_runtime.lte': max_time, 'page': 1
            }
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
            traceback.print_exc()
            error_msg = f"TMDB API Error: {str(e)}"

    # B. Cache fresh results if any
    if api_results:
        cache_content_to_db(api_results)
    
    # C. If API failed or yielded 0 results, query Vector DB (Semantic Fallback)
    candidates = api_results
    if not candidates and mood:
        # Search DB for movies
        candidates = query_vector_db(mood, n_results=15, where_filter={"type": "movie"})
        if candidates:
            error_msg = "Showing similar cached movies (API unavailable/empty)." if error_msg else "Showing semantically similar movies from cache."
    
    # If still empty, use Static Pool
    if not candidates:
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
        # Use our manual re-ranker for the fresh API batch
        candidates = semantic_rerank(candidates, mood, text_key='overview', top_k=10)
        for item in candidates:
            score = item.get('similarity_score', 0)
            item['match_reason'] = f"<b>{int(score*100)}% Match</b> to '{mood}'"

    return candidates[:10], error_msg

def fetch_video_recommendations(mood, max_time_mins=40):
    api_results = []
    error_msg = None

    if config.YOUTUBE_API_KEY != "YOUR_YOUTUBE_KEY":
        try:
            youtube = build('youtube', 'v3', developerKey=config.YOUTUBE_API_KEY)
            # Query for substantial content
            res = youtube.search().list(q=f"{mood} deep dive video essay analysis documentary", part='id,snippet', maxResults=15, type='video', videoDuration='long').execute()
            
            for item in res.get('items', []):
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
            error_msg = f"YouTube API Error: {str(e)}"

    if api_results:
        cache_content_to_db(api_results)
    
    candidates = api_results
    if not candidates and mood:
        candidates = query_vector_db(mood, n_results=10, where_filter={"type": "video"})
        if candidates: error_msg = "Showing cached videos."

    if not candidates:
         pool = [x for x in get_static_content_pool() if x['type'] == 'video']
         candidates = pool[:5]

    if mood and api_results:
         candidates = semantic_rerank(candidates, mood, 'description', top_k=5)
         for item in candidates:
            score = item.get('similarity_score', 0)
            item['match_reason'] = f"<b>{int(score*100)}% Match</b> to '{mood}'"

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