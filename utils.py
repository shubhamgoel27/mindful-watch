import random
import requests
import json
import os
from googleapiclient.discovery import build
from tmdbv3api import TMDb, Movie, Discover, Person, Configuration
import isodate
import config

# Setup TMDB
tmdb = TMDb()
tmdb.api_key = config.TMDB_API_KEY
tmdb.language = 'en'
tmdb.debug = True

# --- User Data Management ---
def load_user_data():
    """Loads user data from local JSON file."""
    if not os.path.exists(config.USER_DATA_FILE):
        return {}
    try:
        with open(config.USER_DATA_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_user_data(data):
    """Saves user data to local JSON file."""
    with open(config.USER_DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# --- Onboarding Logic ---

def get_static_content_pool():
    """Returns a large, diverse list of mock/fallback content with valid placeholder images."""
    
    def movie_item(id, title, genre):
        color = f"{random.randint(0, 255):02x}{random.randint(0, 255):02x}{random.randint(0, 255):02x}"
        return {
            "id": f"mock_m_{id}",
            "title": title,
            "type": "movie",
            "poster_path": f"https://placehold.co/300x450/{color}/FFF?text={title.replace(' ', '+')}",
            "overview": f"A classic {genre} movie that everyone talks about.",
            "match_reason": f"Popular in {genre}"
        }

    def video_item(id, title, category):
        color = f"{random.randint(0, 255):02x}{random.randint(0, 255):02x}{random.randint(0, 255):02x}"
        return {
            "id": f"mock_v_{id}",
            "title": title,
            "type": "video",
            "poster_path": f"https://placehold.co/400x225/{color}/FFF?text={title.replace(' ', '+')}",
            "overview": f"An engaging video about {category}.",
            "match_reason": f"Trending in {category}"
        }

    content = []
    
    # Movies (Diverse Genres)
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
    
    # Videos (Educational & Fun)
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
        content.append(movie_item(i, t, g))
        
    for i, (t, c) in enumerate(videos):
        content.append(video_item(i, t, c))

    random.shuffle(content)
    return content

def get_onboarding_content():
    """Fetches a mix of movies and videos for onboarding. Uses APIs if available, else static pool."""
    
    # 1. Use Static Pool if Keys are missing
    if config.TMDB_API_KEY == "YOUR_TMDB_KEY":
        return get_static_content_pool()

    all_content = []
    
    # 2. Fetch Movies (TMDB)
    try:
        discover = Discover()
        # Fetch pages of popular movies from different genres to ensure variety
        # Genres: 28=Action, 35=Comedy, 18=Drama, 878=Sci-Fi, 99=Documentary
        genres = [28, 35, 18, 878, 99]
        for g_id in genres:
            results = discover.discover_movies({
                'sort_by': 'popularity.desc',
                'with_genres': g_id,
                'vote_count.gte': 500,
                'page': 1
            })
            for m in results[:5]: # Take top 5 from each genre
                if m.poster_path:
                    all_content.append({
                        "id": str(m.id),
                        "title": m.title,
                        "type": "movie",
                        "poster_path": f"https://image.tmdb.org/t/p/w300{m.poster_path}",
                        "overview": m.overview
                    })
    except Exception as e:
        print(f"Error fetching movies: {e}")

    # 3. Fetch Videos (YouTube) - optional for onboarding but good for mix
    if config.YOUTUBE_API_KEY != "YOUR_YOUTUBE_KEY":
        try:
            youtube = build('youtube', 'v3', developerKey=config.YOUTUBE_API_KEY)
            # Search for broad popular educational/interesting channels or topics
            topics = ["science explained", "short documentary", "interesting facts", "movie video essay"]
            
            for topic in topics:
                search_response = youtube.search().list(
                    q=topic,
                    part='id,snippet',
                    maxResults=5,
                    type='video',
                    videoDuration='medium' # 4-20 mins
                ).execute()

                for item in search_response.get('items', []):
                    all_content.append({
                        "id": item['id']['videoId'],
                        "title": item['snippet']['title'],
                        "type": "video",
                        "poster_path": item['snippet']['thumbnails']['high']['url'],
                        "overview": item['snippet']['description']
                    })
        except Exception as e:
            print(f"Error fetching videos: {e}")

    # If API calls returned nothing (or failed), fallback
    if not all_content:
        return get_static_content_pool()

    random.shuffle(all_content)
    return all_content

# --- Recommendation Logic ---

def search_person_id(name):
    try:
        if config.TMDB_API_KEY == "YOUR_TMDB_KEY": return None
        search = Person()
        res = search.search(name)
        if res:
            return res[0].id
    except:
        return None
    return None

def fetch_movie_recommendations(subscriptions, watched_movies, actors, directors, max_time, focus_mode, mood):
    """
    Fetches movie recommendations using TMDB API or Mock Data.
    """
    if config.TMDB_API_KEY == "YOUR_TMDB_KEY":
        # Return a subset of the static pool that are movies
        pool = [x for x in get_static_content_pool() if x['type'] == 'movie']
        return pool[:5]

    try:
        discover = Discover()
        
        # Base Filters
        kwargs = {
            'sort_by': 'popularity.desc',
            'vote_average.gte': 7.0,
            'with_runtime.lte': max_time,
            'page': 1
        }

        # Subscriptions (Watch Providers)
        provider_ids = [str(config.PROVIDER_MAP[s]) for s in subscriptions if s in config.PROVIDER_MAP]
        if provider_ids:
            kwargs['with_watch_providers'] = '|'.join(provider_ids)
            kwargs['watch_region'] = 'US'

        # Actors & Directors
        people_ids = []
        if actors:
            for actor in actors.split(','):
                pid = search_person_id(actor.strip())
                if pid: people_ids.append(str(pid))
        
        if directors:
            for director in directors.split(','):
                pid = search_person_id(director.strip())
                if pid: people_ids.append(str(pid))
        
        if people_ids:
            kwargs['with_people'] = '|'.join(people_ids)

        # Focus Mode
        if focus_mode:
            kwargs['with_genres'] = ','.join([str(g) for g in config.FOCUS_GENRES])

        results = discover.discover_movies(kwargs)
        
        watched_list = [w.strip().lower() for w in watched_movies.split(',')]
        final_recs = []
        
        for m in results:
            if m.title.lower() not in watched_list:
                match_reason = f"Rated {m.vote_average}/10."
                if focus_mode: match_reason += " Fits Focus Mode genres."
                elif mood and mood.lower() in (m.overview or "").lower(): match_reason += f" Matches '{mood}'."
                
                final_recs.append({
                    "title": m.title,
                    "overview": m.overview,
                    "release_date": getattr(m, 'release_date', 'N/A'),
                    "vote_average": m.vote_average,
                    "runtime": getattr(m, 'runtime', 'N/A'),
                    "poster_path": f"https://image.tmdb.org/t/p/w500{m.poster_path}" if m.poster_path else None,
                    "match_reason": match_reason,
                    "id": m.id
                })
                
            if len(final_recs) >= 10:
                break
                
        return final_recs if final_recs else [x for x in get_static_content_pool() if x['type'] == 'movie'][:5]

    except Exception as e:
        print(f"TMDB Error: {e}")
        return [x for x in get_static_content_pool() if x['type'] == 'movie'][:5]

def fetch_video_recommendations(mood, max_time_mins=40):
    """
    Fetches YouTube videos using Google API or Mock Data.
    """
    if config.YOUTUBE_API_KEY == "YOUR_YOUTUBE_KEY":
        pool = [x for x in get_static_content_pool() if x['type'] == 'video']
        # Add fake fields expected by UI
        for p in pool:
            p['video_id'] = p['id'].replace('mock_v_', 'vid')
            p['description'] = p['overview']
            p['thumbnail'] = p['poster_path']
            p['duration'] = "25 mins"
        return pool[:5]

    try:
        youtube = build('youtube', 'v3', developerKey=config.YOUTUBE_API_KEY)
        
        query = f"fun informative {mood} explained"
        
        # Search for videos
        search_response = youtube.search().list(
            q=query,
            part='id,snippet',
            maxResults=10,
            type='video',
            videoDuration='long' # > 20 mins
        ).execute()

        videos = []
        for item in search_response.get('items', []):
            vid_id = item['id']['videoId']
            
            # Get content details for exact duration
            vid_response = youtube.videos().list(
                part='contentDetails,statistics',
                id=vid_id
            ).execute()
            
            if not vid_response['items']:
                continue
                
            details = vid_response['items'][0]
            duration_iso = details['contentDetails']['duration']
            duration_dt = isodate.parse_duration(duration_iso)
            duration_mins = duration_dt.total_seconds() / 60
            
            if 20 <= duration_mins <= (max_time_mins + 5):
                videos.append({
                    "title": item['snippet']['title'],
                    "description": item['snippet']['description'],
                    "thumbnail": item['snippet']['thumbnails']['high']['url'],
                    "video_id": vid_id,
                    "duration": f"{int(duration_mins)} mins",
                    "match_reason": f"Matches '{mood}' and duration criteria."
                })
                
            if len(videos) >= 5:
                break
        
        return videos if videos else []

    except Exception as e:
        print(f"YouTube Error: {e}")
        return []