import os
from dotenv import load_dotenv
from tmdbv3api import TMDb, Discover
from googleapiclient.discovery import build
import sys

# Force reload of env vars
load_dotenv(override=True)

TMDB_KEY = os.getenv("TMDB_API_KEY")
YT_KEY = os.getenv("YOUTUBE_API_KEY")

print(f"--- API Diagnostic ---")
print(f"TMDB Key Present: {bool(TMDB_KEY and TMDB_KEY != 'YOUR_TMDB_KEY')}")
print(f"YouTube Key Present: {bool(YT_KEY and YT_KEY != 'YOUR_YOUTUBE_KEY')}")
print("-" * 20)

# 1. Test TMDB
print("\nTesting TMDB API...")
if TMDB_KEY and TMDB_KEY != "YOUR_TMDB_KEY":
    try:
        tmdb = TMDb()
        tmdb.api_key = TMDB_KEY
        tmdb.language = 'en'
        tmdb.debug = False # Reduce noise

        discover = Discover()
        # Try specific query used in app
        print("Sending request to discover.discover_movies...")
        res = discover.discover_movies({
            'sort_by': 'popularity.desc',
            'page': 1
        })
        
        # Check raw type
        print(f"Raw Result Type: {type(res)}")
        
        items = []
        if isinstance(res, dict) and 'results' in res:
            items = res['results']
            print("Detected Dictionary response.")
        elif hasattr(res, '__iter__'):
            items = list(res)
            print("Detected List/Iterable response.")
        
        if items:
            print(f"✅ Success! Found {len(items)} movies.")
            first = items[0]
            if isinstance(first, dict):
                print(f"Sample Movie (Dict): {first.get('title', 'No Title')}")
            else:
                print(f"Sample Movie (Obj): {getattr(first, 'title', 'No Title')}")
        else:
            print("❌ Request succeeded but returned 0 movies.")
            
    except Exception as e:
        print(f"❌ TMDB Failed: {e}")
else:
    print("⚠️  Skipping TMDB (Key missing or default)")

# 2. Test YouTube
print("\nTesting YouTube API...")
if YT_KEY and YT_KEY != "YOUR_YOUTUBE_KEY":
    try:
        youtube = build('youtube', 'v3', developerKey=YT_KEY)
        req = youtube.search().list(
            q="science",
            part='snippet',
            maxResults=1,
            type='video'
        )
        res = req.execute()
        
        items = res.get('items', [])
        if items:
            print(f"✅ Success! Found {len(items)} videos.")
            print(f"Sample Video: {items[0]['snippet']['title']}")
        else:
            print("❌ Request succeeded but returned 0 videos.")
            
    except Exception as e:
        print(f"❌ YouTube Failed: {e}")
else:
    print("⚠️  Skipping YouTube (Key missing or default)")

print("\n--- End Diagnostic ---")
