#!/usr/bin/env python3
"""
Database Seeding Script for MindfulWatch

Pre-populates the vector database with high-quality content from:
1. TMDB API - Movies from all genres
2. yt-dlp - High-value YouTube videos (educational, documentaries, essays)

Run this script once to seed the database, or periodically to refresh content.
"""

import sys
import time
import random
from datetime import datetime

# Add the project root to path
sys.path.insert(0, '.')

from utils import (
    cache_content_to_db, 
    get_vector_collection, 
    fetch_tmdb_discover,
    search_youtube_ytdlp,
    YT_DLP_AVAILABLE
)
from logging_config import logger
import config

# ============================================================================
# HIGH-VALUE YOUTUBE SEARCH QUERIES (200+ curated queries)
# Focus on: Educational, Documentary, Analysis, Deep Dives, Essays
# Avoid: Clickbait, Reactions, Drama, Low-effort content
# ============================================================================

YOUTUBE_QUERIES = [
    # === SCIENCE & PHYSICS ===
    "quantum mechanics explained documentary",
    "string theory documentary physics",
    "black holes explained astrophysics",
    "general relativity explained",
    "particle physics CERN documentary",
    "dark matter dark energy documentary",
    "cosmology origin universe documentary",
    "thermodynamics explained physics",
    "electromagnetic waves explained",
    "nuclear physics documentary",
    "astronomy deep dive",
    "James Webb telescope discoveries",
    "evolution of stars documentary",
    "quantum computing explained",
    "superconductivity explained",
    
    # === MATHEMATICS ===
    "history of mathematics documentary",
    "prime numbers explained",
    "infinity mathematics documentary",
    "Gödel incompleteness theorem",
    "topology mathematics explained",
    "calculus history documentary",
    "game theory explained",
    "chaos theory documentary",
    "fractals mathematics explained",
    "Riemann hypothesis explained",
    "number theory documentary",
    "mathematical proofs explained",
    
    # === BIOLOGY & NATURE ===
    "evolution documentary David Attenborough",
    "genetics DNA documentary",
    "microbiology bacteria documentary",
    "neuroscience brain documentary",
    "deep sea creatures documentary",
    "rainforest ecosystem documentary",
    "animal behavior documentary",
    "plant biology documentary",
    "immune system explained",
    "cell biology documentary",
    "ecology documentary",
    "endangered species conservation documentary",
    "marine biology ocean documentary",
    "insects documentary",
    "bird migration documentary",
    
    # === HISTORY ===
    "ancient Rome documentary",
    "ancient Egypt civilization documentary",
    "World War II documentary",
    "World War I documentary",
    "Cold War history documentary",
    "Renaissance history documentary",
    "Medieval Europe documentary",
    "ancient Greece documentary",
    "Industrial Revolution documentary",
    "French Revolution documentary",
    "American Civil War documentary",
    "Byzantine Empire documentary",
    "Ottoman Empire documentary",
    "Mongol Empire documentary",
    "Viking history documentary",
    "ancient Mesopotamia documentary",
    "Chinese dynasties history documentary",
    "Japanese history documentary",
    "African kingdoms history documentary",
    "Mayan civilization documentary",
    "Inca Empire documentary",
    "Persian Empire documentary",
    "Alexander the Great documentary",
    "Napoleon Bonaparte documentary",
    "history of warfare documentary",
    
    # === PHILOSOPHY ===
    "philosophy of mind documentary",
    "existentialism explained Sartre",
    "Stoicism philosophy explained",
    "Plato philosophy explained",
    "Aristotle philosophy documentary",
    "Nietzsche philosophy explained",
    "Kant ethics explained",
    "philosophy of science documentary",
    "Eastern philosophy documentary",
    "Buddhist philosophy explained",
    "ethics moral philosophy",
    "free will determinism debate",
    "consciousness philosophy explained",
    "metaphysics documentary",
    "epistemology knowledge explained",
    "political philosophy documentary",
    "philosophy of religion documentary",
    
    # === TECHNOLOGY & ENGINEERING ===
    "how computers work documentary",
    "history of the internet documentary",
    "artificial intelligence documentary",
    "machine learning explained",
    "robotics engineering documentary",
    "aerospace engineering documentary",
    "civil engineering megastructures",
    "electrical engineering explained",
    "chemical engineering documentary",
    "materials science documentary",
    "nanotechnology documentary",
    "renewable energy technology documentary",
    "nuclear power plant documentary",
    "semiconductor manufacturing documentary",
    "space exploration technology",
    "Mars colonization documentary",
    "electric vehicles technology",
    "blockchain technology explained",
    "cybersecurity documentary",
    "biotechnology documentary",
    
    # === ECONOMICS & FINANCE ===
    "history of money documentary",
    "economics explained documentary",
    "stock market history documentary",
    "2008 financial crisis documentary",
    "Federal Reserve explained",
    "cryptocurrency explained",
    "inflation economics explained",
    "behavioral economics documentary",
    "wealth inequality documentary",
    "globalization economics documentary",
    "trade economics documentary",
    "banking system explained",
    "economic history documentary",
    "capitalism documentary",
    "socialism economics documentary",
    
    # === PSYCHOLOGY ===
    "cognitive psychology documentary",
    "social psychology experiments",
    "developmental psychology documentary",
    "memory psychology explained",
    "perception psychology documentary",
    "personality psychology explained",
    "abnormal psychology documentary",
    "behavioral psychology documentary",
    "emotional intelligence explained",
    "decision making psychology",
    "cognitive biases explained",
    "psychology of happiness",
    "psychology of motivation",
    "group psychology documentary",
    
    # === ART & CULTURE ===
    "history of art documentary",
    "Renaissance art documentary",
    "modern art documentary",
    "Impressionism art movement",
    "Baroque art documentary",
    "classical music documentary",
    "jazz history documentary",
    "rock music history documentary",
    "architecture history documentary",
    "film history documentary",
    "photography history documentary",
    "literature analysis video essay",
    "theater history documentary",
    "sculpture art documentary",
    "street art graffiti documentary",
    
    # === VIDEO ESSAYS (HIGH QUALITY CHANNELS) ===
    "Kurzgesagt science",
    "Veritasium science explained",
    "3Blue1Brown mathematics",
    "Numberphile math",
    "Vsauce explained",
    "Tom Scott interesting places",
    "Wendover Productions",
    "Polymatter explained",
    "RealLifeLore geography",
    "Half as Interesting",
    "CGP Grey explained",
    "Philosophy Tube",
    "Contrapoints video essay",
    "Every Frame a Painting",
    "Lessons from the Screenplay",
    "Nerdwriter1 video essay",
    "Vox explained documentary",
    "Johnny Harris documentary",
    "Captain Disillusion explained",
    "Technology Connections",
    "Practical Engineering",
    "Real Engineering explained",
    "Mustard aviation",
    "Atomic Frontier nuclear",
    "Neo explained",
    "Aperture philosophy",
    "exurb1a philosophy",
    
    # === COOKING & FOOD SCIENCE ===
    "food science documentary",
    "cooking chemistry explained",
    "culinary history documentary",
    "fermentation science",
    "baking science explained",
    "world cuisines documentary",
    "sustainable food documentary",
    "food production documentary",
    
    # === SPACE & ASTRONOMY ===
    "NASA documentary",
    "SpaceX documentary",
    "Apollo missions documentary",
    "planetary science documentary",
    "exoplanets documentary",
    "solar system documentary",
    "galaxy formation documentary",
    "International Space Station documentary",
    "Hubble telescope documentary",
    "space exploration history",
    "future of space travel",
    "colonizing other planets",
    
    # === MEDICINE & HEALTH ===
    "medical history documentary",
    "anatomy explained",
    "surgery documentary",
    "vaccine science explained",
    "cancer research documentary",
    "mental health documentary",
    "epidemiology documentary",
    "pharmaceutical industry documentary",
    "nutrition science documentary",
    "sleep science documentary",
    "exercise physiology explained",
    
    # === GEOGRAPHY & EARTH SCIENCE ===
    "plate tectonics documentary",
    "volcano documentary",
    "earthquake science documentary",
    "climate science documentary",
    "oceanography documentary",
    "meteorology weather documentary",
    "geology documentary",
    "glacier documentary",
    "desert ecosystem documentary",
    "river systems documentary",
    "geopolitics explained",
    "maps cartography history",
    
    # === LINGUISTICS & LANGUAGE ===
    "linguistics documentary",
    "history of English language",
    "language evolution documentary",
    "phonetics explained",
    "etymology word origins",
    "sign language documentary",
    "ancient languages documentary",
    "translation interpretation documentary",
    
    # === SOCIOLOGY & ANTHROPOLOGY ===
    "anthropology documentary",
    "sociology documentary",
    "cultural anthropology",
    "archaeology documentary",
    "ancient civilizations documentary",
    "human evolution documentary",
    "tribal cultures documentary",
    "urban sociology documentary",
    
    # === INVESTIGATIVE & JOURNALISM ===
    "investigative journalism documentary",
    "corruption exposed documentary",
    "environmental documentary",
    "social issues documentary",
    "human rights documentary",
    "privacy surveillance documentary",
    "technology ethics documentary",
    
    # === MATHEMATICS FOR CURIOUS ===
    "Numberphile prime numbers",
    "3Blue1Brown linear algebra",
    "Stand-up Maths",
    "Mathologer",
    "mathematical puzzles explained",
    "probability explained",
    "statistics explained",
    
    # === DEEP DIVES & ANALYSIS ===
    "film analysis video essay",
    "book analysis video essay",
    "game design analysis",
    "music theory analysis",
    "architectural analysis",
    "historical analysis documentary",
    "scientific analysis documentary",
    
    # === CRITICAL THINKING ===
    "logical fallacies explained",
    "critical thinking documentary",
    "scientific method explained",
    "skepticism documentary",
    "misinformation documentary",
    "media literacy explained",
    
    # === BIOGRAPHIES & PROFILES ===
    "Einstein biography documentary",
    "Marie Curie documentary",
    "Nikola Tesla documentary",
    "Leonardo da Vinci documentary",
    "Alan Turing documentary",
    "Stephen Hawking documentary",
    "Richard Feynman lectures",
    "Carl Sagan cosmos",
    "Neil deGrasse Tyson",
    "great scientists documentary",
    "inventors history documentary",
    "philosophers biography documentary",
    
    # === ENVIRONMENT & SUSTAINABILITY ===
    "climate change documentary",
    "renewable energy documentary",
    "plastic pollution documentary",
    "deforestation documentary",
    "wildlife conservation documentary",
    "sustainable living documentary",
    "ocean conservation documentary",
    "air pollution documentary",
    "water crisis documentary",
    "circular economy explained",
    
    # === MUSIC THEORY & COMPOSITION ===
    "music theory explained",
    "classical music analysis",
    "composition techniques explained",
    "harmony explained music",
    "rhythm music theory",
    "orchestration explained",
    "electronic music production",
    "sound design explained",
    
    # === ARCHITECTURE & DESIGN ===
    "architectural history documentary",
    "famous buildings documentary",
    "sustainable architecture",
    "interior design documentary",
    "urban planning documentary",
    "industrial design documentary",
    "graphic design documentary",
    
    # === CRAFTS & MAKING ===
    "how its made documentary",
    "craftsmanship documentary",
    "woodworking documentary",
    "metalworking documentary",
    "pottery ceramics documentary",
    "glassblowing documentary",
    "textile manufacturing documentary",
    "traditional crafts documentary",
]

# ============================================================================
# TMDB GENRE IDS (All genres for comprehensive movie coverage)
# ============================================================================

TMDB_GENRES = {
    28: "Action",
    12: "Adventure", 
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    99: "Documentary",
    18: "Drama",
    10751: "Family",
    14: "Fantasy",
    36: "History",
    27: "Horror",
    10402: "Music",
    9648: "Mystery",
    10749: "Romance",
    878: "Science Fiction",
    10770: "TV Movie",
    53: "Thriller",
    10752: "War",
    37: "Western",
}

# ============================================================================
# SEEDING FUNCTIONS
# ============================================================================

def seed_movies(max_per_genre=30):
    """Fetch movies from all TMDB genres."""
    if not config.TMDB_API_KEY or config.TMDB_API_KEY == "YOUR_TMDB_KEY":
        logger.warning("TMDB API key not configured. Skipping movie seeding.")
        return 0
    
    logger.info("=" * 60)
    logger.info("SEEDING MOVIES FROM TMDB")
    logger.info("=" * 60)
    
    all_movies = []
    seen_ids = set()
    
    for genre_id, genre_name in TMDB_GENRES.items():
        logger.info(f"Fetching {genre_name} movies...")
        
        # Fetch popular movies in this genre
        movies = fetch_tmdb_discover(
            params={
                "with_genres": genre_id,
                "vote_count.gte": 100,
                "sort_by": "popularity.desc"
            },
            max_pages=3
        )
        
        count = 0
        for movie in movies:
            if count >= max_per_genre:
                break
            
            movie_id = str(movie.get("id"))
            if movie_id in seen_ids:
                continue
            
            seen_ids.add(movie_id)
            movie["type"] = "movie"
            all_movies.append(movie)
            count += 1
        
        logger.info(f"  -> Got {count} {genre_name} movies")
        time.sleep(0.25)  # Rate limiting
    
    # Also fetch top-rated and popular lists
    for list_type in ["popular", "top_rated"]:
        logger.info(f"Fetching {list_type} movies...")
        movies = fetch_tmdb_discover(
            params={"sort_by": f"{list_type}.desc" if list_type == "popular" else "vote_average.desc"},
            max_pages=5
        )
        
        for movie in movies:
            movie_id = str(movie.get("id"))
            if movie_id not in seen_ids:
                seen_ids.add(movie_id)
                movie["type"] = "movie"
                all_movies.append(movie)
        
        time.sleep(0.25)
    
    logger.info(f"Total unique movies fetched: {len(all_movies)}")
    
    if all_movies:
        logger.info("Caching movies to vector database...")
        cache_content_to_db(all_movies)
    
    return len(all_movies)


def seed_videos(queries=None, results_per_query=10):
    """Fetch high-value YouTube videos using yt-dlp."""
    if not YT_DLP_AVAILABLE:
        logger.warning("yt-dlp not available. Skipping video seeding.")
        return 0
    
    if queries is None:
        queries = YOUTUBE_QUERIES
    
    logger.info("=" * 60)
    logger.info("SEEDING VIDEOS FROM YOUTUBE (via yt-dlp)")
    logger.info(f"Total queries: {len(queries)}")
    logger.info("=" * 60)
    
    all_videos = []
    seen_ids = set()
    failed_queries = []
    
    for i, query in enumerate(queries, 1):
        logger.info(f"[{i}/{len(queries)}] Searching: '{query}'")
        
        try:
            videos = search_youtube_ytdlp(query, max_results=results_per_query)
            
            count = 0
            for video in videos:
                video_id = video.get("video_id") or video.get("id")
                if video_id and video_id not in seen_ids:
                    seen_ids.add(video_id)
                    all_videos.append(video)
                    count += 1
            
            logger.info(f"  -> Got {count} new videos")
            
            # Cache in batches to avoid memory issues
            if len(all_videos) >= 100:
                logger.info(f"  -> Caching batch of {len(all_videos)} videos...")
                cache_content_to_db(all_videos)
                all_videos = []
            
            # Small delay to be nice to YouTube
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"  -> Failed: {e}")
            failed_queries.append(query)
    
    # Cache remaining videos
    if all_videos:
        logger.info(f"Caching final batch of {len(all_videos)} videos...")
        cache_content_to_db(all_videos)
    
    total = len(seen_ids)
    logger.info(f"Total unique videos fetched: {total}")
    
    if failed_queries:
        logger.warning(f"Failed queries ({len(failed_queries)}): {failed_queries[:10]}...")
    
    return total


def get_db_stats():
    """Get current database statistics."""
    collection = get_vector_collection()
    if not collection:
        return {"error": "Could not connect to database"}
    
    try:
        # Get all items
        result = collection.get()
        total = len(result["ids"]) if result["ids"] else 0
        
        # Count by type
        movies = 0
        videos = 0
        for meta in result.get("metadatas", []):
            if meta.get("type") == "movie":
                movies += 1
            elif meta.get("type") == "video":
                videos += 1
        
        return {
            "total": total,
            "movies": movies,
            "videos": videos,
            "other": total - movies - videos
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    """Main seeding function."""
    print("\n" + "=" * 60)
    print("MindfulWatch Database Seeder")
    print("=" * 60 + "\n")
    
    # Check current stats
    print("Current database stats:")
    stats = get_db_stats()
    print(f"  Total items: {stats.get('total', 'N/A')}")
    print(f"  Movies: {stats.get('movies', 'N/A')}")
    print(f"  Videos: {stats.get('videos', 'N/A')}")
    print()
    
    start_time = datetime.now()
    
    # Seed movies
    movies_added = seed_movies(max_per_genre=30)
    print(f"\n✓ Added {movies_added} movies")
    
    # Seed videos
    videos_added = seed_videos(results_per_query=10)
    print(f"✓ Added {videos_added} videos")
    
    # Final stats
    print("\n" + "=" * 60)
    print("SEEDING COMPLETE")
    print("=" * 60)
    
    end_stats = get_db_stats()
    print(f"\nFinal database stats:")
    print(f"  Total items: {end_stats.get('total', 'N/A')}")
    print(f"  Movies: {end_stats.get('movies', 'N/A')}")
    print(f"  Videos: {end_stats.get('videos', 'N/A')}")
    
    elapsed = datetime.now() - start_time
    print(f"\nTime elapsed: {elapsed}")


if __name__ == "__main__":
    main()
