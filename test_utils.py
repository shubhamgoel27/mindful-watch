import pytest
from unittest.mock import MagicMock, patch
import utils
import config

# Mock Data Constants
MOCK_MOVIE_RESULT = MagicMock()
MOCK_MOVIE_RESULT.title = "Test Movie"
MOCK_MOVIE_RESULT.overview = "Test Overview"
MOCK_MOVIE_RESULT.release_date = "2023-01-01"
MOCK_MOVIE_RESULT.vote_average = 8.0
MOCK_MOVIE_RESULT.poster_path = "/path.jpg"
MOCK_MOVIE_RESULT.id = 123
# Use configure_mock to set attributes that might be accessed dynamically or if they don't exist on the class
MOCK_MOVIE_RESULT.configure_mock(runtime=100) 

@pytest.fixture
def mock_tmdb_setup():
    """Fixture to mock TMDB API setup and Discover call"""
    with patch('utils.Discover') as MockDiscover, \
         patch('utils.Person') as MockPerson:
        
        # Setup Discover Mock
        mock_discover_instance = MockDiscover.return_value
        mock_discover_instance.discover_movies.return_value = [MOCK_MOVIE_RESULT]
        
        # Setup Person Mock (for actor search)
        mock_person_instance = MockPerson.return_value
        mock_person_result = MagicMock()
        mock_person_result.id = 999
        mock_person_instance.search.return_value = [mock_person_result]
        
        yield mock_discover_instance, mock_person_instance

@pytest.fixture
def mock_youtube_setup():
    """Fixture to mock Google API Client"""
    with patch('utils.build') as mock_build:
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        
        # Mock Search
        mock_search = mock_service.search.return_value.list.return_value.execute
        mock_search.return_value = {
            "items": [
                {
                    "id": {"videoId": "test_vid_1"},
                    "snippet": {
                        "title": "Test Video",
                        "description": "Test Desc",
                        "thumbnails": {"high": {"url": "http://thumb.jpg"}}
                    }
                }
            ]
        }
        
        # Mock Video Details (Duration)
        mock_videos = mock_service.videos.return_value.list.return_value.execute
        mock_videos.return_value = {
            "items": [
                {
                    "contentDetails": {"duration": "PT30M"} # 30 mins
                }
            ]
        }
        
        yield mock_service

def test_fetch_movie_recommendations_success(mock_tmdb_setup):
    """Test fetching movies works with valid inputs"""
    mock_discover, _ = mock_tmdb_setup
    
    # Force API Key to be "set" so we don't trigger the explicit fallback check in utils.py
    # (assuming utils.py checks config.TMDB_API_KEY)
    with patch('config.TMDB_API_KEY', 'REAL_KEY'):
        recs = utils.fetch_movie_recommendations(
            subscriptions=["Netflix"],
            watched_movies="",
            actors="Some Actor",
            directors="",
            max_time=120,
            focus_mode=False,
            mood="relax"
        )
        
        assert len(recs) == 1
        assert recs[0]['title'] == "Test Movie"
        assert recs[0]['runtime'] == 100
        mock_discover.discover_movies.assert_called_once()

def test_fetch_movie_recommendations_filters_watched(mock_tmdb_setup):
    """Test that watched movies are filtered out"""
    mock_discover, _ = mock_tmdb_setup
    
    with patch('config.TMDB_API_KEY', 'REAL_KEY'):
        # Pass "Test Movie" as watched
        recs = utils.fetch_movie_recommendations(
            subscriptions=[],
            watched_movies="Test Movie",
            actors="",
            directors="",
            max_time=120,
            focus_mode=False,
            mood=""
        )
        
        # Should return 0 results because the only mock result was filtered out
        # OR it falls back to mock data if empty (logic in utils: return final_recs if final_recs else get_mock_movies())
        # Let's check what utils.py does. 
        # utils.py: returns get_mock_movies() if list is empty.
        
        assert len(recs) > 0
        # Since our mock returned 1 movie and we filtered it, utils returns get_mock_movies()
        # Mock data has 3 items
        assert len(recs) == 3 
        assert recs[0]['title'] == "The Mindful Coder" # From fallback mock data

def test_fetch_video_recommendations_success(mock_youtube_setup):
    """Test fetching videos works"""
    
    with patch('config.YOUTUBE_API_KEY', 'REAL_KEY'):
        recs = utils.fetch_video_recommendations(mood="happy", max_time_mins=40)
        
        assert len(recs) == 1
        assert recs[0]['title'] == "Test Video"
        assert recs[0]['duration'] == "30 mins"

def test_fetch_video_recommendations_duration_filter(mock_youtube_setup):
    """Test videos are filtered by duration"""
    mock_service = mock_youtube_setup
    
    # Change duration mock to 10 mins (too short, min is 20)
    mock_videos = mock_service.videos.return_value.list.return_value.execute
    mock_videos.return_value = {
        "items": [
            {
                "contentDetails": {"duration": "PT10M"} 
            }
        ]
    }
    
    with patch('config.YOUTUBE_API_KEY', 'REAL_KEY'):
        recs = utils.fetch_video_recommendations(mood="happy")
        
        # Should fall back to mock data because the only result was filtered out
        assert len(recs) > 0
        assert recs[0]['video_id'] == "dQw4w9WgXcQ" # ID from get_mock_videos()

def test_fallback_when_keys_missing():
    """Test that functions return mock data immediately if keys are placeholders"""
    
    with patch('config.TMDB_API_KEY', 'YOUR_TMDB_KEY'):
        recs = utils.fetch_movie_recommendations([], "", "", "", 120, False, "")
        assert recs[0]['title'] == "The Mindful Coder" # Verify it's the mock data

    with patch('config.YOUTUBE_API_KEY', 'YOUR_YOUTUBE_KEY'):
        recs = utils.fetch_video_recommendations("mood")
        assert recs[0]['video_id'] == "dQw4w9WgXcQ"
