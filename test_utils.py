import pytest
from unittest.mock import MagicMock, patch
import utils
import config
import numpy as np

# --- Mock Data ---
# Object-style mock that mimics TMDB's AsObj (has both .attr and .get() access)
MOCK_MOVIE_OBJ = MagicMock()
MOCK_MOVIE_OBJ.title = "Obj Movie"
MOCK_MOVIE_OBJ.overview = "Obj Desc"
MOCK_MOVIE_OBJ.release_date = "2023-01-01"
MOCK_MOVIE_OBJ.vote_average = 8.0
MOCK_MOVIE_OBJ.poster_path = "/obj.jpg"
MOCK_MOVIE_OBJ.id = 101
MOCK_MOVIE_OBJ.runtime = 100
# Add get() method like TMDB's AsObj has
def mock_get(key, default=None):
    return getattr(MOCK_MOVIE_OBJ, key, default)
MOCK_MOVIE_OBJ.get = mock_get

# Dict-style mock matching TMDB API response format
MOCK_MOVIE_DICT = {
    "title": "Dict Movie",
    "overview": "Dict Desc",
    "release_date": "2023-01-01",
    "vote_average": 7.5,
    "poster_path": "/dict.jpg",
    "id": 102,
    "runtime": 90
}

# Mock TMDB API response (what /discover/movie returns)
MOCK_TMDB_RESPONSE = {
    "page": 1,
    "results": [
        {
            "id": 12345,
            "title": "Test Movie",
            "overview": "A great test movie",
            "poster_path": "/test.jpg",
            "vote_average": 8.5,
            "release_date": "2024-01-01",
            "popularity": 100.0
        }
    ],
    "total_pages": 1,
    "total_results": 1
}

# --- Fixtures ---

@pytest.fixture
def mock_deps():
    """Mock external dependencies: requests, YouTube, ChromaDB, SentenceTransformer"""
    with patch('utils.requests.get') as MockRequestsGet, \
         patch('utils.build') as MockBuild, \
         patch('utils.get_vector_collection') as MockGetCollection, \
         patch('utils.load_embedding_model') as MockLoadModel, \
         patch('utils.cosine_similarity') as MockCosine:
    
        # Setup mock response for requests.get (TMDB API)
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = MOCK_TMDB_RESPONSE
        MockRequestsGet.return_value = mock_response
        
        # YouTube
        mock_youtube = MockBuild.return_value
        mock_youtube.search.return_value.list.return_value.execute.return_value = {
            "items": [{"id": {"videoId": "vid1"}, "snippet": {"title": "Vid", "description": "Desc", "thumbnails": {"high": {"url": "url"}}}}] 
        }
        
        # Chroma
        mock_collection = MagicMock()
        MockGetCollection.return_value = mock_collection
        mock_collection.query.return_value = {'ids': [], 'metadatas': [], 'distances': []} # Empty DB
        
        # Models
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2]])
        MockLoadModel.return_value = mock_model
        MockCosine.return_value = [[0.9]]
        
        yield MockRequestsGet, mock_youtube, mock_collection

def test_fetch_movie_success(mock_deps):
    """Test that fetch_movie_recommendations returns movies with correct structure"""
    mock_requests_get, _, _ = mock_deps
    with patch('config.TMDB_API_KEY', 'REAL_KEY'):
        recs, err = utils.fetch_movie_recommendations(["Netflix"], "", "", "", 120, False, "")
        # Should return movies from 2 pages (same mock response returns for each page)
        assert len(recs) >= 1, "Should return at least 1 movie"
        assert recs[0]['title'] == "Test Movie"
        assert recs[0]['id'] == 12345
        assert 'poster_path' in recs[0]
        assert recs[0]['poster_path'].startswith('http')

def test_fetch_movie_dict_response(mock_deps):
    """Test that movies have proper poster URLs"""
    mock_requests_get, _, _ = mock_deps
    
    # Mock a different movie
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "page": 1,
        "results": [MOCK_MOVIE_DICT],
        "total_results": 1
    }
    mock_requests_get.return_value = mock_response
    
    with patch('config.TMDB_API_KEY', 'REAL_KEY'):
        recs, err = utils.fetch_movie_recommendations(["Netflix"], "", "", "", 120, False, "")
        
        assert len(recs) >= 1
        assert recs[0]['title'] == "Dict Movie"
        # Poster should be full URL with TMDB image path
        assert "image.tmdb.org" in recs[0]['poster_path'] or "placeholder" in recs[0]['poster_path']

def test_onboarding_tmdb_obj_response(mock_deps):
    """Test onboarding retrieval returns movies"""
    mock_requests_get, _, _ = mock_deps
    
    with patch('config.TMDB_API_KEY', 'REAL_KEY'):
        recs = utils.get_onboarding_content()
        
        # Should have movies from the mock response
        movies = [r for r in recs if r['type'] == 'movie']
        assert len(movies) > 0, "Should return at least one movie"
        assert movies[0]['title'] == "Test Movie"

def test_onboarding_tmdb_dict_response(mock_deps):
    """Test onboarding with different movie data"""
    mock_requests_get, _, _ = mock_deps
    
    # Override mock for this test
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "page": 1,
        "results": [MOCK_MOVIE_DICT],
        "total_results": 1
    }
    mock_requests_get.return_value = mock_response
    
    with patch('config.TMDB_API_KEY', 'REAL_KEY'):
        recs = utils.get_onboarding_content()
        
        movies = [r for r in recs if r['type'] == 'movie']
        assert len(movies) > 0
        assert movies[0]['title'] == "Dict Movie"

# ==================== RIGOROUS TESTS ====================

class TestSafeGet:
    """Test safe_get function handles all data types correctly"""
    
    def test_safe_get_with_dict(self):
        """safe_get should work with regular dicts"""
        data = {"title": "Test Movie", "id": 123}
        assert utils.safe_get(data, "title") == "Test Movie"
        assert utils.safe_get(data, "id") == 123
        assert utils.safe_get(data, "missing", "default") == "default"
    
    def test_safe_get_with_object(self):
        """safe_get should work with objects having attributes"""
        obj = MagicMock()
        obj.title = "Object Title"
        obj.id = 456
        # Add get() method like TMDB's AsObj
        obj.get = lambda key, default=None: getattr(obj, key, default)
        
        assert utils.safe_get(obj, "title") == "Object Title"
        assert utils.safe_get(obj, "id") == 456
    
    def test_safe_get_with_string_should_not_return_method(self):
        """safe_get on a string should NOT return str.title method"""
        result = utils.safe_get("hello world", "title", "default")
        # Should return default, not the str.title method
        assert result == "default" or result is None or isinstance(result, str)
        # CRITICAL: Should never be a callable method
        assert not callable(result) or isinstance(result, str)
    
    def test_safe_get_returns_none_for_none(self):
        """safe_get should handle None gracefully"""
        assert utils.safe_get(None, "title", "default") == "default"


class TestMovieOutputStructure:
    """Test that movie recommendations have correct structure"""
    
    def test_movie_has_required_fields(self, mock_deps):
        """Each movie must have title, id, poster_path, and type"""
        mock_discover, _, _ = mock_deps
        
        with patch('config.TMDB_API_KEY', 'REAL_KEY'):
            recs, err = utils.fetch_movie_recommendations(
                ["Netflix"], "", "", "", 120, False, ""
            )
            
            assert len(recs) > 0, "Should return at least one movie"
            
            for i, movie in enumerate(recs):
                # REQUIRED: title must be a non-empty string
                assert 'title' in movie, f"Movie {i} missing 'title'"
                assert isinstance(movie['title'], str), f"Movie {i} title must be string"
                assert len(movie['title']) > 0, f"Movie {i} has empty title"
                
                # REQUIRED: id must be a valid ID (int or non-empty string)
                assert 'id' in movie, f"Movie {i} missing 'id'"
                assert movie['id'], f"Movie {i} has falsy id: {movie['id']}"
                
                # REQUIRED: poster_path must be a URL string
                assert 'poster_path' in movie, f"Movie {i} missing 'poster_path'"
                assert isinstance(movie['poster_path'], str), f"Movie {i} poster_path must be string"
                assert movie['poster_path'].startswith('http'), f"Movie {i} poster_path must be URL"
                
                # REQUIRED: type must be 'movie'
                assert movie.get('type') == 'movie', f"Movie {i} type must be 'movie'"
    
    def test_movie_title_is_not_method_string(self, mock_deps):
        """Movie title must not be a method representation string"""
        mock_discover, _, _ = mock_deps
        
        with patch('config.TMDB_API_KEY', 'REAL_KEY'):
            recs, err = utils.fetch_movie_recommendations(
                ["Netflix"], "", "", "", 120, False, ""
            )
            
            for i, movie in enumerate(recs):
                title = movie.get('title', '')
                # Title should never contain "<built-in method" 
                assert '<built-in method' not in title, f"Movie {i} title is a method repr: {title}"
                assert '<MagicMock' not in title, f"Movie {i} title is a mock repr: {title}"
    
    def test_movie_poster_is_valid_url(self, mock_deps):
        """Movie poster_path must be a valid http URL, not a relative path"""
        mock_discover, _, _ = mock_deps
        
        with patch('config.TMDB_API_KEY', 'REAL_KEY'):
            recs, err = utils.fetch_movie_recommendations(
                ["Netflix"], "", "", "", 120, False, ""
            )
            
            for i, movie in enumerate(recs):
                poster = movie.get('poster_path', '')
                # Poster should be full URL (http://...) not relative (/path.jpg)
                assert poster.startswith('http'), f"Movie {i} poster not a full URL: {poster}"


class TestDataValidation:
    """Test that invalid data is properly filtered out"""
    
    def test_invalid_movies_are_skipped(self, mock_deps):
        """Movies with empty title or id should be filtered out"""
        mock_discover, _, _ = mock_deps
        
        # Create mock with invalid items mixed in
        invalid_obj = MagicMock()
        invalid_obj.title = ""  # Empty title
        invalid_obj.id = 0  # Zero id
        invalid_obj.get = lambda key, default=None: getattr(invalid_obj, key, default)
        
        valid_obj = MagicMock()
        valid_obj.title = "Valid Movie"
        valid_obj.id = 999
        valid_obj.overview = "Description"
        valid_obj.poster_path = "/valid.jpg"
        valid_obj.get = lambda key, default=None: getattr(valid_obj, key, default)
        
        mock_discover.discover_movies.return_value = [invalid_obj, valid_obj]
        
        with patch('config.TMDB_API_KEY', 'REAL_KEY'):
            recs, err = utils.fetch_movie_recommendations(
                ["Netflix"], "", "", "", 120, False, ""
            )
            
            # Only valid movies should be returned
            for movie in recs:
                assert movie['title'] != "", "Empty title movie should be filtered"
                assert movie['id'] != 0, "Zero id movie should be filtered"
