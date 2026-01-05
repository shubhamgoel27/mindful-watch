import pytest
from unittest.mock import MagicMock, patch
import utils
import config
import numpy as np

# --- Mock Data ---
# Object-style mock
MOCK_MOVIE_OBJ = MagicMock()
MOCK_MOVIE_OBJ.title = "Obj Movie"
MOCK_MOVIE_OBJ.overview = "Obj Desc"
MOCK_MOVIE_OBJ.release_date = "2023-01-01"
MOCK_MOVIE_OBJ.vote_average = 8.0
MOCK_MOVIE_OBJ.poster_path = "/obj.jpg"
MOCK_MOVIE_OBJ.id = 101
MOCK_MOVIE_OBJ.runtime = 100

# Dict-style mock
MOCK_MOVIE_DICT = {
    "title": "Dict Movie",
    "overview": "Dict Desc",
    "release_date": "2023-01-01",
    "vote_average": 7.5,
    "poster_path": "/dict.jpg",
    "id": 102,
    "runtime": 90
}

# --- Fixtures ---

@pytest.fixture
def mock_deps():
    """Mock external dependencies: TMDB, YouTube, ChromaDB, SentenceTransformer"""
    with patch('utils.Discover') as MockDiscover, \
         patch('utils.Person') as MockPerson, \
         patch('utils.build') as MockBuild, \
         patch('utils.get_vector_collection') as MockGetCollection, \
         patch('utils.load_embedding_model') as MockLoadModel, \
         patch('utils.cosine_similarity') as MockCosine:
    
        # TMDB default (can be overridden in tests)
        mock_discover = MockDiscover.return_value
        mock_discover.discover_movies.return_value = [MOCK_MOVIE_OBJ]
        
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
        
        yield mock_discover, mock_youtube, mock_collection

def test_fetch_movie_success(mock_deps):
    mock_discover, _, _ = mock_deps
    with patch('config.TMDB_API_KEY', 'REAL_KEY'):
        recs, err = utils.fetch_movie_recommendations(["Netflix"], "", "", "", 120, False, "")
        # Expect 2 items (1 per page * 2 pages)
        assert len(recs) == 2 
        assert recs[0]['title'] == "Obj Movie"

def test_fetch_movie_dict_response(mock_deps):
    """Test robustness against raw JSON dict response from TMDB"""
    mock_discover, _, _ = mock_deps
    
    # Mock return value as a DICT containing results list of DICTS
    mock_discover.discover_movies.return_value = {
        "page": 1,
        "results": [MOCK_MOVIE_DICT],
        "total_results": 1
    }
    
    with patch('config.TMDB_API_KEY', 'REAL_KEY'):
        recs, err = utils.fetch_movie_recommendations(["Netflix"], "", "", "", 120, False, "")
        
        assert len(recs) == 2 # 1 item per page * 2 pages
        assert recs[0]['title'] == "Dict Movie"
        assert recs[0]['poster_path'].endswith("/dict.jpg")

def test_onboarding_tmdb_obj_response(mock_deps):
    """Test onboarding retrieval with Object-style TMDB response"""
    mock_discover, _, _ = mock_deps
    mock_discover.discover_movies.return_value = [MOCK_MOVIE_OBJ]
    
    with patch('config.TMDB_API_KEY', 'REAL_KEY'):
        recs = utils.get_onboarding_content()
        
        # Should have movies + videos (if YT also works) or at least movies
        movies = [r for r in recs if r['type'] == 'movie']
        assert len(movies) > 0
        assert movies[0]['title'] == "Obj Movie"

def test_onboarding_tmdb_dict_response(mock_deps):
    """Test onboarding retrieval with Dict-style TMDB response"""
    mock_discover, _, _ = mock_deps
    mock_discover.discover_movies.return_value = {
        "results": [MOCK_MOVIE_DICT]
    }
    
    with patch('config.TMDB_API_KEY', 'REAL_KEY'):
        recs = utils.get_onboarding_content()
        
        movies = [r for r in recs if r['type'] == 'movie']
        assert len(movies) > 0
        assert movies[0]['title'] == "Dict Movie"
