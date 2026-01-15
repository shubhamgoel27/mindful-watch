try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import utils
import config
from logging_config import logger

# --- Startup Database Seeding ---
# Seeds the database on first load if it's sparse (< 200 items)
# This ensures cloud deployments always have content
@st.cache_resource(show_spinner="🌱 Seeding database with content...")
def initialize_database():
    """
    Seeds the database on app startup if it's sparse.
    Uses @st.cache_resource to only run once per deployment.
    """
    from seed_config import ESSENTIAL_VIDEO_QUERIES, ESSENTIAL_TMDB_GENRES, MIN_DB_SIZE
    
    # Check current DB size
    collection = utils.get_vector_collection()
    if collection:
        try:
            result = collection.get()
            current_size = len(result.get("ids", []))
            logger.info(f"Database check: {current_size} items")
            
            if current_size >= MIN_DB_SIZE:
                logger.info(f"Database already has {current_size} items, skipping seeding")
                return {"status": "skipped", "size": current_size}
        except Exception as e:
            logger.warning(f"Could not check DB size: {e}")
            current_size = 0
    else:
        current_size = 0
    
    logger.info(f"Database sparse ({current_size} items), starting seeding...")
    
    # Seed movies from TMDB
    movies_added = 0
    if config.TMDB_API_KEY and config.TMDB_API_KEY != "YOUR_TMDB_KEY":
        for genre_id, genre_name in ESSENTIAL_TMDB_GENRES.items():
            try:
                movies = utils.fetch_tmdb_discover(
                    params={"with_genres": genre_id, "vote_count.gte": 100},
                    max_pages=1
                )
                for movie in movies[:10]:  # 10 per genre = ~80 movies
                    movie["type"] = "movie"
                utils.cache_content_to_db(movies[:10])
                movies_added += min(10, len(movies))
                logger.debug(f"Seeded {genre_name}: {min(10, len(movies))} movies")
            except Exception as e:
                logger.warning(f"Failed to seed {genre_name}: {e}")
    
    # Seed videos using yt-dlp (no quota limits!)
    videos_added = 0
    if utils.YT_DLP_AVAILABLE:
        for query in ESSENTIAL_VIDEO_QUERIES[:30]:  # First 30 queries for speed
            try:
                videos = utils.search_youtube_ytdlp(query, max_results=5)
                if videos:
                    utils.cache_content_to_db(videos)
                    videos_added += len(videos)
                    logger.debug(f"Seeded '{query}': {len(videos)} videos")
            except Exception as e:
                logger.warning(f"Failed to seed '{query}': {e}")
    
    total_added = movies_added + videos_added
    logger.info(f"Database seeding complete: +{movies_added} movies, +{videos_added} videos")
    
    return {"status": "seeded", "movies": movies_added, "videos": videos_added}

# Initialize database on startup
_db_init_result = initialize_database()

st.set_page_config(page_title="Unrot Me", layout="wide", page_icon="🧠")

# --- Modern Dark Theme CSS ---
# Netflix/Spotify-inspired styling with glassmorphism and gradients
st.markdown("""
    <style>
    /* ========================================
       IMPORT MODERN FONT
    ======================================== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* ========================================
       GLOBAL DARK THEME BASE
    ======================================== */
    .stApp {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%) !important;
    }
    
    /* Override Streamlit's default white backgrounds */
    .stApp > header {
        background: transparent !important;
    }
    
    .main .block-container {
        padding-top: 2rem;
        max-width: 1400px;
    }
    
    /* All text should use Inter font */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* ========================================
       CENTER IMAGES IN CONTAINERS
    ======================================== */
    [data-testid="stImage"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
    }
    
    [data-testid="stImage"] > img {
        max-height: 350px !important;
        width: auto !important;
        object-fit: contain !important;
    }
    
    /* ========================================
       SIDEBAR STYLING
    ======================================== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #f0f6fc;
    }
    
    /* Sidebar form styling */
    [data-testid="stSidebar"] [data-testid="stForm"] {
        background: rgba(22, 27, 34, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
    }
    
    /* ========================================
       TYPOGRAPHY
    ======================================== */
    h1 {
        background: linear-gradient(135deg, #7c3aed 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    
    h2, h3 {
        color: #f0f6fc !important;
        font-weight: 600 !important;
    }
    
    p, span, label {
        color: #c9d1d9 !important;
    }
    
    /* ========================================
       INPUTS & FORM ELEMENTS
    ======================================== */
    /* Text inputs */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(22, 27, 34, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        color: #f0f6fc !important;
        backdrop-filter: blur(10px);
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #7c3aed !important;
        box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.2) !important;
    }
    
    /* Slider */
    .stSlider > div > div > div > div {
        background: linear-gradient(135deg, #7c3aed 0%, #3b82f6 100%) !important;
    }
    
    /* Selectbox & Multiselect */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: rgba(22, 27, 34, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
    }
    
    /* Checkbox */
    .stCheckbox > label > span {
        color: #c9d1d9 !important;
    }
    
    /* ========================================
       BUTTONS
    ======================================== */
    /* Primary button - gradient style */
    .stButton > button[kind="primary"],
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #7c3aed 0%, #3b82f6 100%) !important;
        border: none !important;
        border-radius: 10px !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.5rem !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
    }
    
    .stButton > button[kind="primary"]:hover,
    .stFormSubmitButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.4) !important;
    }
    
    /* Secondary buttons */
    .stButton > button:not([kind="primary"]) {
        background: rgba(22, 27, 34, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
        color: #f0f6fc !important;
        transition: all 0.2s !important;
    }
    
    .stButton > button:not([kind="primary"]):hover {
        background: rgba(33, 38, 45, 0.9) !important;
        border-color: #7c3aed !important;
    }
    
    /* ========================================
       CARDS & CONTAINERS
    ======================================== */
    /* Content cards with glassmorphism */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
        background: rgba(22, 27, 34, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    /* Bordered containers (cards) */
    [data-testid="stVerticalBlock"]:has(> [data-testid="stImage"]) {
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    [data-testid="stVerticalBlock"]:has(> [data-testid="stImage"]):hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(124, 58, 237, 0.15);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(22, 27, 34, 0.6) !important;
        border-radius: 10px !important;
        color: #c9d1d9 !important;
    }
    
    /* ========================================
       IMAGES & THUMBNAILS
    ======================================== */
    [data-testid="stImage"] {
        display: flex;
        justify-content: center;
    }
    
    [data-testid="stImage"] img {
        height: 280px !important;
        object-fit: cover !important;
        object-position: top !important;
        border-radius: 12px !important;
        width: 100% !important;
        transition: transform 0.3s ease;
    }
    
    [data-testid="stImage"]:hover img {
        transform: scale(1.02);
    }
    
    /* ========================================
       CUSTOM CARD TITLE
    ======================================== */
    .card-title {
        height: 2.8em;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        font-weight: 600;
        font-size: 1rem;
        color: #f0f6fc !important;
        margin-bottom: 0.5rem;
    }
    
    /* Match percentage badge */
    .match-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.3) 0%, rgba(59, 130, 246, 0.3) 100%);
        border: 1px solid rgba(124, 58, 237, 0.5);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.85rem;
        font-weight: 500;
        color: #c4b5fd !important;
    }
    
    /* ========================================
       DIVIDERS & MISC
    ======================================== */
    hr {
        border-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Info/Warning boxes */
    .stAlert {
        background: rgba(22, 27, 34, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #7c3aed !important;
    }
    
    /* ========================================
       SCROLLBAR STYLING
    ======================================== */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0d1117;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #30363d;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #484f58;
    }
    
    /* ========================================
       COLUMN LAYOUT FIXES
    ======================================== */
    .stColumn > div {
        display: flex;
        flex-direction: column;
    }
    
    /* Radio buttons as pills */
    .stRadio > div {
        flex-direction: row !important;
        gap: 0.5rem;
    }
    
    .stRadio > div > label {
        background: rgba(22, 27, 34, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 20px !important;
        padding: 0.4rem 1rem !important;
        transition: all 0.2s !important;
    }
    
    .stRadio > div > label:hover {
        border-color: #7c3aed !important;
    }
    
    .stRadio > div > label[data-checked="true"] {
        background: linear-gradient(135deg, #7c3aed 0%, #3b82f6 100%) !important;
        border-color: transparent !important;
    }
    
    /* ========================================
       HERO SECTION STYLING
    ======================================== */
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #7c3aed 0%, #3b82f6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        color: #8b949e;
        font-size: 1.1rem;
        font-weight: 400;
    }
    
    /* ========================================
       ANIMATIONS
    ======================================== */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stMarkdown, .stImage {
        animation: fadeIn 0.4s ease-out;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Session State Initialization ---
if 'user' not in st.session_state:
    st.session_state.user = None
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = {'movies': [], 'videos': []}
if 'api_errors' not in st.session_state:
    st.session_state.api_errors = []
if 'submitted' not in st.session_state:
    st.session_state.submitted = False

# --- Helper Functions ---
def login_user(username):
    all_data = utils.load_user_data()
    st.session_state.user = username
    if username in all_data:
        st.session_state.user_data = all_data[username]
        return False # Not a new user
    else:
        st.session_state.user_data = {
            "history": [],
            "preferences": {},
            "liked_movies_onboarding": []
        }
        return True # New user

def save_current_state():
    if st.session_state.user:
        all_data = utils.load_user_data()
        all_data[st.session_state.user] = st.session_state.user_data
        utils.save_user_data(all_data)

# --- Views ---

def show_login():
    # Centered login with hero styling
    st.markdown("<div style='height: 10vh'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h1 class='hero-title'>🧠 Unrot Me</h1>
                <p class='hero-subtitle'>Doomscrolling lost. You won.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Google OAuth Login (if configured)
        if config.GOOGLE_AUTH_ENABLED:
            try:
                from streamlit_google_auth import Authenticate
                import json
                import tempfile
                import os
                
                # Create temporary JSON credentials file (required by library)
                creds_data = {
                    "web": {
                        "client_id": config.GOOGLE_CLIENT_ID,
                        "client_secret": config.GOOGLE_CLIENT_SECRET,
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "redirect_uris": ["http://localhost:8501"]
                    }
                }
                
                # Write to temp file (cached in session to avoid recreating)
                if 'google_creds_path' not in st.session_state:
                    fd, path = tempfile.mkstemp(suffix='.json')
                    with os.fdopen(fd, 'w') as f:
                        json.dump(creds_data, f)
                    st.session_state.google_creds_path = path
                
                # Create authenticator
                authenticator = Authenticate(
                    secret_credentials_path=st.session_state.google_creds_path,
                    cookie_name='unrot_me_auth',
                    cookie_key='unrot_me_secret_key_12345',
                    redirect_uri="http://localhost:8501",
                )
                
                # Check if already authenticated
                authenticator.check_authentification()
                
                if st.session_state.get("connected"):
                    user_info = st.session_state.get("user_info", {})
                    user_email = user_info.get("email", "")
                    user_name = user_info.get("name", user_email.split("@")[0])
                    
                    # Use name for display and login (not email)
                    display_name = user_name if user_name else user_email.split("@")[0]
                    
                    st.success(f"Signed in as {display_name}")
                    
                    # Check for admin
                    if user_email == config.ADMIN_USERNAME or user_name == config.ADMIN_USERNAME:
                        st.session_state.user = display_name
                        st.session_state.view = 'admin'
                        st.rerun()
                    else:
                        # Regular user login - use name as user ID
                        is_new = login_user(display_name)
                        if is_new:
                            st.session_state.view = 'onboarding'
                        else:
                            st.session_state.view = 'dashboard'
                        st.rerun()
                else:
                    # Show official Google Sign-In button (centered, standard width)
                    authorization_url = authenticator.get_authorization_url()
                    st.markdown(f"""
                        <div style="display: flex; justify-content: center;">
                            <a href="{authorization_url}" target="_self" style="text-decoration: none;">
                                <div style="
                                    display: flex;
                                    align-items: center;
                                    justify-content: center;
                                    background: white;
                                    border: 1px solid #dadce0;
                                    border-radius: 4px;
                                    padding: 10px 24px;
                                    cursor: pointer;
                                    transition: background 0.2s;
                                    width: 220px;
                                ">
                                    <img src="https://www.gstatic.com/firebasejs/ui/2.0.0/images/auth/google.svg" 
                                         style="width: 18px; height: 18px; margin-right: 12px;" />
                                    <span style="color: #1a1a1a; font-size: 14px; font-weight: 500; font-family: 'Roboto', sans-serif;">
                                        Sign in with Google
                                    </span>
                                </div>
                            </a>
                        </div>
                    """, unsafe_allow_html=True)
                    
            except ImportError:
                st.warning("Google Auth package not installed. Using username login.")
                config.GOOGLE_AUTH_ENABLED = False
            except Exception as e:
                logger.error(f"Google Auth error: {e}")
                st.warning(f"Google login unavailable: {e}")
        
        # Divider if both login methods available
        if config.GOOGLE_AUTH_ENABLED:
            st.markdown("<p style='text-align: center; color: #8b949e; margin: 1rem 0;'>— or —</p>", unsafe_allow_html=True)
        
        # Username login (fallback or primary if Google not configured)
        username = st.text_input("Name", placeholder="Enter your name...", label_visibility="collapsed")
        
        # Show password field only if admin username is entered
        admin_password_input = None
        if username.strip() == config.ADMIN_USERNAME:
            admin_password_input = st.text_input(
                "Admin Password", 
                type="password", 
                placeholder="Enter admin password...",
                label_visibility="collapsed"
            )
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            if st.button("✨ Get Started", type="primary", width="stretch"):
                if username.strip():
                    # Check for power user
                    if username.strip() == config.ADMIN_USERNAME:
                        # Verify password
                        if not config.ADMIN_PASSWORD:
                            st.error("Admin access not configured on this deployment.")
                        elif admin_password_input == config.ADMIN_PASSWORD:
                            st.session_state.user = username.strip()
                            st.session_state.view = 'admin'
                            st.rerun()
                        else:
                            st.error("Invalid admin password.")
                    else:
                        is_new = login_user(username.strip())
                        if is_new:
                            st.session_state.view = 'onboarding'
                        else:
                            st.session_state.view = 'dashboard'
                        st.rerun()

def show_onboarding():
    # Initialize onboarding state
    if 'onboarding_movies' not in st.session_state:
        st.session_state.onboarding_movies = utils.get_onboarding_content()
    if 'onboarding_index' not in st.session_state:
        st.session_state.onboarding_index = 0
    if 'liked_onboarding' not in st.session_state:
        st.session_state.liked_onboarding = []
    if 'disliked_onboarding' not in st.session_state:
        st.session_state.disliked_onboarding = []
    
    content_list = st.session_state.onboarding_movies
    current_idx = st.session_state.onboarding_index
    total_rated = len(st.session_state.liked_onboarding) + len(st.session_state.disliked_onboarding)
    min_ratings = 5
    
    # Compact header
    st.markdown(f"""
        <div style='text-align: center; margin-bottom: 0.5rem;'>
            <h2 style='margin: 0;'>Welcome {st.session_state.user}! 👑</h2>
            <p style='color: #8b949e; margin: 0.2rem 0;'>Swipe your taste. Trash banned.</p>
            <p style='color: #7c3aed; margin: 0.3rem 0; font-size: 0.9rem;'>Rated: {total_rated} | Need {min_ratings}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Show current card
    if current_idx < len(content_list):
        current_item = content_list[current_idx]
        
        # Center the card
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Image centered using HTML (works reliably)
            img_url = current_item.get('poster_path') or current_item.get('thumbnail')
            if img_url:
                st.markdown(f"""
                    <div style='display: flex; justify-content: center; margin-bottom: 0.5rem;'>
                        <img src='{img_url}' style='max-height: 280px; max-width: 100%; border-radius: 12px; object-fit: contain;'/>
                    </div>
                """, unsafe_allow_html=True)
            
            # Title only
            title = current_item.get('title', 'Unknown')
            st.markdown(f"<h3 style='text-align: center; margin: 0.5rem 0;'>{title}</h3>", unsafe_allow_html=True)
            
            # Like/Dislike buttons (Tinder style)
            btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
            with btn_col1:
                if st.button("👎 Nope", key="dislike_btn", width="stretch"):
                    st.session_state.disliked_onboarding.append(current_item['title'])
                    st.session_state.onboarding_index += 1
                    st.rerun()
            with btn_col2:
                if st.button("⏭️ Skip", key="skip_btn", width="stretch"):
                    st.session_state.onboarding_index += 1
                    st.rerun()
            with btn_col3:
                if st.button("👍 Like", key="like_btn", type="primary", width="stretch"):
                    st.session_state.liked_onboarding.append(current_item['title'])
                    st.session_state.onboarding_index += 1
                    st.rerun()
    else:
        # No more content to rate
        st.info("You've rated all available content!")
    
    # Finish button (only enabled if min ratings met)
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if total_rated >= min_ratings:
            if st.button("✨ Finish Setup & Get Recommendations", type="primary", width="stretch"):
                st.session_state.user_data['liked_movies_onboarding'] = st.session_state.liked_onboarding
                save_current_state()
                st.session_state.view = 'dashboard'
                st.rerun()
        else:
            remaining = min_ratings - total_rated
            st.markdown(f"<p style='text-align: center; color: #8b949e;'>Rate {remaining} more to continue</p>", unsafe_allow_html=True)

def show_dashboard():
    st.sidebar.title(f"👤 {st.session_state.user}")
    
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.session_state.view = 'login'
        st.rerun()
        
    if st.sidebar.button("Retake Onboarding"):
        st.session_state.view = 'onboarding'
        # Reset onboarding state
        st.session_state.onboarding_index = 0
        st.session_state.liked_onboarding = []
        st.session_state.disliked_onboarding = []
        st.rerun()
    
    st.sidebar.markdown("---")

    user_prefs = st.session_state.user_data.get('preferences', {})

    with st.sidebar.form("preferences_form"):
        st.markdown("### 🎯 What are you in the mood for?")
        mood_goal = st.text_input(
            "Mood / Goal", 
            value=user_prefs.get('mood_goal', ""), 
            placeholder="e.g. relax, learn science, feel inspired",
            label_visibility="collapsed"
        )
        
        st.markdown("### ⏱️ How much time do you have?")
        # Slider with 5-minute intervals
        # Snap saved value to nearest valid option (handles old user data)
        valid_options = list(range(10, 185, 5))  # 10-180 in steps of 5
        saved_time = user_prefs.get('max_watch_time', 60)
        # Round to nearest 5 and clamp to valid range
        snapped_time = max(10, min(180, round(saved_time / 5) * 5))
        
        max_watch_time = st.select_slider(
            "Max Watch Time",
            options=valid_options,
            value=snapped_time,
            format_func=lambda x: f"{x} mins",
            label_visibility="collapsed"
        )
        
        submit_button = st.form_submit_button("🔍 Get Recommendations", type="primary", width="stretch")

    # Clear search button (outside form)
    if st.sidebar.button("🗑️ Clear Search", width="stretch"):
        st.session_state.recommendations = {'movies': [], 'videos': []}
        st.session_state.user_data['preferences']['mood_goal'] = ""
        st.session_state.submitted = False
        save_current_state()
        st.rerun()

    if submit_button:
        # Clear old results first to ensure fresh state
        st.session_state.recommendations = {'movies': [], 'videos': []}
        
        st.session_state.user_data['preferences'] = {
            "max_watch_time": max_watch_time, 
            "mood_goal": mood_goal
        }
        save_current_state()
        st.session_state.submitted = True
        st.session_state.api_errors = []
        
        with st.spinner("Finding your perfect content..."):
            # Build liked_content tuple from session state for query enrichment
            liked_titles = st.session_state.user_data.get('liked_movies_onboarding', [])
            onboarding_content = st.session_state.get('onboarding_movies', [])
            liked_content = (liked_titles, onboarding_content) if liked_titles else None
            
            logger.info(f"Dashboard search: liked_titles={len(liked_titles) if liked_titles else 0}, onboarding_content={len(onboarding_content) if onboarding_content else 0}")
            logger.info(f"Dashboard search: liked_content is {'set' if liked_content else 'None'}")
            logger.info(f"Dashboard search: mood_goal='{mood_goal}'")
            
            # Build search query: use mood if provided
            search_query = mood_goal if mood_goal else ""
            
            # Fetch videos
            st.session_state.recommendations['movies'] = []
            videos, v_err = utils.fetch_video_recommendations(
                search_query,
                max_time_mins=max_watch_time,
                liked_content=liked_content
            )
            st.session_state.recommendations['videos'] = videos
            if v_err: st.session_state.api_errors.append(v_err)

    st.markdown("""
        <h1 style='margin-bottom: 0.5rem;'>🧠 Unrot Me</h1>
        <p style='color: #8b949e; margin-bottom: 1.5rem;'>Personalized content curated just for you</p>
    """, unsafe_allow_html=True)
    
    # Display API Errors/Warnings if any
    if st.session_state.api_errors:
        for err in st.session_state.api_errors:
            if "Demo Mode" in err:
                st.warning(f"⚠️ {err}")
            else:
                st.error(f"❌ {err}")

    if st.session_state.submitted:
        movies = st.session_state.recommendations['movies']
        videos = st.session_state.recommendations['videos']
        
        # Mix Content
        mixed_content = []
        import itertools
        for m, v in itertools.zip_longest(movies, videos):
            if m: mixed_content.append({'type': 'movie', 'data': m})
            if v: mixed_content.append({'type': 'video', 'data': v})

        # --- Output Filters ---
        if mixed_content:
            st.markdown("### Refine Results")
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                filter_type = st.radio("Show Type", ["All", "Movies", "Videos"], horizontal=True)
            with f_col2:
                sort_order = st.selectbox("Sort By", ["Default", "Shortest First", "Longest First"])
            
            # Apply Filters
            filtered_content = mixed_content
            if filter_type == "Movies":
                filtered_content = [x for x in filtered_content if x['type'] == 'movie']
            elif filter_type == "Videos":
                filtered_content = [x for x in filtered_content if x['type'] == 'video']
            
            # Apply Sorting
            if sort_order != "Default":
                def get_duration(item):
                    if item['type'] == 'movie': return item['data'].get('runtime', 999)
                    # Parse "25 mins" -> int
                    d_str = item['data'].get('duration', '0 mins')
                    return int(d_str.split()[0]) if d_str.split()[0].isdigit() else 0
                
                filtered_content.sort(key=get_duration, reverse=(sort_order == "Longest First"))

            st.write(f"Showing {len(filtered_content)} recommendations:")
            st.divider()

            for item in filtered_content:
                data = item['data']
                is_movie = item['type'] == 'movie'
                
                with st.container(border=True):
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        img_url = data.get('poster_path') if is_movie else data.get('thumbnail')
                        if img_url: st.image(img_url, width="stretch")
                        else: st.image("https://via.placeholder.com/150?text=No+Img", width="stretch")
                    
                    with col2:
                        icon = "🎥" if is_movie else "▶️"
                        title = data['title']
                        duration = f"{data.get('runtime', 'N/A')} mins" if is_movie else data.get('duration', '')
                        match_reason = data.get('match_reason', '')
                        
                        # Modern styled card title
                        st.markdown(f"<div class='card-title'>{icon} {title}</div>", unsafe_allow_html=True)
                        
                        # Match badge and duration
                        if match_reason:
                            st.markdown(f"<span class='match-badge'>{match_reason}</span> &nbsp; ⏱️ {duration}", unsafe_allow_html=True)
                        else:
                            st.markdown(f"⏱️ {duration}", unsafe_allow_html=True)
                        
                        # Description (truncated)
                        desc = data.get('overview') if is_movie else data.get('description', '')
                        if desc:
                            truncated = desc[:200] + "..." if len(desc) > 200 else desc
                            st.markdown(f"<p style='color: #8b949e; font-size: 0.9rem; margin-top: 0.5rem;'>{truncated}</p>", unsafe_allow_html=True)
                        
                        if is_movie:
                            # Robust Link Logic
                            mid = data.get('id')
                            title_enc = data['title'].replace(' ', '+')
                            
                            if mid and not (isinstance(mid, str) and mid.startswith('mock')):
                                link = f"https://www.themoviedb.org/movie/{mid}"
                            else:
                                link = f"https://www.themoviedb.org/search?query={title_enc}"
                            
                            st.markdown(f"[View on TMDB]({link})")
                        else:
                            vid_id = data.get('video_id')
                            title_enc = data['title'].replace(' ', '+')
                            
                            if vid_id and not (isinstance(vid_id, str) and vid_id.startswith('mock')):
                                link = f"https://www.youtube.com/watch?v={vid_id}"
                            else:
                                link = f"https://www.youtube.com/results?search_query={title_enc}"
                            st.markdown(f"[Watch on YouTube]({link})")
                st.write("") 
        else:
            st.info("No recommendations found.")
    else:
        st.info("👈 Set your mood and preferences to start.")

# --- Admin Panel (Power User Only) ---

def show_admin():
    """Admin panel for power_user_27 with database management tools."""
    
    st.sidebar.markdown("### 🔧 Admin Panel")
    st.sidebar.write(f"👤 **{st.session_state.user}**")
    
    # System Status (admin only)
    with st.sidebar.expander("🛠️ System Status"):
        tmdb_ok = config.TMDB_API_KEY and config.TMDB_API_KEY != "YOUR_TMDB_KEY"
        yt_ok = config.YOUTUBE_API_KEY and config.YOUTUBE_API_KEY != "YOUR_YOUTUBE_KEY"
        st.write(f"**TMDB API:** {'✅' if tmdb_ok else '❌'}")
        st.write(f"**YouTube API:** {'✅' if yt_ok else '❌'}")
        st.write(f"**yt-dlp:** {'✅' if utils.YT_DLP_AVAILABLE else '❌'}")
    
    if st.sidebar.button("🚪 Logout"):
        st.session_state.user = None
        st.session_state.view = 'login'
        st.rerun()
    
    st.markdown("""
        <h1 style='margin-bottom: 0.5rem;'>🔧 Admin Dashboard</h1>
        <p style='color: #8b949e;'>Database management and content tools</p>
    """, unsafe_allow_html=True)
    
    # Database Stats Section
    st.markdown("### 📊 Database Stats")
    col1, col2, col3 = st.columns(3)
    
    collection = utils.get_vector_collection()
    if collection:
        try:
            result = collection.get()
            total = len(result.get("ids", []))
            movies = sum(1 for m in result.get("metadatas", []) if m.get("type") == "movie")
            videos = sum(1 for m in result.get("metadatas", []) if m.get("type") == "video")
            
            col1.metric("Total Items", total)
            col2.metric("Movies", movies)
            col3.metric("Videos", videos)
        except:
            st.error("Could not fetch database stats")
    
    st.divider()
    
    # Admin Actions in Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🗑️ Clear Data", "🌱 Seed Content", "📹 Fetch Videos", "🎬 Fetch Movies", "👀 View Content"])
    
    # Tab 1: Clear Data
    with tab1:
        st.markdown("#### Clear Database")
        st.warning("⚠️ This will permanently delete all content from the database!")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear All Data", type="primary"):
                with st.spinner("Clearing database..."):
                    try:
                        result = collection.get()
                        ids = result.get("ids", [])
                        if ids:
                            collection.delete(ids=ids)
                            st.success(f"✅ Cleared {len(ids)} items!")
                            st.rerun()
                        else:
                            st.info("Database is already empty")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        with col2:
            if st.button("🧹 Clear User Data"):
                import os
                if os.path.exists("user_data.json"):
                    os.remove("user_data.json")
                    st.success("✅ User data cleared!")
                else:
                    st.info("No user data file found")
    
    # Tab 2: Seed Content
    with tab2:
        st.markdown("#### Seed Database with Content")
        
        st.markdown("##### Quick Seed (Fast)")
        col1, col2 = st.columns(2)
        with col1:
            num_queries = st.slider("Video queries", 5, 50, 20)
        with col2:
            results_per = st.slider("Results per query", 5, 20, 10)
        
        if st.button("🌱 Quick Seed Videos", type="primary"):
            from seed_config import ESSENTIAL_VIDEO_QUERIES
            with st.spinner(f"Seeding {num_queries} queries × {results_per} results..."):
                total = 0
                progress = st.progress(0)
                for i, query in enumerate(ESSENTIAL_VIDEO_QUERIES[:num_queries]):
                    try:
                        videos = utils.search_youtube_ytdlp(query, max_results=results_per)
                        if videos:
                            utils.cache_content_to_db(videos)
                            total += len(videos)
                    except:
                        pass
                    progress.progress((i + 1) / num_queries)
                st.success(f"✅ Added {total} videos!")
                st.rerun()
        
        st.divider()
        st.markdown("##### Full Seed (All Content)")
        st.info("⏱️ This will fetch ~500 movies + ~10K videos. Takes 1-2 hours.")
        
        if st.button("🚀 Run Full Seed", type="secondary"):
            with st.spinner("Starting full database seeding..."):
                try:
                    from seed_database import seed_movies, seed_bollywood, seed_videos, YOUTUBE_QUERIES, EXTRA_VIDEO_QUERIES
                    
                    # Seed movies
                    st.write("📽️ Seeding movies...")
                    movies = seed_movies(max_per_genre=25)
                    st.write(f"  → {movies} Western movies")
                    
                    # Seed Bollywood
                    st.write("🎬 Seeding Indian movies...")
                    bollywood = seed_bollywood()
                    st.write(f"  → {bollywood} Indian movies")
                    
                    # Seed videos
                    st.write("📹 Seeding videos (this takes a while)...")
                    all_queries = YOUTUBE_QUERIES + EXTRA_VIDEO_QUERIES
                    videos = seed_videos(queries=all_queries[:100], results_per_query=15)
                    st.write(f"  → {videos} videos")
                    
                    st.success(f"✅ Full seed complete! Added {movies + bollywood} movies, {videos} videos")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # Tab 3: Fetch Videos
    with tab3:
        st.markdown("#### Fetch Videos by Query")
        
        query = st.text_input("Search query", placeholder="e.g. quantum physics documentary", key="yt_query")
        col1, col2 = st.columns(2)
        with col1:
            max_results = st.slider("Max results", 5, 50, 20, key="yt_max")
        with col2:
            save_to_db = st.checkbox("Save to database", value=True, key="yt_save")
        
        if st.button("🔍 Search YouTube", type="primary"):
            if query:
                with st.spinner(f"Searching for '{query}'..."):
                    videos = utils.search_youtube_ytdlp(query, max_results=max_results)
                    
                    if videos:
                        st.success(f"Found {len(videos)} videos!")
                        
                        if save_to_db:
                            utils.cache_content_to_db(videos)
                            st.info(f"✅ Saved {len(videos)} videos to database")
                        
                        # Display results
                        for v in videos[:10]:
                            with st.expander(f"📹 {v.get('title', 'Unknown')[:60]}..."):
                                col1, col2 = st.columns([1, 3])
                                with col1:
                                    if v.get('thumbnail'):
                                        st.image(v['thumbnail'], width=150)
                                with col2:
                                    st.write(f"**Duration:** {v.get('duration', 'N/A')}")
                                    st.write(f"**Channel:** {v.get('channel', 'N/A')}")
                                    desc = (v.get('description') or '')[:200]
                                    if desc:
                                        st.write(f"_{desc}..._")
                    else:
                        st.warning("No videos found")
            else:
                st.warning("Enter a search query")
    
    # Tab 4: Fetch Movies (TMDB)
    with tab4:
        st.markdown("#### Fetch Movies from TMDB")
        
        tmdb_ok = config.TMDB_API_KEY and config.TMDB_API_KEY != "YOUR_TMDB_KEY"
        if not tmdb_ok:
            st.error("TMDB API key not configured!")
        else:
            # Search by title
            st.markdown("##### Search by Title")
            movie_query = st.text_input("Movie title", placeholder="e.g. Inception", key="tmdb_query")
            
            if st.button("🎬 Search TMDB", type="primary"):
                if movie_query:
                    with st.spinner(f"Searching for '{movie_query}'..."):
                        import requests
                        url = f"https://api.themoviedb.org/3/search/movie"
                        params = {
                            "api_key": config.TMDB_API_KEY,
                            "query": movie_query,
                            "language": "en-US",
                            "page": 1
                        }
                        resp = requests.get(url, params=params)
                        if resp.ok:
                            results = resp.json().get("results", [])
                            if results:
                                st.success(f"Found {len(results)} movies!")
                                
                                # Process and save
                                movies = []
                                for m in results[:20]:
                                    m["type"] = "movie"
                                    m["poster_path"] = f"https://image.tmdb.org/t/p/w500{m.get('poster_path')}" if m.get('poster_path') else ""
                                    movies.append(m)
                                
                                utils.cache_content_to_db(movies)
                                st.info(f"✅ Saved {len(movies)} movies to database")
                                
                                # Display
                                for m in movies[:5]:
                                    with st.expander(f"🎬 {m.get('title', 'Unknown')}"):
                                        col1, col2 = st.columns([1, 3])
                                        with col1:
                                            if m.get('poster_path'):
                                                st.image(m['poster_path'], width=100)
                                        with col2:
                                            st.write(f"**Year:** {m.get('release_date', 'N/A')[:4] if m.get('release_date') else 'N/A'}")
                                            st.write(f"**Rating:** {m.get('vote_average', 'N/A')}")
                                            st.write(m.get('overview', '')[:200])
                            else:
                                st.warning("No movies found")
                        else:
                            st.error(f"TMDB Error: {resp.status_code}")
            
            st.divider()
            st.markdown("##### Discover by Genre/Year")
            
            genres = {
                28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy",
                80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family",
                14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music",
                9648: "Mystery", 10749: "Romance", 878: "Sci-Fi", 53: "Thriller"
            }
            
            col1, col2, col3 = st.columns(3)
            with col1:
                selected_genre = st.selectbox("Genre", list(genres.values()), key="tmdb_genre")
                genre_id = [k for k, v in genres.items() if v == selected_genre][0]
            with col2:
                year = st.number_input("Year (optional)", min_value=1980, max_value=2025, value=2023, key="tmdb_year")
            with col3:
                language = st.selectbox("Language", ["en", "hi", "ta", "te", "ko", "ja"], key="tmdb_lang")
            
            if st.button("🔎 Discover Movies"):
                with st.spinner("Discovering movies..."):
                    movies = utils.fetch_tmdb_discover(
                        params={
                            "with_genres": genre_id,
                            "primary_release_year": year,
                            "with_original_language": language,
                            "vote_count.gte": 50,
                            "sort_by": "popularity.desc"
                        },
                        max_pages=2
                    )
                    
                    if movies:
                        for m in movies:
                            m["type"] = "movie"
                        utils.cache_content_to_db(movies)
                        st.success(f"✅ Added {len(movies)} {selected_genre} movies from {year}!")
                        st.rerun()
                    else:
                        st.warning("No movies found with these filters")
    
    # Tab 5: View Content
    with tab5:
        st.markdown("#### Sample Database Content")
        
        content_type = st.selectbox("Filter by type", ["All", "Movies", "Videos"])
        sample_size = st.slider("Sample size", 5, 50, 20, key="sample_size")
        
        if st.button("👀 Load Sample"):
            try:
                result = collection.get(limit=sample_size * 2)
                items = []
                for i, meta in enumerate(result.get("metadatas", [])):
                    item_type = meta.get("type", "unknown")
                    if content_type == "All" or (content_type == "Movies" and item_type == "movie") or (content_type == "Videos" and item_type == "video"):
                        items.append({
                            "title": meta.get("title", "Unknown"),
                            "type": item_type,
                            "id": result["ids"][i][:20] + "..."
                        })
                    if len(items) >= sample_size:
                        break
                
                if items:
                    import pandas as pd
                    st.dataframe(pd.DataFrame(items), use_container_width=True)
                else:
                    st.info("No content found")
            except Exception as e:
                st.error(f"Error: {e}")

# --- Main App Controller ---

if 'view' not in st.session_state:
    st.session_state.view = 'login'

if st.session_state.view == 'login':
    show_login()
elif st.session_state.view == 'onboarding':
    show_onboarding()
elif st.session_state.view == 'dashboard':
    show_dashboard()
elif st.session_state.view == 'admin':
    show_admin()