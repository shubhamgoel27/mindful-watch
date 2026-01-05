__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import utils
import config

st.set_page_config(page_title="MindfulWatch Recommender", layout="wide", page_icon="🧘")

# --- Custom CSS for Uniform Thumbnails & Cards ---
st.markdown("""
    <style>
    /* Target onboarding and result images */
    [data-testid="stImage"] {
        display: flex;
        justify-content: center;
    }
    
    [data-testid="stImage"] img {
        height: 300px !important;
        object-fit: cover !important;
        object-position: top !important; /* Keep faces/titles visible */
        border-radius: 10px;
        width: 100% !important;
    }
    
    /* Ensure containers don't collapse */
    .stColumn > div {
        display: flex;
        flex-direction: column;
    }
    
    /* Fixed height title for cards */
    .card-title {
        height: 3em;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        font-weight: 600;
        margin-bottom: 0.5rem;
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
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🧘 MindfulWatch")
        st.write("Your personalized guide to intentional viewing.")
        username = st.text_input("Who is watching?", placeholder="Enter your name")
        if st.button("Enter"):
            if username.strip():
                is_new = login_user(username.strip())
                if is_new:
                    st.session_state.view = 'onboarding'
                else:
                    st.session_state.view = 'dashboard'
                st.rerun()

def show_onboarding():
    st.title(f"Welcome, {st.session_state.user}! Let's get to know you.")
    st.write("Select the content (Movies & Videos) you enjoy to help us calibrate recommendations.")
    
    if 'onboarding_movies' not in st.session_state:
        st.session_state.onboarding_movies = utils.get_onboarding_content()
    
    # Simple Pagination
    if 'page_count' not in st.session_state:
        st.session_state.page_count = 1
    
    items_per_page = 9
    movies = st.session_state.onboarding_movies[:st.session_state.page_count * items_per_page]
    
    if 'selected_onboarding' not in st.session_state:
        st.session_state.selected_onboarding = set()

    # 3-Column Grid
    cols = st.columns(3)
    for idx, movie in enumerate(movies):
        with cols[idx % 3]:
            with st.container(border=True):
                # Standardized key 'poster_path' from utils.py
                img_url = movie.get('poster_path')
                if img_url:
                    st.image(img_url, width="stretch")
                else:
                    st.image("https://via.placeholder.com/300x450?text=No+Image", width="stretch")
                
                # Truncate title for checkbox to ensure relatively uniform height
                title = movie['title']
                display_title = (title[:45] + '..') if len(title) > 45 else title
                
                is_selected = movie['title'] in st.session_state.selected_onboarding
                if st.checkbox(display_title, key=f"fav_{movie['id']}", value=is_selected):
                    st.session_state.selected_onboarding.add(movie['title'])
                elif is_selected:
                     if movie['title'] in st.session_state.selected_onboarding:
                         st.session_state.selected_onboarding.remove(movie['title'])

    col1, col2 = st.columns([1, 4])
    with col1:
        if len(movies) < len(st.session_state.onboarding_movies):
            if st.button("Load More"):
                st.session_state.page_count += 1
                st.rerun()
    
    st.markdown("---")
    st.write(f"**Selected:** {len(st.session_state.selected_onboarding)} items")
    
    if st.button("Finish Setup", type="primary"):
        st.session_state.user_data['liked_movies_onboarding'] = list(st.session_state.selected_onboarding)
        save_current_state()
        st.session_state.view = 'dashboard'
        st.rerun()

def show_dashboard():
    st.sidebar.title(f"👤 {st.session_state.user}")
    
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.session_state.view = 'login'
        st.rerun()
        
    if st.sidebar.button("Retake Onboarding"):
        st.session_state.view = 'onboarding'
        st.session_state.page_count = 1
        st.rerun()
    
    st.sidebar.markdown("---")

    user_prefs = st.session_state.user_data.get('preferences', {})
    default_watched = user_prefs.get('watched_movies', ", ".join(st.session_state.user_data.get('liked_movies_onboarding', [])))

    with st.sidebar.form("preferences_form"):
        subscriptions = st.multiselect("Subscriptions", list(config.PROVIDER_MAP.keys()), default=user_prefs.get('subscriptions', ["Netflix"]))
        watched_movies = st.text_area("Watched / Favorites", value=default_watched)
        liked_actors = st.text_input("Liked Actors", value=user_prefs.get('liked_actors', ""))
        liked_directors = st.text_input("Liked Directors", value=user_prefs.get('liked_directors', ""))
        max_watch_time = st.slider("Max Watch Time (mins)", 10, 180, user_prefs.get('max_watch_time', 120))
        focus_mode = st.checkbox("Focus Mode (Includes Videos)", value=user_prefs.get('focus_mode', True))
        mood_goal = st.text_input("Mood / Goal", value=user_prefs.get('mood_goal', ""), placeholder="relax, learn something new")
        submit_button = st.form_submit_button("Get Recommendations", type="primary")

    if submit_button:
        st.session_state.user_data['preferences'] = {
            "subscriptions": subscriptions, "watched_movies": watched_movies,
            "liked_actors": liked_actors, "liked_directors": liked_directors,
            "max_watch_time": max_watch_time, "focus_mode": focus_mode, "mood_goal": mood_goal
        }
        save_current_state()
        st.session_state.submitted = True
        st.session_state.api_errors = [] # Reset errors
        
        with st.spinner("Analyzing semantic matches & curating your mix..."):
            # Fetch Movies
            movies, m_err = utils.fetch_movie_recommendations(
                subscriptions, watched_movies, liked_actors, liked_directors, max_watch_time, focus_mode, mood_goal
            )
            st.session_state.recommendations['movies'] = movies
            if m_err: st.session_state.api_errors.append(m_err)

            # Fetch Videos
            if focus_mode:
                videos, v_err = utils.fetch_video_recommendations(mood_goal if mood_goal else "interesting")
                st.session_state.recommendations['videos'] = videos
                if v_err: st.session_state.api_errors.append(v_err)
            else:
                st.session_state.recommendations['videos'] = []

    st.title("MindfulWatch Recommender 🎬")
    
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
                        duration = f"{data.get('runtime', 'N/A')} mins" if is_movie else data.get('duration')
                        
                        # Use custom HTML for fixed-height title
                        st.markdown(f"<div class='card-title'>{icon} {title}</div>", unsafe_allow_html=True)
                        st.markdown(f"**Duration:** {duration} | {data['match_reason']}", unsafe_allow_html=True)
                        st.write(data.get('overview') if is_movie else data.get('description'))
                        
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

# --- Main App Controller ---

if 'view' not in st.session_state:
    st.session_state.view = 'login'

if st.session_state.view == 'login':
    show_login()
elif st.session_state.view == 'onboarding':
    show_onboarding()
elif st.session_state.view == 'dashboard':
    show_dashboard()