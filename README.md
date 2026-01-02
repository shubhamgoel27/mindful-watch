# MindfulWatch Recommender 🧘🎬

**MindfulWatch Recommender** is a Streamlit-based web application designed to combat "doomscrolling" by promoting intentional content consumption. Instead of endless algorithmic feeds, it provides curated, high-quality recommendations for Movies and YouTube videos based on your specific mood, time availability, and interests.

## 🚀 Features

-   **Visual Onboarding:** A rich, card-based interface to select your initial content preferences from a mixed pool of popular movies and videos.
-   **Intentional Search:** Filter recommendations by:
    -   **Mood/Goal:** (e.g., "Relax", "Learn Physics", "Feel Inspired")
    -   **Time Constraint:** Slider to set available watch time (10-180 mins).
    -   **Streaming Services:** Filter by subscriptions (Netflix, Prime, Disney+, Hulu).
-   **Focus Mode:** A special mode that prioritizes educational, documentary, and thoughtful content over mindless entertainment.
-   **Mixed Media Feed:** Recommendations include both feature films (via TMDB) and medium-length YouTube videos (via YouTube Data API), presented in a unified visual feed.
-   **Smart Persistence:** Remembers your user profile, preferences, and watch history locally.
-   **Mock Data Fallback:** Fully functional "Demo Mode" that runs out-of-the-box without API keys using a diverse static content pool.

## 🛠️ Tech Stack

-   **Frontend:** [Streamlit](https://streamlit.io/)
-   **Language:** Python 3.11+
-   **Package Manager:** [uv](https://github.com/astral-sh/uv)
-   **APIs:**
    -   [The Movie Database (TMDB) API](https://www.themoviedb.org/documentation/api)
    -   [YouTube Data API v3](https://developers.google.com/youtube/v3)

## 📦 Installation & Setup

### 1. Clone & Install
Ensure you have `uv` installed (or use pip).

```bash
git clone https://github.com/yourusername/mindful-watch.git
cd mindful-watch
uv sync  # Installs dependencies
```

### 2. Configure API Keys (Optional)
To fetch live data, you need API keys. If skipped, the app runs in **Demo Mode**.

1.  Create a `.env` file in the root directory:
    ```bash
    cp .env.example .env
    ```
2.  Add your keys:
    ```ini
    TMDB_API_KEY=your_tmdb_key_here
    YOUTUBE_API_KEY=your_youtube_key_here
    ```

### 3. Run the App
```bash
uv run streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`.

## 🧪 Running Tests
The project includes a test suite for the data fetching logic.
```bash
uv run pytest
```

## 📂 Project Structure

-   `app.py`: Main application logic and UI components.
-   `utils.py`: Data fetching service (API integration + Mock Data generators).
-   `config.py`: Configuration and environment variable management.
-   `test_utils.py`: Unit tests.

## 📝 License
MIT