# Movie Recommendation System

## Overview
This is a **Movie Recommendation System** built using **Streamlit, FAISS, and TMDb API**. It suggests similar movies based on genres and displays movie posters.

## Features
- Provides movie recommendations based on **genre similarity**.
- Uses **TF-IDF** and **FAISS** for fast nearest-neighbor search.
- Fetches **movie posters** from TMDb API.
- Filters out movies without posters.
- Allows users to set **minimum rating** for recommendations.
- **Live Demo:** [Movie Recommendation App](https://movierecommendation-sv.streamlit.app/)

## Model Used
### **TF-IDF + FAISS**
1. **TF-IDF (Term Frequency-Inverse Document Frequency)**
   - Converts movie genres into numerical vectors based on word importance.
   - Measures similarity between movies using text features.

2. **FAISS (Facebook AI Similarity Search)**
   - A highly optimized nearest-neighbor search algorithm.
   - Finds movies with the most similar genre vectors efficiently.

This combination allows for **fast and accurate** movie recommendations based on genre similarity.

## Installation
### Prerequisites
- Python 3.7+
- pip

### Install Dependencies
```sh
pip install streamlit pandas numpy requests scikit-learn faiss-cpu
```

## How to Run
```sh
streamlit run app.py
```

## API Key Setup
Replace `TMDB_API_KEY` in `app.py` with your own TMDb API key:
```python
TMDB_API_KEY = "your_tmdb_api_key"
```

## Usage
1. Enter a movie title in the input box.
2. Adjust the number of recommendations.
3. Set a minimum rating filter.
4. Click **Get Recommendations** to see similar movies with posters.

## Dataset
The recommendation system uses **MovieLens dataset** from GitHub:
- `movies.csv` - Movie details
- `ratings.csv` - User ratings
- `tags.csv` - Movie tags
- `links.csv` - TMDb & IMDb IDs

## Credits
- **MovieLens** dataset for movie details and ratings.
- **TMDb API** for fetching movie posters.
- **FAISS** for efficient similarity search.

## License
This project is for educational purposes. API usage may be subject to TMDb's terms of service.
