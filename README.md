# 🎬 Movie Recommendation System

This is a simple **Movie Recommendation System** built using **Streamlit**, **Pandas**, and **Scikit-learn**. It recommends movies based on genre similarity and average user ratings.

## 🚀 Features
- **TF-IDF Based Genre Similarity**: Uses the **TfidfVectorizer** to analyze genres.
- **Cosine Similarity**: Computes similarity between movies.
- **Average Ratings**: Filters recommendations based on user ratings.
- **Interactive UI**: Uses **Streamlit** for a user-friendly interface.
- **IMDB Links**: Provides direct links to IMDB for movie details.
- **Dynamic Rating Filtering**: Allows users to filter recommendations based on minimum rating.
- **Weighted Scoring**: Uses a weighted scoring system to rank recommendations.

## 🤖 Machine Learning Algorithm Used
This project uses **TF-IDF (Term Frequency-Inverse Document Frequency) Vectorization** combined with **Cosine Similarity** to measure the similarity between movies based on their genres. Additionally, a weighted scoring system is applied to prioritize recommendations based on rating and similarity ranking.

## 📦 Requirements
Make sure you have the following dependencies installed:
```bash
pip install pandas numpy scikit-learn streamlit
```

## 📂 Dataset Requirements
Your project should include the following CSV files:
- `movies.csv` (Must include `movieId`, `title`, and `genres` columns)
- `ratings.csv` (Must include `movieId` and `rating` columns)
- `tags.csv` (Optional, used for metadata)
- `links.csv` (Must include `movieId` and `imdbId` columns)

## 🏃‍♂️ How to Run
Run the Streamlit app with the following command:
```bash
streamlit run app.py
```

## 🎥 How It Works
1. Enter a movie title.
2. Select the number of recommendations.
3. Set the minimum rating filter.
4. Click **Get Recommendations**.

## 🔥 Example Output
```
Movies similar to 'Inception':
- Interstellar (⭐ 4.5) - [IMDB Link](https://www.imdb.com/title/tt0816692/)
- The Prestige (⭐ 4.3) - [IMDB Link](https://www.imdb.com/title/tt0482571/)
```

## 📜 License
This project is open-source and available under the **MIT License**.

---
Feel free to contribute and improve this project! 😊

