# 🛍️ Product Recommendation System

## 📌 Overview
A Machine Learning-based web application that recommends similar products using product names, categories, and descriptions.

## 🚀 Features
- Search products
- Filter by category
- Recommend similar products
- Display prices and similarity scores
- Interactive Streamlit UI

## 🛠️ Technologies
- Python
- Streamlit
- Pandas
- Scikit-learn
- TF-IDF and Cosine Similarity

## 📂 Project Files
- `app.py` — Main application
- `products.csv` — Product dataset
- `requirements.txt` — Required libraries

## ⚙️ Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## 🧠 Algorithm
The system uses **Content-Based Filtering** with TF-IDF and Cosine Similarity to recommend similar products.

## 👨‍💻 Author
Your Name
