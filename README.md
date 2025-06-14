# 📰 Fake News Detection Web App

This project is an end-to-end Fake News Detection system using Natural Language Processing (NLP) and Machine Learning. A web interface is provided using **Streamlit**, where users can enter news content (title + article body), and the system will classify it as either **REAL** or **FAKE**.

---

## 🚀 Features

- NLP preprocessing (stopwords removal, punctuation cleaning, etc.)
- TF-IDF feature extraction
- Machine learning model: **Logistic Regression**
- Web interface using **Streamlit**
- Pretrained model (`model.pkl`) and vectorizer (`vectorizer.pkl`) included

---

## 📌 Steps Followed

### 1. Dataset Preparation
- Collected datasets: `Fake.csv` and `True.csv` from Kaggle.
- Merged both datasets and added a `target` column:  
  - `0` for fake news  
  - `1` for real news
- Combined `title` and `text` columns into a single column: `content`

### 2. Data Cleaning and Preprocessing
- Converted text to lowercase
- Removed digits, punctuation, and special characters using regex
- Removed common English stopwords using NLTK

### 3. Feature Extraction
- Used `TfidfVectorizer` with a limit of 5000 features to vectorize text

### 4. Model Training
- Split data into training and test sets (80/20)
- Trained a **Logistic Regression** model with `class_weight='balanced'`
- Evaluated using accuracy, confusion matrix, and classification report

### 5. Model Saving
- Saved the trained model (`model.pkl`) and TF-IDF vectorizer (`vectorizer.pkl`) using `joblib`

### 6. Web App Interface
- Built with **Streamlit**
- Users can paste news title + content and get prediction
- Displays result as: ✅ Real or 🚨 Fake

---

## 📂 Dataset Links

- 🗂️ **Fake News Dataset (Fake.csv)**: [Kaggle Link](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?select=Fake.csv)
- 🗂️ **True News Dataset (True.csv)**: [Kaggle Link](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?select=True.csv)

---

## ⚠️ Limitations

- The model is trained **only on Fake.csv and True.csv**, which contain **mostly American political news** from reliable/public sources.
- It may not generalize well to:
  - Non-political or entertainment news
  - News in other languages or regions
  - Very short or vague headlines without content
- Model is classical ML (Logistic Regression). Deep learning models (e.g. BERT) may improve accuracy.

---

# Run the web app
streamlit run app.py
