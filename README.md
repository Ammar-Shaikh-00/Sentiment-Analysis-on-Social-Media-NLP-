# Sentiment Analysis on Social Media (NLP)

# Project Overview

This project aims to perform sentiment analysis on social media text using NLP techniques. The workflow includes data preprocessing, training a baseline model using TF-IDF and Logistic Regression, fine-tuning a pre-trained DistilBERT model, and deploying the model as an API using FastAPI.

✅ Data Preprocessing: Cleaning text (removing stopwords, URLs, emojis).
✅ Baseline Model: TF-IDF + Logistic Regression.
✅ Fine-tuning DistilBERT: Using Hugging Face Transformers.
✅ Deploying as an API: FastAPI-based real-time sentiment prediction.

# Dataset Selection & Loading
We use the Twitter Sentiment Analysis Dataset from Kaggle.

# 🔹 Download the Dataset
Go to Kaggle Twitter Sentiment Dataset.
Download the training.1600000.processed.noemoticon.csv file.
Move it into your project directory.
Installation & Setup

# 🔹 Clone the Repository
sh
Copy
Edit
git clone https://github.com/Ammar-Shaikh-00/Sentiment-Analysis-NLP.git
cd Sentiment-Analysis-NLP

# 🔹 Install Dependencies
sh
Copy
Edit
pip install -r requirements.txt

# 🔹 Running the API
Ensure all dependencies are installed.
Start the FastAPI server:
sh
Copy
Edit
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
Access the API Documentation:
Open: http://127.0.0.1:8000/docs


# Training the Sentiment Analysis Model

# 🔹 Steps
Preprocessing:

Remove stopwords, punctuation, URLs, and emojis.
Convert text to lowercase and tokenize it.
Train a Baseline Model:

Use TF-IDF for text vectorization.
Train a Logistic Regression model.
Fine-Tune DistilBERT:

Use Hugging Face Transformers.
Tokenize using DistilBertTokenizer.
Train on a cleaned dataset.


# Deploying as an API
Once trained, the model is deployed using FastAPI. The API allows real-time sentiment classification.

# 🔹 Endpoints

Method	Endpoint	Description
POST	/predict	Predict sentiment from text.
GET	/docs	API documentation.

# Model Storage
Since GitHub LFS has storage limits, the trained model is stored separately.

# Download Model from Google Drive
Go to Google Drive Model Link: Download Here(https://drive.google.com/drive/folders/13cEWBNCzru9snUNzgYZ5OO5IcWf2EEK2?usp=sharing).
Download distilbert_sentiment_model.pth.
Place it in the sentiment_model directory.


# Project Status & Future Improvements
✅ Completed:

Successfully trained DistilBERT for sentiment analysis.
Achieved high accuracy (~89%) on Twitter sentiment dataset.
Deployed the model as an API.
🚀 Future Enhancements:

Add real-time streaming analysis for Twitter & Reddit.
Optimize model inference speed.
Extend to multi-language sentiment classification.


# Contributing
Feel free to contribute by opening an issue or pull request! If you have suggestions for improvements, reach out.

# 🔹 Fork & Contribute
sh
Copy
Edit
git clone https://github.com/Ammar-Shaikh-00/Sentiment-Analysis-on-Social-Media-NLP-.git
cd Sentiment-Analysis-NLP
git checkout -b feature-branch

# Author
👨‍💻 Ammar Shaikh
🔗 GitHub: Ammar-Shaikh-00
📧 Email: m.ammarshaikh31@gmail.com
