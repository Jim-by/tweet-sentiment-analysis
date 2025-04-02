import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from textblob import TextBlob 

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Setup paths relative to the script file or project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
input_file = os.path.join(project_root, "data", "submission.csv")
output_file = os.path.join(project_root, "data", "tweets_sentiments.csv")

# Define functions
def preprocess_text(text):
    """Preprocess text by tokenizing, removing stopwords and lemmatizing."""
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(words)

def get_sentiment(text): # Определена функция get_sentiment
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def get_sentiment_label(sentiment_score): # Определена функция get_sentiment_label
    if sentiment_score > 0.1:
        return 'positive'
    elif sentiment_score < -0.1:
        return 'negative'
    else:
        return 'neutral'

def main():
    # Load data
    try:
        tweets = pd.read_csv(input_file)
        print(f"Loaded {len(tweets)} tweets from {input_file}")
    except FileNotFoundError:
        print(f"Error: File {input_file} not found")
        return

    # Preprocess text data
    tweets['processed_text'] = tweets['selected_text'].apply(preprocess_text)

    # Calculate sentiment using TextBlob
    tweets['sentiment'] = tweets['selected_text'].apply(get_sentiment)
    tweets['sentiment_label'] = tweets['sentiment'].apply(get_sentiment_label)

    # Save sentiment analysis results
    tweets.to_csv(output_file, index=False)
    print(f"Sentiment analysis results saved to {output_file}")

    # Prepare data for machine learning
    reviews = pd.read_csv(output_file)
    reviews['sentiment_label'] = reviews['sentiment_label'].replace({'negative': 0, 'positive': 1, 'neutral': 2})

    # Remove rows with missing values in 'processed_text'
    reviews = reviews.dropna(subset=['processed_text'])

    # Shuffle data and split into training and test sets
    reviews = shuffle(reviews, random_state=42)
    labeled_reviews, _ = train_test_split(reviews, train_size=0.8, random_state=42)

    # Vectorize text data
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(labeled_reviews['processed_text'])
    y = labeled_reviews['sentiment_label']

    # Define model and parameters for GridSearchCV
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X, y)

    # Best model and score
    best_model = grid_search.best_estimator_
    print(f"Best model parameters: {grid_search.best_params_}")

    # Test model
    reviews_test = pd.read_csv(output_file)
    reviews_test['sentiment_label'] = reviews_test['sentiment_label'].replace({'negative': 0, 'positive': 1, 'neutral': 2})

    # Remove rows with missing values in 'processed_text' for test set
    reviews_test = reviews_test.dropna(subset=['processed_text'])

    X_test = vectorizer.transform(reviews_test['processed_text'])
    y_test_predicted = best_model.predict(X_test)

    # Evaluate model
    f1 = f1_score(reviews_test['sentiment_label'], y_test_predicted, average='macro')
    f1_percent = f1 * 100
    print(f"Model prediction accuracy: {f1_percent:.2f}%")

if __name__ == "__main__":
    main()