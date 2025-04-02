# Analyzing the tone of tweets using TextBlob and RandomForest

This project demonstrates the process of analyzing the tone of tweets. It involves pre-processing text, marking up the tonality using the TextBlob library, and training a RandomForestClassifier model to classify the tonality.

## Description

The main goal of the project is to demonstrate skills in Natural Language Processing (NLP) and machine learning for text analysis tasks.

**Includes the following steps:**

1.  **Data Loading:** Reading tweets from a CSV file.
2.  **Text preprocessing:**
    ** Tokenization
    ** Stop word removal
    ** Lemmatization
3.  **Tone analysis using TextBlob:** Each tweet is assigned a tone score, based on which a label ('positive', 'negative', 'neutral') is determined.
4.  **Preparing data for machine learning:**
    * Converting text labels into numerical labels.
    * Text vectorization using TF-IDF.
5.  **Model training:**
    * Splitting the data into training and test samples (this code uses a specific approach, see Note).
    * Training the `RandomForestClassifier'.
    * Selection of hyperparameters using `GridSearchCV`.
6.  **Model Evaluation:** Evaluating the quality of the model using F1 measure.

## Requirements

The following libraries are required to run the project:

* Python 3.x
* The main dependencies are listed in the `requirements.txt` file.

Additionally, the script loads NLTK resources: 'punkt', 'stopwords', 'wordnet'.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Jim-by/tweet-sentiment-analysis.git
    cd tweet-sentiment-analysis
    ```

2. (Recommended) Create and activate a virtual environment:
    ````bash
    python -m venv venv
    source venv/bin/activate # For Linux/Mac
    # venv\Scripts\activate # For Windows
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Make sure the `submission.csv` file is in the `data/` folder and contains a `selected_text` column with the texts of the tweets.
2.  Run the script:
    ``bash
    python src/sentiment_analysis.py
    ```

**Expected output:**

* The console will display progress messages, best model parameters and F1 measure.
* A `tweets_sentiments.csv` file will be created in the `data/` folder containing the original tweets, TextBlob analysis results, and preprocessed text.

## Note on model estimation

In this script, tone labels are generated programmatically using TextBlob on the entire dataset. A machine learning model is then trained to predict these labels. The evaluation is performed on the same dataset on which the labels were generated.

This demonstrates the model's ability to approximate TextBlob logic based on TF-IDF features. Evaluation on fully independent data would require “true” tone labels assigned by a human or other reliable source.

## Possible Improvements

* Use of more advanced Word Embeddings techniques such as Word2Vec, GloVe, or BERT.
* Utilizing other classification models.
* More thorough cleaning and preparation of text data.
