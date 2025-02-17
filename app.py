import requests
from flask import Flask, request, render_template, jsonify
import time
import pandas as pd
import numpy as np
import re
import pickle
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, RegexpTokenizer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Load models and tokenizers
models = {}
sentiment_models = {}
sentiment_tokenizers = {}
max_length = 500

# Load aspect models
aspect_list = ['Gameplay', 'Graphics', 'Story', 'Sound', 'Developer', 'Content',
              'Multiplayer', 'Performance', 'Value', 'No']

with open('aspect/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
for aspect in aspect_list:
    path = f'aspect/{aspect}.keras'
    models[aspect] = keras.models.load_model(path)

aspect_list = ['Gameplay', 'Graphics', 'Story', 'Sound', 'Developer', 'Content',
              'Multiplayer', 'Performance', 'Value', 'Overall']

# Load sentiment models
for aspect in aspect_list:
    model_path = f'sentiment/{aspect}_sentimen.keras'
    tokenizer_path = f'sentiment/tokenizer_{aspect}_sentimen.pkl'
    sentiment_models[aspect] = keras.models.load_model(model_path)
    with open(tokenizer_path, 'rb') as handle:
        sentiment_tokenizers[aspect] = pickle.load(handle)

# Mapping score to qualitative description
score_mapping = {
    '12/10': 'excellent',
    '11/10': 'excellent',
    '10/10': 'excellent',
    '9/10': 'great',
    '8/10': 'very good',
    '7/10': 'good',
    '6/10': 'decent',
    '5/10': 'average',
    '4/10': 'poor',
    '3/10': 'very poor',
    '2/10': 'terrible',
    '1/10': 'awful',
    '0/10': 'unplayable',
    '5/5': 'excellent',
    '4/5': 'very good',
    '3/5': 'decent',
    '2/5': 'poor',
    '1/5': 'terrible',
    '0/5': 'unplayable'
    }

to_replace = {'don\'t': 'do not', 'dont': 'do not',
    'doesn\'t': 'does not', 'doesnt': 'does not',
    'didn\'t': 'did not', 'didnt': 'did not',
    'shouldn\'t': 'should not', 'shouldnt': 'should not',
    'haven\'t': 'have not', 'hvn\'t': 'have not',
    'havent': 'have not', 'hadn\'t': 'had not',
    'hadnt': 'had not', 'cannt': 'can not',
    'cann\'t': 'can not', 'couldn\'t': 'could not',
    'couldnt': 'could not', 'wouldn\'t': 'would not',
    'wouldnt': 'would not', 'nt': 'not'
    }

def remove_elongation(kata):
    if wordnet.synsets(kata):
      return kata
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", kata)

def preprocessing(text):
    # Casefolding
    text = text.lower()

    # Remove review with ☐
    text = re.sub(r'☐.*', '', text).strip()

    # Replace abbreviation
    for abbreviation, full in to_replace.items():
        text = re.sub(re.escape(abbreviation), full, text)

    # Mapping score to qualitative description
    for score, description in score_mapping.items():
        text = text.replace(score, description)

    # Tokenization
    tokenizer = RegexpTokenizer(r'\w+|[^\w\s]')
    tokens = tokenizer.tokenize(text)

    # Change slang and abbreviation
    slang_df = pd.read_csv('Slang.csv', encoding='latin-1')
    slang_dict = dict(zip(slang_df['Slang'], slang_df['Baku']))
    tokens = [slang_dict.get(token, token) for token in tokens]

    # Stopword Removal
    stop_words = set(stopwords.words('english'))
    stop_words.remove('not')
    tokens = [token for token in tokens if token not in stop_words]

    # Remove special characters, numbers, empty string
    tokens = [remove_elongation(token) for token in tokens]
    tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens]
    tokens = [re.sub(r'[\u4e00-\u9fff]', '', token) for token in tokens]
    tokens = [re.sub(r'\d+', '', token) for token in tokens]
    tokens = [token for token in tokens if token]

    #Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return " ".join(tokens)

def get_reviews(appid, params={'json':1}):
        url = 'https://store.steampowered.com/appreviews/'
        response = requests.get(url=url+appid, params=params, headers={'User-Agent': 'Mozilla/5.0'})
        return response.json()

# Scrape review based on review_type (positive/negative), appid, and number of review scraped.
def get_n_reviews(review_type, appid, n):
    reviews = []
    cursor = '*'
    params = {
            'json' : 1,
            'filter' : 'all',
            'language' : 'english',
            'day_range' : 999999999,
            'review_type' : review_type,
            'purchase_type' : 'all'
            }

    while n > 0:
        params['cursor'] = cursor.encode()
        params['num_per_page'] = min(20, n)
        n -= 20

        response = get_reviews(appid, params)
        cursor = response['cursor']
        reviews += response['reviews']

    print(f"Number of reviews scraped: {len(reviews)}") 
    return reviews

def aspect_classification(df, tokenizer, max_length, models):
    # Padding data
    sequences = tokenizer.texts_to_sequences(df["preprocessed"].tolist())
    padded_sequences = pad_sequences(sequences, maxlen=max_length)

    # Predict aspect from review (1 - have aspect, 0 - no aspect)
    for aspect, model in models.items():
        pred_aspect = (model.predict(padded_sequences) > 0.5).astype("int32").flatten()
        df[aspect] = pred_aspect

    return df

def remove_noise(df):
    pred_data_noise = df['No']
    pred_aspect = df.columns.difference(['No', 'review', 'preprocessed'])
    aspect_count = df[pred_aspect].sum(axis=1)

    # Find index from review that predicted as noise
    noise_index = df.index[
        (pred_data_noise == 1) & (aspect_count <= 1) #| (aspect_count == 0)
    ]

    filtered_df = df.drop(index=noise_index)
    filtered_df = filtered_df.reset_index(drop=True)

    return filtered_df

def sentiment_classification(df, sentiment_models, sentiment_tokenizers, max_length):
    for aspect in sentiment_models.keys():
        df[f"{aspect}_sentimen"] = np.nan
        sentimen_model = sentiment_models[aspect]
        sentimen_tokenizer = sentiment_tokenizers[aspect]

        # Padding data
        sequences = sentimen_tokenizer.texts_to_sequences(df["preprocessed"].tolist())
        padded_sequences = pad_sequences(sequences, maxlen=max_length)

        # Predict sentiment from review (1 - positive, 0 - negative)
        # If aspect is Overall, predict all reviews
        if aspect == 'Overall':
            pred_overall = (sentimen_model.predict(padded_sequences) > 0.5).astype("int32").flatten()
            df['Overall_sentimen'] = pred_overall

        # If aspect is not Overall, predict only reviews with aspect=1 (have aspect in reviews)
        else:
            # Find index of aspect=1
            pred_aspect = df[aspect].values
            sentiment_indices = np.where(pred_aspect.flatten() == 1)[0]

            if sentiment_indices.size > 0:
                pred_sentiment = (sentimen_model.predict(padded_sequences[sentiment_indices]) > 0.5).astype("int32").flatten()
                df.loc[sentiment_indices, f"{aspect}_sentimen"] = pred_sentiment

    return df

def sentiment_category(positive_percentage):
    if positive_percentage > 80:
        return 'Very Positive'
    elif 55 < positive_percentage <= 80:
        return 'Positive'
    elif 45 <= positive_percentage <= 55:
        return 'Mixed'
    elif 20 < positive_percentage < 45:
        return 'Negative'
    else:
        return 'Very Negative'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search_games', methods=['GET'])
def search_games():
    query = request.args.get('query', '')
    games_df = pd.read_csv('appid.csv', usecols=['game_id', 'name'])
    
    if 'game_id' in games_df.columns and 'name' in games_df.columns:
        # Filter games based on the query
        filtered_games = games_df[games_df['name'].str.contains(query, case=False, na=False)]
        
        # Convert to dictionaries
        results = filtered_games.to_dict(orient='records')
        print(f"Results: {results}")
        return jsonify(results)
    else:
        print("Columns not found") 
        return jsonify([]) 

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    appid = request.form['appid']  # Get appid from user input
    game_name = request.form['game_name']

    review_type = ['positive', 'negative']
    reviews = []
    for sentiment in review_type:
        reviews.extend(get_n_reviews(sentiment, appid, 500)) 

    fetch_reviews_time = time.time()
    print(f"Time to fetch reviews: {fetch_reviews_time - start_time} seconds")

    # Check if there are no reviews
    if reviews == []:
        return render_template('results.html', game_name=game_name, results={}, message="The game doesn't have any reviews")
    
    # Preprocess reviews
    df_review = pd.DataFrame(reviews)[['review']]
    df_review['preprocessed'] = df_review['review'].apply(preprocessing)  # Implement preprocessing function
    df_review = df_review[df_review['preprocessed'].str.count(' ') + 1 > 1]
    df_review.reset_index()

    # Check if the number of reviews after preprocessing is less than 50
    if len(df_review) < 50:
        return render_template('results.html', game_name=game_name, results={}, message="The game doesn't have enough reviews")

    # Classify aspect and sentiment
    df_review_copy = df_review.copy()
    df_review = aspect_classification(df_review_copy, tokenizer, max_length, models)
    df_review = remove_noise(df_review)
    df_review = sentiment_classification(df_review, sentiment_models, sentiment_tokenizers, max_length)

    results = {}
    for aspect in aspect_list:
        if aspect != 'Overall':
            total_aspect = df_review[aspect].sum()
            no_data = total_aspect < 30
        else:
            no_data = False

        # Calculate percentage of positive and negative sentiment
        sentiment = f"{aspect}_sentimen"
        if sentiment in df_review.columns:
            total = df_review[sentiment].count()
            positive = (df_review[sentiment] == 1).sum()
            negative = (df_review[sentiment] == 0).sum()
            positive_percentage = (positive / total) * 100 if total > 0 else 0
            negative_percentage = (negative / total) * 100 if total > 0 else 0
            category = sentiment_category(positive_percentage)
            results[aspect] = {
                'category': category,
                'positive_percentage': positive_percentage,
                'negative_percentage': negative_percentage,
                'no_data': no_data
            }

    end_time = time.time()
    print(f"Total time: {end_time - start_time} seconds")

    return render_template('results.html', game_name=game_name, results=results)

if __name__ == '__main__':
    app.run(debug=True)