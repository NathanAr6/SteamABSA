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

list_aspek = ['Gameplay', 'Graphics', 'Story', 'Sound', 'Developer', 'Content',
              'Multiplayer', 'Performance', 'Value', 'No']
# Load aspect models
with open('D:/ABSA/aspek/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

for aspek in list_aspek:
    path = f'D:/ABSA/aspek/{aspek}.keras'
    models[aspek] = keras.models.load_model(path)

list_aspek = ['Gameplay', 'Graphics', 'Story', 'Sound', 'Developer', 'Content',
              'Multiplayer', 'Performance', 'Value', 'Overall']
# Load sentiment models and tokenizers
for aspek in list_aspek:
    model_path = f'D:/ABSA/sentimen/{aspek}_sentimen.keras'
    tokenizer_path = f'D:/ABSA/sentimen/tokenizer_{aspek}_sentimen.pkl'
    sentiment_models[aspek] = keras.models.load_model(model_path)
    with open(tokenizer_path, 'rb') as handle:
        sentiment_tokenizers[aspek] = pickle.load(handle)

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

# Mengubah singkatan dari kata-kata negatif menjadi kepanjangannya
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

def hapus_huruf_berulang(kata):
    if wordnet.synsets(kata):
      return kata
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", kata)

def preprocessing(teks):
    # Mengubah menjadi huruf kecil
    teks = teks.lower()

    # Menghapus kalimat yang diawali ☐
    teks = re.sub(r'☐.*', '', teks).strip()

    # Mengubah singkatan dari kata-kata negatif menjadi kepanjangannya
    for singkatan, kepanjangan in to_replace.items():
        teks = re.sub(re.escape(singkatan), kepanjangan, teks)

    # Mapping score to qualitative description
    for score, description in score_mapping.items():
        teks = teks.replace(score, description)

    # Tokenization
    tokenizer = RegexpTokenizer(r'\w+|[^\w\s]')
    tokens = tokenizer.tokenize(teks)

    # Mengubah slang dan abbreviation
    slang_df = pd.read_csv('D:\ABSA\Slang.csv', encoding='latin-1')
    slang_dict = dict(zip(slang_df['Slang'], slang_df['Baku']))
    tokens = [slang_dict.get(token, token) for token in tokens]

    # Stopword Removal
    stop_words = set(stopwords.words('english'))
    stop_words.remove('not')
    tokens = [token for token in tokens if token not in stop_words]

    # Menghapus special characters, angka, huruf yang berulang, string kosong
    tokens = [hapus_huruf_berulang(token) for token in tokens]
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

# Fungsi untuk mengambil sejumlah ulasan berdasarkan tipe ulasan (positif/negatif), appid, dan jumlah ulasan yang diinginkan.
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

        #if len(response['reviews']) < 100: break

    print(f"Number of reviews scraped: {len(reviews)}")  # Add this line to print the number of reviews scraped
    return reviews

def klasifikasi_aspek(df, tokenizer, max_length, models):
    # Padding data
    sequences = tokenizer.texts_to_sequences(df["preprocessed"].tolist())
    padded_sequences = pad_sequences(sequences, maxlen=max_length)

    # Prediksi aspek dari ulasan
    for aspek, model in models.items():
        pred_aspek = (model.predict(padded_sequences) > 0.5).astype("int32").flatten()
        df[aspek] = pred_aspek

    return df

def hapus_noise(df):
    pred_data_noise = df['No']
    pred_aspek = df.columns.difference(['No', 'review', 'preprocessed'])
    aspect_count = df[pred_aspek].sum(axis=1)

    # Cari index dari noise
    noise_index = df.index[
        (pred_data_noise == 1) & (aspect_count <= 1) #| (aspect_count == 0)
    ]

    filtered_df = df.drop(index=noise_index)
    filtered_df = filtered_df.reset_index(drop=True)

    return filtered_df

def klasifikasi_sentimen(df, sentiment_models, sentiment_tokenizers, max_length):
    for aspek in sentiment_models.keys():
        df[f"{aspek}_sentimen"] = np.nan
        sentimen_model = sentiment_models[aspek]
        sentimen_tokenizer = sentiment_tokenizers[aspek]

        # Padding data
        sequences = sentimen_tokenizer.texts_to_sequences(df["preprocessed"].tolist())
        padded_sequences = pad_sequences(sequences, maxlen=max_length)

        if aspek == 'Overall':
            pred_overall = (sentimen_model.predict(padded_sequences) > 0.5).astype("int32").flatten()
            df['Overall_sentimen'] = pred_overall

        else:
            # Cari indeks dimana aspek=1
            pred_aspek = df[aspek].values
            sentiment_indices = np.where(pred_aspek.flatten() == 1)[0]

            if sentiment_indices.size > 0:
                # Prediksi sentimen dari ulasan
                pred_sentimen = (sentimen_model.predict(padded_sequences[sentiment_indices]) > 0.5).astype("int32").flatten()
                df.loc[sentiment_indices, f"{aspek}_sentimen"] = pred_sentimen

    return df

def kategori_sentimen(persentase_positif):
    if persentase_positif > 80:
        return 'Very Positive'
    elif 55 < persentase_positif <= 80:
        return 'Positive'
    elif 45 <= persentase_positif <= 55:
        return 'Mixed'
    elif 20 < persentase_positif < 45:
        return 'Negative'
    else:
        return 'Very Negative'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search_games', methods=['GET'])
def search_games():
    query = request.args.get('query', '')
    games_df = pd.read_csv('D:/ABSA/appid.csv', usecols=['game_id', 'name'])
    
    # Ensure the column names match the CSV file
    if 'game_id' in games_df.columns and 'name' in games_df.columns:
        # Filter games based on the query
        filtered_games = games_df[games_df['name'].str.contains(query, case=False, na=False)]
        
        # Convert to a list of dictionaries
        results = filtered_games.to_dict(orient='records')
        print(f"Results: {results}")  # Debugging statement
        
        return jsonify(results)
    else:
        print("Columns not found")  # Debugging statement
        return jsonify([])  # Return an empty list if columns are not found

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    
    appid = request.form['appid']  # Get appid from user input
    game_name = request.form['game_name']
    review_type = ['positive', 'negative']
    reviews = []

    for sentiment in review_type:
        reviews.extend(get_n_reviews(sentiment, appid, 300))  # Reduced to 200 reviews

    fetch_reviews_time = time.time()
    print(f"Time to fetch reviews: {fetch_reviews_time - start_time} seconds")

    if reviews == []:
        return render_template('results.html', game_name=game_name, results={}, message="The game doesn't have any reviews")
    
    df_review = pd.DataFrame(reviews)[['review']]
    df_review['preprocessed'] = df_review['review'].apply(preprocessing)  # Implement preprocessing function
    df_review = df_review[df_review['preprocessed'].str.count(' ') + 1 > 1]
    df_review.reset_index()

    # Check if the number of reviews after preprocessing is less than 50
    if len(df_review) < 50:
        return render_template('results.html', game_name=game_name, results={}, message="The game doesn't have enough reviews")

    df_review_copy = df_review.copy()
    df_review = klasifikasi_aspek(df_review_copy, tokenizer, max_length, models)
    df_review = hapus_noise(df_review)
    df_review = klasifikasi_sentimen(df_review, sentiment_models, sentiment_tokenizers, max_length)

    results = {}
    for aspek in list_aspek:
        if aspek != 'Overall':
            total_aspek = df_review[aspek].sum()
            no_data = total_aspek < 30
        else:
            no_data = False

        sentimen = f"{aspek}_sentimen"
        if sentimen in df_review.columns:
            total = df_review[sentimen].count()
            positif = (df_review[sentimen] == 1).sum()
            negatif = (df_review[sentimen] == 0).sum()
            persentase_positif = (positif / total) * 100 if total > 0 else 0
            persentase_negatif = (negatif / total) * 100 if total > 0 else 0
            kategori = kategori_sentimen(persentase_positif)
            results[aspek] = {
                'kategori': kategori,
                'persentase_positif': persentase_positif,
                'persentase_negatif': persentase_negatif,
                'no_data': no_data
            }

    end_time = time.time()
    print(f"Total time: {end_time - start_time} seconds")

    return render_template('results.html', game_name=game_name, results=results)

if __name__ == '__main__':
    app.run(debug=True)