import requests
from bs4 import BeautifulSoup
from urllib.parse import unquote
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BartTokenizer, BartForConditionalGeneration
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

def scraping_article(url):
    headers = {
        'User-Agent': 'Your User Agent String',
    }
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = [paragraph.text for paragraph in paragraphs]
    words = ' '.join(text).split(' ')
    article = ' '.join(words)
    return article

def find_url(keyword):
    root = "https://www.google.com/"
    search_query = keyword.replace(" ", "+")
    link = f"https://www.google.com/search?q={search_query}&tbm=nws"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(link, headers=headers)
    webpage = response.content
    soup = BeautifulSoup(webpage, 'html5lib')
    links = []
    for div_tag in soup.find_all('div', class_='Gx5Zad'):
        a_tag = div_tag.find('a')
        if a_tag:
            if 'href' in a_tag.attrs:
                href = a_tag['href']
                if href.startswith('/url?q='):
                    url = href.split('/url?q=')[1].split('&sa=')[0]
                    links.append(url)
    return links

def to_chunks(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=50
    )
    docs = text_splitter.split_text(data)
    return docs

def load_bart_model(model_name="facebook/bart-large-cnn"):
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

def summarize_text(tokenizer, model, text, max_chunk_length, summary_max_length):
    inputs = tokenizer(text, return_tensors="pt", max_length=max_chunk_length, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=summary_max_length, min_length=200, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def summarize_article(url, model_name="facebook/bart-large-cnn"):
    data = scraping_article(url)
    chunks = to_chunks(data)
    tokenizer, model = load_bart_model(model_name)
    summaries = []
    for chunk in chunks:
        chunk_text = chunk
        summary = summarize_text(tokenizer, model, chunk_text, 3000, 800)
        summaries.append(summary)
    concatenated_summaries = " ".join(summaries)
    intermediate_chunks = [concatenated_summaries[i:i+3000] for i in range(0, len(concatenated_summaries), 3000)]
    final_summaries = []
    for intermediate_chunk in intermediate_chunks:
        final_summary = summarize_text(tokenizer, model, intermediate_chunk, 3000, 800)
        final_summaries.append(final_summary)    
    final_summary_text = " ".join(final_summaries)    
    return final_summary_text

def senti_model(model_name="cardiffnlp/twitter-roberta-base-sentiment"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

def find_senti(news_texts):
    tokenizer, model = senti_model()
    encoded = tokenizer(news_texts, return_tensors='pt', truncation=True, padding=True, max_length=512)
    output = model(**encoded)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    weights = {
        'neg': -1,
        'neu': 0,
        'pos': 1
    }
    probabilities = {
        'neg': scores[0],
        'neu': scores[1],
        'pos': scores[2]
    }
    compound_score = sum(probabilities[label] * weights[label] for label in probabilities)
    senti_dict = {
        'neg': scores[0],
        'neu': scores[1],
        'pos': scores[2],
        'polarity': compound_score        
    }
    return senti_dict

def extract_features(summary):
    sentiment_scores = find_senti(summary)
    features = {
        'compound_sentiment_score': sentiment_scores['polarity'],  
        'negative_sentiment_score': sentiment_scores['neg'],
        'neutral_sentiment_score': sentiment_scores['neu'],
        'positive_sentiment_score': sentiment_scores['pos']
    }    
    return features

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    
    # Separate features and target
    X = df[['avg_compound_sentiment_score', 'avg_negative_sentiment_score', 
            'avg_neutral_sentiment_score', 'avg_positive_sentiment_score']]
    y = df['Movement']
    
    return X, y

def train_xgboost(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    params = {
        'max_depth': 3,
        'eta': 0.1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }
    
    num_round = 100
    xgb_model = xgb.train(params, dtrain, num_round)
    
    y_pred = xgb_model.predict(dtest)
    y_pred_binary = [1 if y > 0.5 else 0 for y in y_pred]
    
    accuracy = accuracy_score(y_test, y_pred_binary)
    
    print("XGBoost Model Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_binary))
    
    return xgb_model, accuracy

def predict_stock_movement(xgb_model, features):
    dfeatures = xgb.DMatrix([features])
    prediction = xgb_model.predict(dfeatures)
    return "up" if prediction[0] > 0.5 else "down"

def analyze_stock(stock_name):
    urls = find_url(stock_name)
    summaries = []
    for i in range(5):
        summary = summarize_article(urls[i])
        summaries.append(summary)

    all_scores = []
    for i in range(5):
        scores = extract_features(summaries[i])
        all_scores.append(scores)

    avg_score = {}
    avg_comp = 0
    avg_pos = 0
    avg_neg = 0
    avg_neu = 0
    for i in range(5):
        avg_comp += all_scores[i]["compound_sentiment_score"]
        avg_neg += all_scores[i]["negative_sentiment_score"]
        avg_neu += all_scores[i]["neutral_sentiment_score"]
        avg_pos += all_scores[i]["positive_sentiment_score"]
    avg_score["avg_compound_score"] = avg_comp / 5
    avg_score["avg_negative_score"] = avg_neg / 5
    avg_score["avg_neutral_score"] = avg_neu / 5
    avg_score["avg_positive_score"] = avg_pos / 5

    result = "Based on sentiment analysis:\n"
    if avg_score["avg_compound_score"] > 0:
        result += "The overall sentiment is positive.\n"
    else:
        result += "The overall sentiment is negative.\n"

    return result, avg_score, urls, summaries

if __name__ == "__main__":
    # Load and prepare data
    X, y = load_and_prepare_data("update.csv")
    
    # Train XGBoost model
    xgb_model, model_accuracy = train_xgboost(X, y)
    
    print(f"\nOverall XGBoost Model Accuracy: {model_accuracy:.2f}")
    
    stock_name = input("\nEnter Stock Name: ")
    result, avg_score, urls, summaries = analyze_stock(stock_name)
    
    print("\n" + "="*50)
    print(result)
    print("Average Sentiment Scores:")
    for key, value in avg_score.items():
        print(f"{key}: {value:.4f}")
    
    print("\nArticle Summaries and URLs:")
    for i, (url, summary) in enumerate(zip(urls[:5], summaries), 1):
        print(f"\nArticle {i}:")
        print(f"URL: {url}")
        print(f"Summary: {summary}")
    
    # Prepare features for prediction
    features = [
        avg_score["avg_compound_score"],
        avg_score["avg_negative_score"],
        avg_score["avg_neutral_score"],
        avg_score["avg_positive_score"]
    ]
    
    # Predict stock movement using XGBoost
    xgb_prediction = predict_stock_movement(xgb_model, features)
    print("\n" + "="*50)
    print(f"XGBoost Prediction: Based on the current sentiment analysis and historical data,")
    print(f"the stock is likely to go {xgb_prediction}.")
    print(f"Model Accuracy: {model_accuracy:.2f}")
    print("="*50)