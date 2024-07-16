from flask import Flask, request, jsonify
import requests
from bs4 import BeautifulSoup
from urllib.parse import unquote
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BartTokenizer, BartForConditionalGeneration
from langchain.text_splitter import RecursiveCharacterTextSplitter
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load models once when the application starts
bart_tokenizer, bart_model = BartTokenizer.from_pretrained("facebook/bart-large-cnn"), BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
senti_tokenizer, senti_model = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment"), AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

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

def summarize_text(tokenizer, model, text, max_chunk_length, summary_max_length):
    inputs = tokenizer(text, return_tensors="pt", max_length=max_chunk_length, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=summary_max_length, min_length=0, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def summarize_article(url):
    data = scraping_article(url)
    chunks = to_chunks(data)
    summaries = []
    for chunk in chunks:
        chunk_text = chunk
        summary = summarize_text(bart_tokenizer, bart_model, chunk_text, 3000, 800)
        summaries.append(summary)
    concatenated_summaries = " ".join(summaries)
    intermediate_chunks = [concatenated_summaries[i:i+3000] for i in range(0, len(concatenated_summaries), 3000)]
    final_summaries = []
    for intermediate_chunk in intermediate_chunks:
        final_summary = summarize_text(bart_tokenizer, bart_model, intermediate_chunk, 3000, 800)
        final_summaries.append(final_summary)    
    final_summary_text = " ".join(final_summaries)    
    return final_summary_text

def find_senti(news_texts):
    encoded = senti_tokenizer(news_texts, return_tensors='pt', truncation=True, padding=True, max_length=512)
    output = senti_model(**encoded)
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

@app.route('/analyze', methods=['POST'])
def analyze_stock():
    data = request.json
    stock_name = data['stock_name']
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

    result = {
        "urls": urls[:5],
        "avg_score": avg_score,
        "prediction": "up" if avg_score["avg_compound_score"] > 0 else "down"
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
