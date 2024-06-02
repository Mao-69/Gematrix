import os
import requests
from bs4 import BeautifulSoup
import urllib.parse
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy
import csv
import json
import numpy as np

nltk.download('vader_lexicon')
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")

def calculate_stylometric_features(text):
    words = nltk.word_tokenize(text)
    sentences = nltk.sent_tokenize(text)
    avg_word_length = sum(len(word) for word in words) / len(words) if len(words) > 0 else 0
    avg_sentence_length = len(words) / len(sentences) if len(sentences) > 0 else 0
    return avg_word_length, avg_sentence_length

def calculate_fingerprint_score(fingerprint):
    sentiment_score = fingerprint['Sentiment']['compound']
    avg_word_length, avg_sentence_length = calculate_stylometric_features(fingerprint['Word/Phrase'])
    fingerprint_score = sentiment_score + avg_word_length + avg_sentence_length
    return fingerprint_score

def scrape_gematrix_info(search_query, page=1):
    base_url = "https://www.gematrix.org"
    fingerprints = {}
    if search_query.isdigit():
        search_url = f"{base_url}/?word={search_query}&page={page}"
    else:
        search_query_url_encoded = urllib.parse.quote_plus(search_query)
        search_url = f"{base_url}/?word={search_query_url_encoded}&page={page}"
    
    try:
        response = requests.get(search_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        gematria_results = soup.find(id="results")
        if gematria_results:
            print(gematria_results.find('h2').text.strip())
            table = gematria_results.find('table', class_='results')
            if table:
                rows = table.find_all('tr')
                sid = SentimentIntensityAnalyzer()
                for row in rows:
                    data = [cell.get_text(strip=True) for cell in row.find_all('td')]
                    if len(data) == 5:
                        print("Word/Phrase:", data[0])
                        print("Jewish Gematria:", data[1])
                        print("English Gematria:", data[2])
                        print("Simple Gematria:", data[3])
                        print("Searches:", data[4])
                        sentiment_score = sid.polarity_scores(data[0])
                        print("Sentiment:", sentiment_score)
                        topic_model = perform_topic_modeling(data[0])
                        print("Topic Model:", topic_model)
                        entities = perform_named_entity_recognition(data[0])
                        print("Named Entities:", entities)
                        fingerprint = {
                            "Word/Phrase": data[0],
                            "Jewish Gematria": data[1],
                            "English Gematria": data[2],
                            "Simple Gematria": data[3],
                            "Searches": data[4],
                            "Sentiment": sentiment_score,
                            "Topic Model": topic_model,
                            "Named Entities": entities
                        }
                        fingerprint_score = calculate_fingerprint_score(fingerprint)
                        if fingerprint_score in fingerprints:
                            fingerprints[fingerprint_score]["Occurrences"] += 1
                        else:
                            fingerprints[fingerprint_score] = {"Fingerprint": fingerprint, "Occurrences": 1}
                        print("Fingerprint Score:", fingerprint_score)
                        print("Occurrences:", fingerprints[fingerprint_score]["Occurrences"])
                        
                        print("---------------------")
        else:
            print("No results found.")
    
    except requests.exceptions.RequestException as e:
        print("Error:", e)
    
    return fingerprints

def perform_topic_modeling(text):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([text])
    lda =LatentDirichletAllocation(n_components=1, random_state=42)
    lda.fit(X)
    dominant_topic = lda.components_[0].argsort()[-1]
    return dominant_topic

def perform_named_entity_recognition(text):
    doc = nlp(text)
    entities = [entity.text for entity in doc.ents]
    return entities

def save_to_csv(fingerprints, filename):
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    csv_filename = os.path.join(data_dir, filename)
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        fieldnames = ['Word/Phrase', 'Jewish Gematria', 'English Gematria', 'Simple Gematria',
                      'Sentiment', 'Topic Model', 'Named Entities', 'Fingerprint Score', 'Occurrences']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for score, data in fingerprints.items():
            writer.writerow({
                'Word/Phrase': data['Fingerprint']['Word/Phrase'],
                'Jewish Gematria': data['Fingerprint']['Jewish Gematria'],
                'English Gematria': data['Fingerprint']['English Gematria'],
                'Simple Gematria': data['Fingerprint']['Simple Gematria'],
                'Sentiment': data['Fingerprint']['Sentiment'],
                'Topic Model': data['Fingerprint']['Topic Model'],
                'Named Entities': data['Fingerprint']['Named Entities'],
                'Fingerprint Score': score,
                'Occurrences': data['Occurrences']
            })

def save_to_json(fingerprints, filename):
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    fingerprints_converted = {k: {kk: convert(vv) for kk, vv in v.items()} for k, v in fingerprints.items()}

    json_filename = os.path.join(data_dir, filename)
    with open(json_filename, 'w', encoding='utf-8') as file:
        json.dump(fingerprints_converted, file, indent=4, ensure_ascii=False, default=convert)

search_input = input("Enter a word, phrase, or number to search: ")
page = 1
fingerprints_all_pages = {}
while True:
    print(f"Page {page}:")
    fingerprints = scrape_gematrix_info(search_input, page)
    fingerprints_all_pages.update(fingerprints)
    csv_filename = f"{search_input}_page_{page}.csv"
    save_to_csv(fingerprints, csv_filename)
    json_filename = f"{search_input}_page_{page}.json"
    save_to_json(fingerprints, json_filename)
    similar_occurrences = {}
    threshold = 0.1
    for fingerprint_score, data in fingerprints.items():
        for other_score in fingerprints.keys():
            if fingerprint_score != other_score and abs(fingerprint_score - other_score) < threshold:
                if fingerprint_score not in similar_occurrences:
                    similar_occurrences[fingerprint_score] = 1
                else:
                    similar_occurrences[fingerprint_score] += 1
    print("Occurrences of Similarity:")
    print("---------------------")
    print("#####################")
    print("---------------------")
    for score, data in fingerprints.items():
        print(f"Word/Phrase: {data['Fingerprint']['Word/Phrase']}")
        print("---------------------")
        print(f"Jewish Gematria: {data['Fingerprint']['Jewish Gematria']}")
        print("---------------------")
        print(f"English Gematria: {data['Fingerprint']['English Gematria']}")
        print("---------------------")
        print(f"Simple Gematria: {data['Fingerprint']['Simple Gematria']}")
        print("---------------------")
        print(f"Sentiment: {data['Fingerprint']['Sentiment']}")
        print("---------------------")
        print(f"Topic Model: {data['Fingerprint']['Topic Model']}")
        print("---------------------")
        print(f"Named Entities: {data['Fingerprint']['Named Entities']}")
        print("---------------------")
        print(f"Fingerprint Score: {score}")
        print("---------------------")
        print(f"Occurrences: {data['Occurrences']}")
        if score in similar_occurrences:
            print(f"Occurrences of Similarity: {similar_occurrences[score]}")
        print("---------------------")
        print("#####################")
        print("---------------------")
    next_page = input("Do you want to view the next page? (y/n): ")
    if next_page.lower() != 'y':
        break
    page += 1
