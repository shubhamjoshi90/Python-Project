import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

import nltk
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def read_input_data(url_file, positive_keywords_file, negative_keywords_file):
    # Read URLs from Excel file
    try:
        df_urls = pd.read_excel(url_file)
    except Exception as e:
        print(f"Error reading URL file: {e}")
        return None, None, None
    
    # Read positive keywords
    try:
        with open(positive_keywords_file, 'r') as file:
            positive_keywords = file.read().splitlines()
    except Exception as e:
        print(f"Error reading positive keywords file: {e}")
        return None, None, None
    
    # Read negative keywords
    try:
        with open(negative_keywords_file, 'r') as file:
            negative_keywords = file.read().splitlines()
    except Exception as e:
        print(f"Error reading negative keywords file: {e}")
        return None, None, None
    
    return df_urls, positive_keywords, negative_keywords

def fetch_webpage_content(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            return ""
    except Exception as e:
        print(f"Error fetching URL {url}: {e}")
        return ""

def clean_text(text):
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    
    text = re.sub(r'[^\w\s]', '', text).lower()
    
    words = word_tokenize(text)
    cleaned_text = [word for word in words if word not in stop_words]
    
    return cleaned_text

def count_keywords(cleaned_text, positive_keywords, negative_keywords):
    positive_count = sum(word in positive_keywords for word in cleaned_text)
    negative_count = sum(word in negative_keywords for word in cleaned_text)
    return positive_count, negative_count

def calculate_scores(positive_count, negative_count, total_words):
    polarity_score = (positive_count - negative_count) / ((positive_count + negative_count) + 0.000001)
    subjectivity_score = (positive_count + negative_count) / (total_words + 0.000001)
    return polarity_score, subjectivity_score

def calculate_text_metrics(cleaned_text):
    sentences = sent_tokenize(" ".join(cleaned_text))
    words = cleaned_text
    num_sentences = len(sentences)
    num_words = len(words)
    
    avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0
    
    def syllable_count(word):
        vowels = "aeiou"
        return sum(1 for char in word if char in vowels)
    
    complex_words = [word for word in words if syllable_count(word) > 2]
    percentage_complex_words = len(complex_words) / num_words if num_words > 0 else 0
    
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
    
    avg_words_per_sentence = num_words / num_sentences if num_sentences > 0 else 0
    
    return avg_sentence_length, percentage_complex_words, fog_index, avg_words_per_sentence

def analyze_urls(urls_df, positive_keywords, negative_keywords):
    results = []

    if urls_df is None or positive_keywords is None or negative_keywords is None:
        print("Error: Input data is missing or invalid.")
        return pd.DataFrame(results)

    for index, row in urls_df.iterrows():
        url_id = row['URL_ID']
        url = row['URL']
        
        content = fetch_webpage_content(url)
        if not content:
            continue

        cleaned_text = clean_text(content)
        
        positive_count, negative_count = count_keywords(cleaned_text, positive_keywords, negative_keywords)
        polarity_score, subjectivity_score = calculate_scores(positive_count, negative_count, len(cleaned_text))
        
        avg_sentence_length, percentage_complex_words, fog_index, avg_words_per_sentence = calculate_text_metrics(cleaned_text)
        
        results.append({
            'url_ID': url_id,
            'URL': url,
            'Positive Score': positive_count,
            'Negative Score': negative_count,
            'Polarity Score': polarity_score,
            'Subjectivity Score': subjectivity_score,
            'Average Sentence Length': avg_sentence_length,
            'Percentage of Complex Words': percentage_complex_words,
            'Fog Index': fog_index,
            'Average Words Per Sentence': avg_words_per_sentence
        })
    
    return pd.DataFrame(results)

# Example usage:
url_file = 'url_file.xlsx'
positive_keywords_file = 'positive_keywords_file.txt'
negative_keywords_file = 'negative_keywords_file.txt'

df_urls, positive_keywords, negative_keywords = read_input_data(url_file, positive_keywords_file, negative_keywords_file)
if df_urls is not None and positive_keywords is not None and negative_keywords is not None:
    results_df = analyze_urls(df_urls, positive_keywords, negative_keywords)
    print(results_df)
