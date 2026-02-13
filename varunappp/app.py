from flask import Flask, render_template, request
from google_play_scraper import app as gp_app, reviews, reviews_all, Sort
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.utils
import json
import numpy as np
from collections import defaultdict

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

app = Flask(__name__)

ASPECTS = {
    'usability': ['use', 'user-friendly', 'intuitive', 'easy', 'difficult', 'complicated', 'interface', 'navigation', 'simple', 'hard'],
    'performance': ['fast', 'slow', 'speed', 'crash', 'bug', 'glitch', 'freeze', 'responsive', 'performance', 'loading', 'lag'],
    'design': ['design', 'layout', 'look', 'ui', 'ux', 'beautiful', 'ugly', 'clean', 'modern', 'interface', 'theme', 'style'],
    'features': ['feature', 'functionality', 'option', 'capability', 'tool', 'function', 'add', 'suggestion', 'request'],
    'reliability': ['reliable', 'stable', 'consistent', 'crash', 'bug', 'error', 'issue', 'problem', 'work', 'broken'],
    'support': ['support', 'help', 'contact', 'customer service', 'response', 'assistance', 'email'],
    'privacy': ['privacy', 'data', 'security', 'permission', 'safe', 'secure', 'trust'],
    'price': ['price', 'cost', 'free', 'paid', 'subscription', 'purchase', 'expensive', 'cheap'],
    'updates': ['update', 'version', 'latest', 'new', 'old', 'frequent', 'improvement']
}

def get_playstore_reviews(app_id, analysis_type='quick'):
    try:
        if analysis_type == 'full':
            all_reviews = reviews_all(
                app_id,
                sleep_milliseconds=0,
                lang='en',
                country='us',
                sort=Sort.MOST_RELEVANT,
            )
            return [{'content': review['content'], 'score': review['score']} 
                   for review in all_reviews if review.get('content')]
        else:
            result, _ = reviews(
                app_id,
                lang='en',
                country='us',
                sort=Sort.MOST_RELEVANT,
                count=500
            )
            return [{'content': review['content'], 'score': review['score']} 
                   for review in result if review.get('content')]

    except Exception as e:
        print(f"Error: {e}")
        return None

def identify_aspects(text):
    doc = nlp(text.lower())
    found_aspects = defaultdict(float)
    
    # Get all words and lemmatize them
    text_words = {token.lemma_ for token in doc}
    
    # Check each aspect
    for aspect, keywords in ASPECTS.items():
        for keyword in keywords:
            if keyword in text_words:
                found_aspects[aspect] += 1
                
    return dict(found_aspects)

def analyze_sentiment(reviews):
    if not reviews:
        return {}, {}, [], {}, {}
    
    sia = SentimentIntensityAnalyzer()
    counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    aspect_sentiments = defaultdict(lambda: defaultdict(int))
    detailed_results = []
    aspect_scores = defaultdict(list)
    
    for review in reviews:
        text = review['content']
        score = sia.polarity_scores(text)['compound']
        
        # Overall sentiment
        sentiment = 'neutral'
        if score >= 0.05:
            sentiment = 'positive'
            counts['positive'] += 1
        elif score <= -0.05:
            sentiment = 'negative'
            counts['negative'] += 1
        else:
            counts['neutral'] += 1
            
        # Aspect-based analysis
        aspects = identify_aspects(text)
        review_aspects = {}
        
        for aspect, presence in aspects.items():
            if presence > 0:
                aspect_score = sia.polarity_scores(text)['compound']
                aspect_scores[aspect].append(aspect_score)
                
                if aspect_score >= 0.05:
                    aspect_sentiments[aspect]['positive'] += 1
                elif aspect_score <= -0.05:
                    aspect_sentiments[aspect]['negative'] += 1
                else:
                    aspect_sentiments[aspect]['neutral'] += 1
                    
                review_aspects[aspect] = aspect_score

        detailed_results.append({
            'text': text,
            'sentiment': sentiment,
            'aspects': review_aspects
        })

    total = len(reviews)
    percentages = {k: round((v/total)*100, 2) for k, v in counts.items()} if total > 0 else {}
    
    # Calculate average sentiment scores for each aspect
    aspect_averages = {}
    for aspect, scores in aspect_scores.items():
        if scores:
            aspect_averages[aspect] = sum(scores) / len(scores)
    
    return percentages, counts, detailed_results, dict(aspect_sentiments), aspect_averages

def create_visualizations(results, aspect_sentiments, aspect_averages):
    # Overall sentiment pie chart
    fig1 = go.Figure(data=[go.Pie(
        labels=['Positive', 'Neutral', 'Negative'],
        values=[results['positive'], results['neutral'], results['negative']],
        marker=dict(colors=['#43e97b', '#ffd700', '#ff6f61'])
    )])
    fig1.update_layout(title='Overall Sentiment Distribution')
    
    # Aspect-based sentiment bar chart
    aspects = list(aspect_sentiments.keys())
    positive_vals = [aspect_sentiments[aspect]['positive'] for aspect in aspects]
    neutral_vals = [aspect_sentiments[aspect]['neutral'] for aspect in aspects]
    negative_vals = [aspect_sentiments[aspect]['negative'] for aspect in aspects]

    fig2 = go.Figure(data=[
        go.Bar(name='Positive', x=aspects, y=positive_vals, marker_color='#43e97b'),
        go.Bar(name='Neutral', x=aspects, y=neutral_vals, marker_color='#ffd700'),
        go.Bar(name='Negative', x=aspects, y=negative_vals, marker_color='#ff6f61')
    ])
    fig2.update_layout(
        title='Aspect-based Sentiment Analysis',
        barmode='stack'
    )
    
    # Radar chart for aspect sentiment scores
    fig3 = go.Figure(data=go.Scatterpolar(
        r=[aspect_averages.get(aspect, 0) for aspect in aspects],
        theta=aspects,
        fill='toself',
        marker_color='#ff6f61'
    ))
    fig3.update_layout(
        title='Aspect Sentiment Scores',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-1, 1]
            )
        )
    )
    
    return {
        'sentiment_pie': json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder),
        'aspect_bar': json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder),
        'aspect_radar': json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    counts = None
    detailed_reviews = None
    error = None
    app_title = None
    charts = None
    aspect_sentiments = None
    aspect_averages = None
    
    if request.method == 'POST':
        app_url_or_id = request.form.get('app_url')
        analysis_type = request.form.get('analysis_type', 'quick')
        
        if app_url_or_id:
            try:
                app_id = None
                match = re.search(r'id=([^&]+)', app_url_or_id)
                if match:
                    app_id = match.group(1)
                elif '.' in app_url_or_id and ' ' not in app_url_or_id:
                    app_id = app_url_or_id

                if not app_id:
                    error = "Invalid input. Please enter a valid Google Play Store URL or an App ID."
                else:
                    app_details = gp_app(app_id)
                    app_title = app_details['title']
                    
                    fetched_reviews = get_playstore_reviews(app_id, analysis_type)
                    if fetched_reviews is not None:
                        results, counts, detailed_reviews, aspect_sentiments, aspect_averages = analyze_sentiment(fetched_reviews)
                        charts = create_visualizations(results, aspect_sentiments, aspect_averages)
                    else:
                        error = "Unable to fetch reviews. The app may not exist or has no reviews."
            except Exception as e:
                error = f"An error occurred: Please check the app URL or ID. Details: {e}"
    
    return render_template(
        'index.html',
        results=results,
        counts=counts,
        detailed_reviews=detailed_reviews,
        error=error,
        app_title=app_title,
        charts=charts,
        aspect_sentiments=aspect_sentiments,
        aspect_averages=aspect_averages
    )

if __name__ == '__main__':
    app.run(debug=True)
