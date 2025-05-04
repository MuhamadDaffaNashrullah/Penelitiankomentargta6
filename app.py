from flask import Flask, render_template, request, jsonify
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
from textblob import TextBlob
import re
import base64
import io
from config import ProductionConfig

app = Flask(__name__)
app.config.from_object(ProductionConfig)

# Fungsi-fungsi analisis sentimen
def contains_emoji(text):
    return any(char in text for char in ['😊', '👍', '💪', '🙏', '😂', '🤣', '😅', '😭', '😢', '😔', '😩', '😫', '😤', '😡', '❤️', '💕', '💗', '💓', '💝', '💖', '♥️', '😍', '🥰', '😘', '😗', '😚', '😙', '🤗', '🫂', '👏', '🙌', '✌️', '🤝', '🫡', '🤩', '🌟', '⭐', '✨', '💫', '🎉', '🎊'])

def count_repeated_chars(text):
    patterns = ['w+k+w+k+', 'h+a+h+a+', 'h+e+h+e+', 'h+i+h+i+', 'x+i+x+i+']
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, text.lower()))
    return count

def count_punctuation(text):
    return len(re.findall(r'[!?]{2,}|\.{2,}', text))

def get_emoji_sentiment(text):
    positive_emoji = ['😊', '👍', '💪', '🙏', '❤️', '💕', '😍', '🥰', '🤗']
    negative_emoji = ['😢', '😭', '😔', '😩', '😫', '😤', '😡']
    
    pos_count = sum(text.count(emoji) for emoji in positive_emoji)
    neg_count = sum(text.count(emoji) for emoji in negative_emoji)
    
    if pos_count > neg_count:
        return 1
    elif neg_count > pos_count:
        return -1
    return 0

def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    text_lower = text.lower()
    
    positive_keywords = [
        'awesome', 'amazing', 'incredible', 'fantastic', 'epic', 'masterpiece', 'perfect',
        'hype', 'hyped', 'fire', 'lit', 'goat', 'legendary', 'insane', 'stunning',
        'beautiful', 'brilliant', 'superb', 'excellent', 'love', 'best', 'great',
        'sick', 'dope', 'cool', 'wow', 'impressive', 'revolutionary', 'next level'
    ]
    
    negative_keywords = [
        'trash', 'garbage', 'terrible', 'awful', 'disappointing', 'bad', 'worst',
        'hate', 'sucks', 'boring', 'overrated', 'mid', 'meh', 'waste', 'ruined',
        'broken', 'buggy', 'glitchy', 'unplayable', 'lag', 'crash', 'fail',
        'scam', 'ripoff', 'expensive', 'greedy', 'cash grab', 'downgrade'
    ]

    has_negative = any(keyword in text_lower for keyword in negative_keywords)
    has_positive = any(keyword in text_lower for word in positive_keywords for keyword in [word, f"{word}2", f"{word}nya"])
    has_emoji = contains_emoji(text)
    
    emoji_score = get_emoji_sentiment(text) * 0.3
    keyword_score = (has_positive - has_negative) * 0.4
    polarity_score = polarity * 0.3
    
    final_score = emoji_score + keyword_score + polarity_score
    
    if final_score > 0.2:
        return 'Positif'
    elif final_score < -0.2:
        return 'Negatif'
    else:
        return 'Netral'

def generate_visualizations():
    try:
        # Read data with proper encoding and error handling
        try:
            df = pd.read_csv('komentar_dengan_sentimen.csv', encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv('komentar_dengan_sentimen.csv', encoding='latin1')
        
        # Print initial data info
        print("\nInitial DataFrame Info:")
        print(df.info())
        print("\nInitial DataFrame Shape:", df.shape)
        print("\nFirst few rows:")
        print(df.head())
        
        # Clean the data
        # Remove any duplicate columns and keep only the first two columns
        df = df.iloc[:, :2]
        df.columns = ['komentar', 'sentimen']
        
        # Remove any rows with missing values
        df = df.dropna()
        
        # Print cleaned data info
        print("\nCleaned DataFrame Info:")
        print(df.info())
        print("\nCleaned DataFrame Shape:", df.shape)
        print("\nTotal rows after cleaning:", len(df))
        
        # Create images directory if it doesn't exist
        os.makedirs('static/images', exist_ok=True)
        
        # Count total comments and comments by sentiment
        total_comments = len(df)
        positive_comments = len(df[df['sentimen'] == 'Positif'])
        negative_comments = len(df[df['sentimen'] == 'Negatif'])
        neutral_comments = len(df[df['sentimen'] == 'Netral'])
        
        print(f"\nTotal comments: {total_comments}")
        print(f"Positive comments: {positive_comments}")
        print(f"Negative comments: {negative_comments}")
        print(f"Neutral comments: {neutral_comments}")
        
        # Generate pie chart
        plt.figure(figsize=(8, 6), facecolor='none')
        sentimen_counts = df['sentimen'].value_counts()
        plt.pie(sentimen_counts, labels=sentimen_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Distribusi Sentimen Komentar')
        plt.axis('equal')
        plt.savefig('static/images/pie_chart.png', transparent=True, bbox_inches='tight', dpi=100)
        plt.close()
        
        # Generate bar chart
        plt.figure(figsize=(8, 6), facecolor='none')
        sns.countplot(data=df, x='sentimen', order=['Positif', 'Netral', 'Negatif'])
        plt.title('Jumlah Komentar per Kategori Sentimen')
        plt.xlabel('Sentimen')
        plt.ylabel('Jumlah Komentar')
        plt.savefig('static/images/bar_chart.png', transparent=True, bbox_inches='tight', dpi=100)
        plt.close()
        
        # Generate word clouds
        for sentiment in ['Positif', 'Netral', 'Negatif']:
            text = ' '.join(df[df['sentimen'] == sentiment]['komentar'].dropna())
            wordcloud = WordCloud(width=400, height=300, background_color=None, mode='RGBA').generate(text)
            plt.figure(figsize=(4, 3), facecolor='none')
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.savefig(f'static/images/wordcloud_{sentiment.lower()}.png', transparent=True, bbox_inches='tight', dpi=100)
            plt.close()
        
        # Convert DataFrame to HTML table with styling
        comments_table = df.to_html(
            classes='table table-striped table-bordered table-hover',
            index=False,
            escape=False,
            justify='left',
            table_id='comments-table',
            render_links=True,
            border=0,
            formatters={
                'komentar': lambda x: f'<div style="max-width: 500px; word-wrap: break-word;">{x}</div>',
                'sentimen': lambda x: f'<div style="text-align: center;">{x}</div>'
            }
        )
        
        return {
            'total_comments': total_comments,
            'positive_comments': positive_comments,
            'negative_comments': negative_comments,
            'neutral_comments': neutral_comments,
            'comments_table': comments_table
        }
    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")
        print(f"Error details: {str(e.__class__.__name__)}")
        import traceback
        print(traceback.format_exc())
        return {
            'total_comments': 0,
            'positive_comments': 0,
            'negative_comments': 0,
            'neutral_comments': 0,
            'comments_table': ''
        }

@app.route('/')
def index():
    stats = generate_visualizations()
    return render_template('index.html', stats=stats)

@app.route('/analyze', methods=['POST'])
def analyze():
    comment = request.json.get('comment', '')
    if not comment:
        return jsonify({'error': 'No comment provided'}), 400
    
    sentiment = get_sentiment(comment)
    return jsonify({
        'sentiment': sentiment,
        'comment_length': len(comment),
        'word_count': len(comment.split()),
        'emoji_count': sum(1 for char in comment if char in ['😊', '👍', '💪', '🙏', '😂', '🤣', '😅', '😭', '😢', '😔', '😩', '😫', '😤', '😡', '❤️', '💕', '💗', '💓', '💝', '💖', '♥️', '😍', '🥰', '😘', '😗', '😚', '😙', '🤗', '🫂', '👏', '🙌', '✌️', '🤝', '🫡', '🤩', '🌟', '⭐', '✨', '💫', '🎉', '🎊'])
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 