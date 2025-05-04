import pandas as pd
from textblob import TextBlob
import os
import re
import logging

# Dapatkan path absolut
current_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(current_dir, 'komentar_grandtheftauto6_clean.csv')

print(f"Membaca file dari: {input_file}")

# Baca file awal dengan encoding yang benar
df = pd.read_csv(input_file, encoding='utf-8')

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
    neutral_emoji = ['😂', '🤣', '😅']  # Tawa bisa ambigu
    
    pos_count = sum(text.count(emoji) for emoji in positive_emoji)
    neg_count = sum(text.count(emoji) for emoji in negative_emoji)
    
    if pos_count > neg_count:
        return 1
    elif neg_count > pos_count:
        return -1
    return 0

def get_sentiment(text):
    # Setup logging
    logging.basicConfig(filename='sentiment_analysis.log', level=logging.INFO)
    
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    text_lower = text.lower()
    
    # Kata kunci positif yang lebih lengkap dan relevan dengan konteks
    positive_keywords = [
        'worth the wait', 'perfect', 'masterpiece', 'patiently waiting',
        'please port', 'release', 'coming', 'trailer', 'launch',
        'best way', 'hurry up', 'please fix', 'thank you',
        'waiting', 'support', 'good', 'great', 'amazing',
        'hype', 'excited', 'cant wait', 'awesome',
        'finally', 'lets go', 'nice', 'cool'
    ]
    
    # Kata kunci negatif yang lebih lengkap
    negative_keywords = [
        'impossible', 'terrible', 'boring', 'stupid', 'fake',
        'glitch', 'fix', 'delay', 'tired', 'bored',
        'forever', 'taking', 'where', 'when', 'no',
        'dont', 'stop', 'never', 'bad', 'worse',
        'worst', 'hate', 'trash', 'garbage', 'waste',
        'annoying', 'broken', 'bug', 'problem', 'issue'
    ]

    # Cek kata kunci dan emoji
    has_negative = any(keyword in text_lower for keyword in negative_keywords)
    has_positive = any(keyword in text_lower for word in positive_keywords for keyword in [word, f"{word}2", f"{word}nya"])
    has_emoji = contains_emoji(text)
    
    # Hitung fitur tambahan
    repeated_chars = count_repeated_chars(text)
    excessive_punctuation = count_punctuation(text)
    word_count = len(text_lower.split())
    
    # Tambahkan bobot untuk berbagai faktor
    emoji_score = get_emoji_sentiment(text) * 0.3
    keyword_score = (has_positive - has_negative) * 0.4
    polarity_score = polarity * 0.3
    
    final_score = emoji_score + keyword_score + polarity_score
    
    # Tambahkan penanganan untuk kasus-kasus khusus
    special_cases = {
        'wkwk': 'Positif',
        'haha': 'Positif',
        'hmm': 'Netral',
        'oh': 'Netral',
        'oke': 'Positif',
        'ok': 'Positif',
        'y': 'Netral',
        'no': 'Negatif'
    }
    
    # Logging scores
    logging.info(f"Processing text: {text}")
    logging.info(f"Scores - Emoji: {emoji_score}, Keyword: {keyword_score}, Polarity: {polarity_score}")
    
    if final_score > 0.2:
        return 'Positif'
    elif final_score < -0.2:
        return 'Negatif'
    elif word_count <= 2 or (not has_positive and not has_negative):
        return 'Netral'
    else:
        if has_positive and not has_negative:
            return 'Positif'
        elif has_negative and not has_positive:
            return 'Negatif'
        return 'Netral'

def validate_text(text):
    if not isinstance(text, str):
        return ''
    text = text.strip()
    if len(text) == 0:
        return ''
    return text

print("Memproses sentimen...")
df['sentimen'] = df['komentar'].apply(get_sentiment)

print("Menyimpan hasil...")
output_file = os.path.join(current_dir, 'komentar_dengan_sentimen.csv')
df.to_csv(output_file, index=False, encoding='utf-8')

# Tampilkan statistik
sentimen_counts = df['sentimen'].value_counts()
print("\nDistribusi Sentimen:")
print(sentimen_counts)
print(f"\nPersentase:")
print((sentimen_counts / len(df) * 100).round(2), "%")

# Tampilkan contoh dari setiap kategori
for sentiment in ['Positif', 'Negatif', 'Netral']:
    print(f"\nContoh komentar {sentiment.lower()}:")
    examples = df[df['sentimen'] == sentiment]['komentar'].head()
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example}")

print("\nFile berhasil disimpan!")