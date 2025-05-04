import pandas as pd
import re
import string
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Load data
df = pd.read_csv("komentar_200_baru.csv")

# Preprocessing functions
stop_factory = StopWordRemoverFactory()
stem_factory = StemmerFactory()
stop_remover = stop_factory.create_stop_word_remover()
stemmer = stem_factory.create_stemmer()

def clean_text(text):
    # Hapus kata-kata yang tidak diinginkan
    unwanted_words = [
        'likesReply', 'See translation', 'Reply', 'View replies', 'likes', 
        'w', 'wReply', 'Reply', 'View replies', 'fathianpujakesuma',
        'like', 'reply', 'view', 'replies', 'translation', 'see',
        'likesReply', 'wReply', 'View replies', 'See translation'
    ]
    
    # Hapus tanda kutip di awal dan akhir
    text = text.strip('"\'')
    text = text.lower()
    
    # Hapus username dan format khusus
    text = re.sub(r'[a-zA-Z0-9_]+,[a-zA-Z0-9_]+', '', text)  # Hapus pasangan username dengan koma
    text = re.sub(r'([a-zA-Z0-9_]+)[,.]?\s*\1', '', text)  # Hapus username yang berulang
    text = re.sub(r'[a-zA-Z0-9_]+_[a-zA-Z0-9_]+', '', text)  # Hapus username dengan underscore
    text = re.sub(r'\b[a-z0-9_]+\b,', '', text)  # Hapus username yang diikuti koma
    text = re.sub(r',\s*[a-z0-9_]+\b', '', text)  # Hapus username yang didahului koma
    text = re.sub(r'\b[a-z0-9_]+\.[a-z0-9_]+\b', '', text)  # Hapus username dengan titik
    text = re.sub(r'\b[a-z0-9_]+\b(?=\s*[,."]|$)', '', text)  # Hapus username di akhir atau sebelum tanda baca
    
    # Hapus teks berwarna biru dan format khusus
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)  # Hapus mentions
    text = re.sub(r'#\w+', '', text)  # Hapus hashtag
    text = re.sub(r'http\S+', '', text)  # Hapus URL
    text = re.sub(r'www\.\S+', '', text)  # Hapus www
    text = re.sub(r'instagram\.com/\S+', '', text)  # Hapus link Instagram
    text = re.sub(r'\([^)]*\)', '', text)  # Hapus teks dalam kurung
    text = re.sub(r'\[[^\]]*\]', '', text)  # Hapus teks dalam kurung siku
    text = re.sub(r'\{[^}]*\}', '', text)  # Hapus teks dalam kurung kurawal
    text = re.sub(r'<[^>]+>', '', text)  # Hapus tag HTML
    text = re.sub(r'\d+ likes?', '', text)  # Hapus "X likes"
    text = re.sub(r'\d+ reply', '', text)  # Hapus "X reply"
    text = re.sub(r'\d+ replies', '', text)  # Hapus "X replies"
    text = re.sub(r'View \d+ replies?', '', text)  # Hapus "View X replies"
    
    # Hapus punctuation
    text = re.sub(f"[{string.punctuation}]", " ", text)
    # Hapus numbers
    text = re.sub(r'\d+', '', text)
    
    # Hapus kata-kata yang tidak diinginkan
    for word in unwanted_words:
        text = text.replace(word.lower(), '')
    
    # Hapus spasi berlebih
    text = ' '.join(text.split())
    
    # Hapus stopwords
    text = stop_remover.remove(text)
    # Stemming
    text = stemmer.stem(text)
    
    # Hapus baris yang hanya berisi username atau teks pendek
    if len(text.strip()) < 5:  # Jika teks terlalu pendek, dianggap tidak valid
        return ""
    
    return text

# Terapkan preprocessing
df["cleaned"] = df["komentar"].astype(str).apply(clean_text)

# Hapus baris yang kosong atau terlalu pendek setelah cleaning
df = df[df["cleaned"].str.strip() != ""]
df = df[df["cleaned"].str.len() >= 5]

# Hapus duplikat
df = df.drop_duplicates(subset=['cleaned'])

# Hapus baris yang hanya berisi username
df = df[~df["cleaned"].str.match(r'^[a-z0-9_]+$')]

# Simpan hasil
df.to_csv("komentar_grandtheftauto6_clean.csv", index=False)
print("Preprocessing selesai.")
print(f"Total komentar setelah cleaning: {len(df)}")
