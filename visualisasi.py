import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

# Set style untuk visualisasi
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Baca data
current_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(current_dir, 'komentar_dengan_sentimen.csv')
df = pd.read_csv(input_file, encoding='utf-8')

# 1. Distribusi Sentimen (Pie Chart)
plt.figure(figsize=(10, 6), facecolor='none')
sentimen_counts = df['sentimen'].value_counts()
plt.pie(sentimen_counts, labels=sentimen_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribusi Sentimen Komentar')
plt.axis('equal')
plt.savefig('distribusi_sentimen.png', transparent=True)
plt.close()

# 2. Distribusi Sentimen (Bar Chart)
plt.figure(figsize=(10, 6), facecolor='none')
sns.countplot(data=df, x='sentimen', order=['Positif', 'Netral', 'Negatif'])
plt.title('Jumlah Komentar per Kategori Sentimen')
plt.xlabel('Sentimen')
plt.ylabel('Jumlah Komentar')
plt.savefig('jumlah_sentimen.png', transparent=True)
plt.close()

# 3. Word Cloud untuk setiap kategori sentimen
def create_wordcloud(text, title, filename):
    wordcloud = WordCloud(width=800, height=400, background_color=None, mode='RGBA').generate(text)
    plt.figure(figsize=(10, 6), facecolor='none')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.savefig(filename, transparent=True)
    plt.close()

# Buat word cloud untuk setiap kategori sentimen
for sentiment in ['Positif', 'Netral', 'Negatif']:
    text = ' '.join(df[df['sentimen'] == sentiment]['komentar'].dropna())
    create_wordcloud(text, f'Word Cloud - {sentiment}', f'wordcloud_{sentiment.lower()}.png')

# 4. Distribusi Panjang Komentar per Sentimen
plt.figure(figsize=(12, 6), facecolor='none')
sns.boxplot(data=df, x='sentimen', y='panjang_komentar', order=['Positif', 'Netral', 'Negatif'])
plt.title('Distribusi Panjang Komentar per Kategori Sentimen')
plt.xlabel('Sentimen')
plt.ylabel('Panjang Komentar (karakter)')
plt.savefig('panjang_komentar.png', transparent=True)
plt.close()

# 5. Heatmap Korelasi Fitur
# Hitung beberapa fitur tambahan
df['jumlah_kata'] = df['komentar'].str.split().str.len()
df['jumlah_emoji'] = df['komentar'].apply(lambda x: sum(1 for char in str(x) if char in ['😊', '👍', '💪', '🙏', '😂', '🤣', '😅', '😭', '😢', '😔', '😩', '😫', '😤', '😡', '❤️', '💕', '💗', '💓', '💝', '💖', '♥️', '😍', '🥰', '😘', '😗', '😚', '😙', '🤗', '🫂', '👏', '🙌', '✌️', '🤝', '🫡', '🤩', '🌟', '⭐', '✨', '💫', '🎉', '🎊']))
df['jumlah_tanda_baca'] = df['komentar'].str.count(r'[!?]{2,}|\.{2,}')

# Buat heatmap
plt.figure(figsize=(10, 8), facecolor='none')
numeric_df = df[['panjang_komentar', 'jumlah_kata', 'jumlah_emoji', 'jumlah_tanda_baca']]
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Korelasi antara Fitur Komentar')
plt.savefig('korelasi_fitur.png', transparent=True)
plt.close()

print("\nVisualisasi telah disimpan dalam bentuk file gambar:")
print("- distribusi_sentimen.png - Pie chart distribusi sentimen")
print("- jumlah_sentimen.png - Bar chart jumlah komentar per sentimen")
print("- wordcloud_positif.png - Word cloud komentar positif")
print("- wordcloud_netral.png - Word cloud komentar netral")
print("- wordcloud_negatif.png - Word cloud komentar negatif")
print("- panjang_komentar.png - Box plot panjang komentar per sentimen")
print("- korelasi_fitur.png - Heatmap korelasi fitur komentar")