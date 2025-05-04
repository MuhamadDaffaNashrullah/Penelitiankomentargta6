import pandas as pd
import os

# Baca file CSV
df = pd.read_csv('komentar_dengan_sentimen.csv')

# Hapus kolom 'cleaned'
if 'cleaned' in df.columns:
    df = df.drop('cleaned', axis=1)

# Simpan kembali ke file yang sama
df.to_csv('komentar_dengan_sentimen.csv', index=False)

print("Kolom 'cleaned' berhasil dihapus!") 