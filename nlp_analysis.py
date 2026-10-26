"""
NLP-Analyse von Verbraucherbeschwerden (CFPB-Datensatz)

Zweck:
- Vorverarbeitung von Beschwerdetexten
- Vektorisierung mit TF-IDF und CountVectorizer
- Themenextraktion mit NMF (auf beiden Vektorisierungen) und LDA
- Vergleich der Methoden und Berechnung der Themenanteile

Eingabe:
- complaints_processed.csv (im selben Verzeichnis oder relativer Pfad)

Ausgabe:
- Konsolenausgabe mit Top-Wörtern und Themenanteilen
- CSV-Dateien im Ordner results/ (top_words_*.csv, topic_shares_*.csv)

Autor: Niklas Lazar (IU14136580)
Datum: 2026-10-26
"""

import pandas as pd
import numpy as np
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import nltk
from nltk.corpus import stopwords

# NLTK-Daten laden
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Ergebnisse-Ordner anlegen
os.makedirs('results', exist_ok=True)

def clean_text(text, stop_words):
    """Bereinigt den Text: Kleinbuchstaben, Sonderzeichen, Stoppwörter, XXXX entfernen."""
    text = str(text).lower()
    text = re.sub(r'xxxx+', ' ', text)  # CFPB-Anonymisierung entfernen
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

def get_top_words(model, feature_names, n_top=10):
    """Extrahiert die Top-Wörter pro Thema."""
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top-1:-1]]
        topics.append({'topic': topic_idx + 1, 'top_words': ', '.join(top_words)})
    return pd.DataFrame(topics)

def topic_shares(model, matrix):
    """Berechnet den Anteil jedes Themas (in %)."""
    doc_topics = model.transform(matrix)
    dominant = np.argmax(doc_topics, axis=1)
    shares = pd.Series(dominant).value_counts(normalize=True).sort_index() * 100
    return shares

def main():
    print("Lade Daten...")
    # Relativer Pfad – Datei muss im Arbeitsverzeichnis liegen
    data_path = 'complaints_processed.csv'
    if not os.path.exists(data_path):
        print(f"Hinweis: {data_path} nicht gefunden. Bitte Datensatz herunterladen und hier ablegen.")
        print("Quelle: https://github.com/halpert3/complaint-content-classification-nlp")
        return
    
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['narrative'])
    print(f"Gesamtdatensatz: {len(df)} Beschwerden")
    print(f"Spalten: {list(df.columns)}")
    print(f"Produktverteilung:\n{df['product'].value_counts().head()}")
    
    # Stichprobe für Laufzeit
    df_sample = df.sample(n=min(20000, len(df)), random_state=42).reset_index(drop=True)
    print(f"\nStichprobe: {len(df_sample)} Dokumente")
    
    # Stoppwörter + XXXX
    stop_words = set(stopwords.words('english'))
    stop_words.update(['xxxx', 'xx', 'xxx'])
    
    print("Bereinige Texte...")
    df_sample['clean'] = df_sample['narrative'].apply(lambda x: clean_text(x, stop_words))
    
    # Vektorisierung
    print("\nVektorisierung...")
    tfidf = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.8, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df_sample['clean'])
    
    count_vec = CountVectorizer(max_features=5000, min_df=5, max_df=0.8, ngram_range=(1, 2))
    count_matrix = count_vec.fit_transform(df_sample['clean'])
    
    print(f"TF-IDF Shape: {tfidf_matrix.shape}")
    print(f"Count Shape:  {count_matrix.shape}")
    
    # NMF auf beiden (fairer Vergleich)
    print("\n--- NMF auf TF-IDF ---")
    nmf_tfidf = NMF(n_components=5, random_state=42, max_iter=200)
    nmf_tfidf.fit(tfidf_matrix)
    top_nmf_tfidf = get_top_words(nmf_tfidf, tfidf.get_feature_names_out())
    print(top_nmf_tfidf.to_string(index=False))
    shares_nmf_tfidf = topic_shares(nmf_tfidf, tfidf_matrix)
    print("\nThemenanteile NMF (TF-IDF) in %:")
    print(shares_nmf_tfidf.round(1))
    
    print("\n--- NMF auf Count ---")
    nmf_count = NMF(n_components=5, random_state=42, max_iter=200)
    nmf_count.fit(count_matrix)
    top_nmf_count = get_top_words(nmf_count, count_vec.get_feature_names_out())
    print(top_nmf_count.to_string(index=False))
    shares_nmf_count = topic_shares(nmf_count, count_matrix)
    print("\nThemenanteile NMF (Count) in %:")
    print(shares_nmf_count.round(1))
    
    # LDA auf Count
    print("\n--- LDA auf Count ---")
    lda = LatentDirichletAllocation(n_components=5, random_state=42, max_iter=10)
    lda.fit(count_matrix)
    top_lda = get_top_words(lda, count_vec.get_feature_names_out())
    print(top_lda.to_string(index=False))
    shares_lda = topic_shares(lda, count_matrix)
    print("\nThemenanteile LDA in %:")
    print(shares_lda.round(1))
    
    # Ergebnisse speichern
    top_nmf_tfidf.to_csv('results/top_words_nmf_tfidf.csv', index=False)
    top_nmf_count.to_csv('results/top_words_nmf_count.csv', index=False)
    top_lda.to_csv('results/top_words_lda.csv', index=False)
    pd.DataFrame({'topic': shares_nmf_tfidf.index + 1, 'share_percent': shares_nmf_tfidf.values}).to_csv(
        'results/topic_shares_nmf_tfidf.csv', index=False)
    print("\nErgebnisse gespeichert in Ordner results/")
    
    print("\nHinweis zur Themenanzahl: 5 Themen wurden gewählt, da sie die Hauptprobleme "
          "(Kreditberichte, Zahlungen, Inkasso, Identität, Konten) gut abdecken. "
          "Höhere k-Werte führen oft zu überlappenden oder schwer interpretierbaren Themen.")

if __name__ == "__main__":
    main()
