"""
NLP-Analyse von Verbraucherbeschwerden (CFPB-Datensatz)

Zweck:
- Vorverarbeitung von Beschwerdetexten
- Vektorisierung mit TF-IDF und CountVectorizer (Bag-of-Words)
- Themenextraktion mit NMF (auf beiden Vektorisierungen) und LDA (auf Count)
- Vergleich der Vektorisierungen (Überschneidung der Top-Wörter)
- Berechnung der Themenanteile (dominantes Thema je Dokument)
- Vergleich verschiedener Themenanzahlen k anhand der UMass-Kohärenz

Eingabe:
- complaints_processed.csv (Standard: im Arbeitsverzeichnis, anpassbar mit --data)

Ausgabe:
- Konsolenausgabe mit Top-Wörtern, Themenanteilen und Vergleichen
- CSV-Dateien im Ordner results/

Aufruf:
    python nlp_analysis.py
    python nlp_analysis.py --data pfad/zu/complaints_processed.csv --sample 20000 --k 5

Autor: Niklas Lazar (IU14136580)
Kurs:  Projekt: Data Analysis (DLBDSEDA02_D)
"""

import argparse
import os
import re

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

RANDOM_STATE = 42
N_TOP_WORDS = 10
K_CANDIDATES = [5, 8, 10]
RESULTS_DIR = "results"
DATA_SOURCE = "https://github.com/halpert3/complaint-content-classification-nlp"


# ---------------------------------------------------------------------------
# Vorverarbeitung
# ---------------------------------------------------------------------------
def load_stopwords():
    """Lädt die englischen NLTK-Stoppwörter (Download beim ersten Lauf)."""
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))
    # CFPB ersetzt persönliche Daten durch "XXXX"; diese Platzhalter sind keine Inhalte
    stop_words.update(["xxxx", "xxx", "xx"])
    return stop_words


def clean_text(text, stop_words):
    """Kleinbuchstaben, XXXX-Platzhalter, Zahlen/Sonderzeichen, Stoppwörter und Kurzwörter entfernen."""
    text = str(text).lower()
    text = re.sub(r"x{2,}", " ", text)          # Anonymisierungs-Platzhalter
    text = re.sub(r"[^a-z\s]", " ", text)       # Zahlen und Sonderzeichen
    tokens = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Auswertung der Themenmodelle
# ---------------------------------------------------------------------------
def top_word_lists(model, feature_names, n_top=N_TOP_WORDS):
    """Gibt pro Thema die n wichtigsten Wörter als Liste zurück."""
    return [[feature_names[i] for i in comp.argsort()[::-1][:n_top]] for comp in model.components_]


def top_words_table(word_lists):
    """Top-Wörter als DataFrame (Themen ab 1 nummeriert)."""
    return pd.DataFrame({
        "topic": range(1, len(word_lists) + 1),
        "top_words": [", ".join(words) for words in word_lists],
    })


def topic_shares(model, matrix):
    """Anteil der Dokumente (in %), deren dominantes Thema das jeweilige Thema ist (Themen ab 1)."""
    doc_topics = model.transform(matrix)
    dominant = np.argmax(doc_topics, axis=1) + 1
    shares = (pd.Series(dominant).value_counts(normalize=True) * 100)
    # Themen ohne Dokumente mit 0 % aufführen
    return shares.reindex(range(1, model.n_components + 1), fill_value=0.0)


def umass_coherence(word_lists, binary_doc_term, vocabulary):
    """
    Mittlere UMass-Kohärenz (Mimno et al., 2011) über alle Themen.
    Höhere (weniger negative) Werte bedeuten, dass die Top-Wörter eines Themas
    häufiger gemeinsam in denselben Dokumenten vorkommen.
    """
    doc_freq = np.asarray(binary_doc_term.sum(axis=0)).ravel()
    scores = []
    for words in word_lists:
        idx = [vocabulary[w] for w in words if w in vocabulary]
        score, pairs = 0.0, 0
        for i in range(1, len(idx)):
            for j in range(i):
                co = binary_doc_term[:, idx[i]].multiply(binary_doc_term[:, idx[j]]).sum()
                score += np.log((co + 1) / doc_freq[idx[j]])
                pairs += 1
        scores.append(score / pairs if pairs else np.nan)
    return float(np.nanmean(scores))


def compare_topic_sets(lists_a, lists_b):
    """Ordnet jedem Thema aus A das ähnlichste Thema aus B zu (Jaccard der Top-Wörter)."""
    rows = []
    for i, a in enumerate(lists_a, start=1):
        sims = [len(set(a) & set(b)) / len(set(a) | set(b)) for b in lists_b]
        j = int(np.argmax(sims))
        rows.append({"topic_tfidf": i, "best_match_count": j + 1, "jaccard": round(sims[j], 2),
                     "shared_words": ", ".join(sorted(set(a) & set(lists_b[j])))})
    return pd.DataFrame(rows)


def save_shares(shares, name):
    pd.DataFrame({"topic": shares.index, "share_percent": shares.values.round(1)}).to_csv(
        os.path.join(RESULTS_DIR, f"topic_shares_{name}.csv"), index=False)


def print_model(title, word_lists, shares):
    print(f"\n--- {title} ---")
    for t, words in enumerate(word_lists, start=1):
        print(f"Thema {t} ({shares[t]:.1f} %): {', '.join(words)}")


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Themenextraktion aus CFPB-Beschwerdetexten")
    parser.add_argument("--data", default="complaints_processed.csv", help="Pfad zur CSV-Datei")
    parser.add_argument("--sample", type=int, default=20000, help="Größe der Zufallsstichprobe")
    parser.add_argument("--k", type=int, default=5, help="Anzahl der Themen für die Hauptmodelle")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Datei '{args.data}' nicht gefunden.")
        print(f"Bitte project_data/complaints_processed.csv aus {DATA_SOURCE} herunterladen "
              "und in diesen Ordner legen (oder den Pfad mit --data angeben).")
        return
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Daten laden
    df = pd.read_csv(args.data).dropna(subset=["narrative"])
    print(f"Datensatz: {len(df)} Beschwerden, Spalten: {[c for c in df.columns if not c.startswith('Unnamed')]}")
    print("Verteilung nach Produktkategorie (%):")
    print((df["product"].value_counts(normalize=True) * 100).round(1).to_string())

    sample = df.sample(n=min(args.sample, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"\nZufallsstichprobe: {len(sample)} Dokumente")

    # 2. Bereinigung
    stop_words = load_stopwords()
    sample["clean"] = sample["narrative"].apply(lambda t: clean_text(t, stop_words))
    sample = sample[sample["clean"].str.len() > 0].reset_index(drop=True)
    print(f"Nach Bereinigung: {len(sample)} Dokumente, "
          f"im Mittel {sample['clean'].str.split().str.len().mean():.0f} Wörter je Dokument")

    # 3. Vektorisierung (gleiche Parameter für einen fairen Vergleich)
    params = dict(max_features=5000, min_df=5, max_df=0.8, ngram_range=(1, 2))
    tfidf = TfidfVectorizer(**params)
    count_vec = CountVectorizer(**params)
    tfidf_matrix = tfidf.fit_transform(sample["clean"])
    count_matrix = count_vec.fit_transform(sample["clean"])
    print(f"\nTF-IDF-Matrix: {tfidf_matrix.shape}, Count-Matrix: {count_matrix.shape}")

    tfidf_words = tfidf.get_feature_names_out()
    count_words = count_vec.get_feature_names_out()
    binary = (count_matrix > 0).astype(np.int32).tocsc()
    vocab = count_vec.vocabulary_

    # 4. Themenmodelle
    nmf_tfidf = NMF(n_components=args.k, init="nndsvd", random_state=RANDOM_STATE, max_iter=400).fit(tfidf_matrix)
    nmf_count = NMF(n_components=args.k, init="nndsvd", random_state=RANDOM_STATE, max_iter=400).fit(count_matrix)
    lda = LatentDirichletAllocation(n_components=args.k, random_state=RANDOM_STATE, max_iter=10).fit(count_matrix)

    models = {
        "nmf_tfidf": ("NMF auf TF-IDF", nmf_tfidf, tfidf_matrix, tfidf_words),
        "nmf_count": ("NMF auf Count", nmf_count, count_matrix, count_words),
        "lda_count": ("LDA auf Count", lda, count_matrix, count_words),
    }
    word_lists, summary = {}, []
    for name, (title, model, matrix, words) in models.items():
        lists = top_word_lists(model, words)
        shares = topic_shares(model, matrix)
        word_lists[name] = lists
        print_model(title, lists, shares)
        top_words_table(lists).assign(share_percent=shares.values.round(1)).to_csv(
            os.path.join(RESULTS_DIR, f"top_words_{name}.csv"), index=False)
        save_shares(shares, name)
        summary.append({"model": title, "umass_coherence": round(umass_coherence(lists, binary, vocab), 3)})

    # 5. Vergleich der Vektorisierungen (NMF auf TF-IDF vs. NMF auf Count)
    comparison = compare_topic_sets(word_lists["nmf_tfidf"], word_lists["nmf_count"])
    comparison.to_csv(os.path.join(RESULTS_DIR, "vectorizer_comparison.csv"), index=False)
    print("\n--- Vergleich TF-IDF vs. Count (NMF, Jaccard der Top-10-Wörter) ---")
    print(comparison.to_string(index=False))
    print(f"Mittlere Überschneidung: {comparison['jaccard'].mean():.2f}")

    pd.DataFrame(summary).to_csv(os.path.join(RESULTS_DIR, "model_coherence.csv"), index=False)
    print("\n--- UMass-Kohärenz der Hauptmodelle (höher = besser) ---")
    print(pd.DataFrame(summary).to_string(index=False))

    # 6. Wahl der Themenanzahl: NMF auf TF-IDF mit verschiedenen k
    print("\n--- Vergleich der Themenanzahl k (NMF auf TF-IDF) ---")
    k_rows = []
    for k in K_CANDIDATES:
        model = NMF(n_components=k, init="nndsvd", random_state=RANDOM_STATE, max_iter=400).fit(tfidf_matrix)
        lists = top_word_lists(model, tfidf_words)
        k_rows.append({"k": k, "umass_coherence": round(umass_coherence(lists, binary, vocab), 3),
                       "reconstruction_error": round(model.reconstruction_err_, 2)})
    k_table = pd.DataFrame(k_rows)
    k_table.to_csv(os.path.join(RESULTS_DIR, "k_comparison.csv"), index=False)
    print(k_table.to_string(index=False))

    print(f"\nAlle Ergebnisse wurden im Ordner '{RESULTS_DIR}/' gespeichert.")


if __name__ == "__main__":
    main()
