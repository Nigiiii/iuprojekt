"""
NLP-Analyse von Verbraucherbeschwerden (CFPB-Datensatz)

Zweck:
- Vorverarbeitung von Beschwerdetexten
- Vektorisierung mit TF-IDF und CountVectorizer (Bag-of-Words)
- Themenextraktion mit NMF (auf beiden Vektorisierungen) und LDA (auf Count)
- Vergleich der Vektorisierungen (Überschneidung der Top-Wörter)
- Berechnung der Themenanteile (dominantes Thema je Dokument)
- Systematische Wahl der Themenanzahl k (k = 2 bis 15) anhand von
  C_v-Kohärenz, Themenvielfalt und Stabilität über Teilstichproben

Eingabe:
- complaints_processed.csv (Standard: im Arbeitsverzeichnis, anpassbar mit --data)

Ausgabe:
- Konsolenausgabe mit Top-Wörtern, Themenanteilen und Vergleichen
- CSV-Dateien und das Diagramm k_selection.png im Ordner results/

Aufruf:
    python nlp_analysis.py                      # k wird automatisch bestimmt
    python nlp_analysis.py --k 6                # feste Themenanzahl, ohne k-Auswahl
    python nlp_analysis.py --data pfad/zu/complaints_processed.csv --sample 20000

Autor: Niklas Lazar (IU14136580)
Kurs:  Projekt: Data Analysis (DLBDSEDA02_D)
"""

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")  # Diagramme ohne Bildschirm speichern
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from nltk.corpus import stopwords
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

RANDOM_STATE = 42
N_TOP_WORDS = 10
K_RANGE = range(2, 16)            # getestete Themenanzahlen k = 2 ... 15
STABILITY_SEEDS = list(range(1, 11))  # 10 Zufallsstartwerte für die Teilstichproben
STABILITY_FRACTION = 0.8          # Größe einer Teilstichprobe (80 % der Dokumente)
SCORE_TOLERANCE = 0.03            # k-Werte mit Gesamtwert <= 0,03 unter dem besten gelten als gleichwertig
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


# ---------------------------------------------------------------------------
# Wahl der Themenanzahl k
# ---------------------------------------------------------------------------
def fit_nmf(matrix, k):
    """NMF mit deterministischer Initialisierung (nndsvd), damit Ergebnisse reproduzierbar sind."""
    return NMF(n_components=k, init="nndsvd", random_state=RANDOM_STATE, max_iter=400).fit(matrix)


def cv_texts(clean_texts):
    """
    Token-Folgen für die C_v-Kohärenz. Da die Modelle auch Bigramme nutzen
    (z. B. "credit report"), wird nach jedem Wort das Bigramm mit dem Folgewort
    eingefügt: a, "a b", b, "b c", c, ...
    """
    texts = []
    for text in clean_texts:
        tokens = text.split()
        seq = []
        for i, tok in enumerate(tokens):
            seq.append(tok)
            if i + 1 < len(tokens):
                seq.append(f"{tok} {tokens[i + 1]}")
        texts.append(seq)
    return texts


def cv_per_topic(topic_lists, texts, dictionary):
    """C_v-Kohärenz (Röder et al., 2015) für jede Wortliste in topic_lists."""
    cm = CoherenceModel(topics=topic_lists, texts=texts, dictionary=dictionary,
                        coherence="c_v", topn=N_TOP_WORDS,
                        window_size=220,  # entspricht 110 Wörtern, da jedes 2. Token ein Bigramm ist
                        processes=1)
    return cm.get_coherence_per_topic()


def topic_diversity(word_lists):
    """Anteil verschiedener Wörter unter den Top-Wörtern aller Themen (1 = keine Überschneidung)."""
    all_words = [w for words in word_lists for w in words]
    return len(set(all_words)) / len(all_words)


def topic_stability(ref_lists, matrix, k, words, n_docs):
    """
    Stabilität nach Greene et al. (2014): NMF wird auf mehreren zufälligen
    Teilstichproben neu trainiert; die Themen werden optimal den Referenzthemen
    zugeordnet (ungarische Methode) und die mittlere Jaccard-Übereinstimmung der
    Top-Wörter berechnet. 1 = auf allen Teilstichproben identische Themen.
    """
    scores = []
    for seed in STABILITY_SEEDS:
        rng = np.random.default_rng(seed)
        rows = rng.choice(n_docs, size=int(STABILITY_FRACTION * n_docs), replace=False)
        lists = top_word_lists(fit_nmf(matrix[rows], k), words)
        sim = np.array([[len(set(a) & set(b)) / len(set(a) | set(b)) for b in lists] for a in ref_lists])
        r, c = linear_sum_assignment(-sim)
        scores.append(sim[r, c].mean())
    return float(np.mean(scores))


def select_k(tfidf_matrix, tfidf_words, clean_texts, binary, vocab):
    """
    Testet k = 2 ... 15 für NMF auf TF-IDF und bewertet jedes k mit drei Kriterien:
    C_v-Kohärenz (Interpretierbarkeit), Themenvielfalt (keine doppelten Themen)
    und Stabilität (Reproduzierbarkeit). Die Kriterien werden auf 0-1 normiert und
    zu einem Gesamtwert gemittelt.

    Entscheidungsregel: Alle k, deren Gesamtwert höchstens SCORE_TOLERANCE unter dem
    besten Wert liegt, gelten als gleichwertige Kandidaten. Unter ihnen wird das k
    mit der höchsten C_v-Kohärenz gewählt, weil die Themen für Entscheidungstragende
    vor allem gut interpretierbar sein sollen.
    """
    n_docs = tfidf_matrix.shape[0]
    models, lists_by_k = {}, {}
    for k in K_RANGE:
        models[k] = fit_nmf(tfidf_matrix, k)
        lists_by_k[k] = top_word_lists(models[k], tfidf_words)
        print(f"  k = {k:2d}: Modell trainiert")

    # C_v für alle Themen aller k in einem Durchlauf berechnen (spart Rechenzeit)
    texts = cv_texts(clean_texts)
    dictionary = Dictionary(texts)
    all_lists = [words for k in K_RANGE for words in lists_by_k[k]]
    cv_all = cv_per_topic(all_lists, texts, dictionary)

    rows, pos = [], 0
    for k in K_RANGE:
        cv_k = float(np.mean(cv_all[pos:pos + k]))
        pos += k
        rows.append({
            "k": k,
            "cv_coherence": round(cv_k, 4),
            "diversity": round(topic_diversity(lists_by_k[k]), 4),
            "stability": round(topic_stability(lists_by_k[k], tfidf_matrix, k, tfidf_words, n_docs), 4),
            "umass_coherence": round(umass_coherence(lists_by_k[k], binary, vocab), 4),
            "reconstruction_error": round(models[k].reconstruction_err_, 2),
        })
        print(f"  k = {k:2d}: bewertet")
    table = pd.DataFrame(rows)

    criteria = ["cv_coherence", "diversity", "stability"]
    for col in criteria:
        lo, hi = table[col].min(), table[col].max()
        table[f"{col}_norm"] = (table[col] - lo) / (hi - lo) if hi > lo else 1.0
    table["combined_score"] = table[[f"{c}_norm" for c in criteria]].mean(axis=1).round(4)
    table["candidate"] = table["combined_score"] >= table["combined_score"].max() - SCORE_TOLERANCE
    candidates = table[table["candidate"]]
    best_k = int(candidates.loc[candidates["cv_coherence"].idxmax(), "k"])

    # Top-Wörter aller k speichern, damit die Themen inhaltlich verglichen werden können
    pd.DataFrame([{"k": k, "topic": t, "top_words": ", ".join(words)}
                  for k in K_RANGE for t, words in enumerate(lists_by_k[k], start=1)]).to_csv(
        os.path.join(RESULTS_DIR, "top_words_by_k.csv"), index=False)
    return best_k, table


def plot_k_selection(table, best_k, path):
    """Drei Kriterien plus Gesamtwert in getrennten Feldern (eine y-Achse je Kennzahl)."""
    panels = [("cv_coherence", "C_v-Kohärenz"), ("diversity", "Themenvielfalt"),
              ("stability", "Stabilität"), ("combined_score", "Gesamtwert (normiert)")]
    color, ink, muted = "#2a78d6", "#0b0b0b", "#8a8984"
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.4), sharex=True)
    axes = axes.ravel()
    for ax, (col, title) in zip(axes, panels):
        if col == "combined_score":  # Toleranzband: gleichwertige Kandidaten
            top = table[col].max()
            ax.axhspan(top - SCORE_TOLERANCE, top, color="#e6e5e0", zorder=0)
        ax.plot(table["k"], table[col], color=color, linewidth=2, marker="o", markersize=4)
        ax.axvline(best_k, color=muted, linewidth=1, linestyle="--")
        best_val = table.loc[table["k"] == best_k, col].iloc[0]
        ax.plot([best_k], [best_val], marker="o", markersize=8, color=color,
                markeredgecolor="white", markeredgewidth=2)
        ax.set_title(title, fontsize=11, color=ink, loc="left")
        ax.tick_params(labelsize=9, colors=ink, labelbottom=True)
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(muted)
    axes[0].set_xticks(list(table["k"])[::2])
    for ax in axes[2:]:
        ax.set_xlabel("Anzahl Themen k", fontsize=10, color=ink)
    axes[3].text(table["k"].max(), table["combined_score"].max() + 0.005, "gleichwertige Kandidaten",
                 fontsize=8, color=muted, ha="right", va="bottom")
    fig.suptitle(f"Wahl der Themenanzahl (NMF auf TF-IDF): gewählt k = {best_k}",
                 fontsize=12, x=0.01, ha="left", color=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


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
    parser.add_argument("--k", type=int, default=None,
                        help="Feste Themenanzahl (ohne Angabe wird k automatisch bestimmt)")
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

    # 4. Wahl der Themenanzahl k
    if args.k is None:
        print("\n--- Wahl der Themenanzahl k (NMF auf TF-IDF, k = 2 bis 15) ---")
        k, k_table = select_k(tfidf_matrix, tfidf_words, sample["clean"], binary, vocab)
        k_table.to_csv(os.path.join(RESULTS_DIR, "k_selection.csv"), index=False)
        plot_k_selection(k_table, k, os.path.join(RESULTS_DIR, "k_selection.png"))
        print(k_table[["k", "cv_coherence", "diversity", "stability", "combined_score", "candidate"]]
              .to_string(index=False))
        print(f"Gleichwertige Kandidaten: k = {list(k_table.loc[k_table['candidate'], 'k'])}")
        print(f"Gewählte Themenanzahl (höchste C_v-Kohärenz unter den Kandidaten): k = {k}")
    else:
        k = args.k
        print(f"\nFeste Themenanzahl: k = {k}")

    # 5. Themenmodelle mit dem gewählten k
    nmf_tfidf = fit_nmf(tfidf_matrix, k)
    nmf_count = fit_nmf(count_matrix, k)
    lda = LatentDirichletAllocation(n_components=k, random_state=RANDOM_STATE, max_iter=10).fit(count_matrix)

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

    # C_v-Kohärenz der drei Hauptmodelle (gleiches Maß wie bei der k-Auswahl)
    texts = cv_texts(sample["clean"])
    cv_scores = cv_per_topic([w for lists in word_lists.values() for w in lists], texts, Dictionary(texts))
    for i, row in enumerate(summary):
        row["cv_coherence"] = round(float(np.mean(cv_scores[i * k:(i + 1) * k])), 3)

    # 6. Vergleich der Vektorisierungen (NMF auf TF-IDF vs. NMF auf Count)
    comparison = compare_topic_sets(word_lists["nmf_tfidf"], word_lists["nmf_count"])
    comparison.to_csv(os.path.join(RESULTS_DIR, "vectorizer_comparison.csv"), index=False)
    print("\n--- Vergleich TF-IDF vs. Count (NMF, Jaccard der Top-10-Wörter) ---")
    print(comparison.to_string(index=False))
    print(f"Mittlere Überschneidung: {comparison['jaccard'].mean():.2f}")

    pd.DataFrame(summary).to_csv(os.path.join(RESULTS_DIR, "model_coherence.csv"), index=False)
    print("\n--- Kohärenz der Hauptmodelle (höher = besser) ---")
    print(pd.DataFrame(summary).to_string(index=False))

    print(f"\nAlle Ergebnisse wurden im Ordner '{RESULTS_DIR}/' gespeichert.")


if __name__ == "__main__":
    main()
