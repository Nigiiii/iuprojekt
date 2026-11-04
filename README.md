# NLP-Analyse von Verbraucherbeschwerden (CFPB)

Themenextraktion aus Beschwerdetexten mit TF-IDF, CountVectorizer, NMF und LDA.

**Autor:** Niklas Lazar (Matrikelnummer: IU14136580)
**Kurs:** Projekt: Data Analysis (DLBDSEDA02_D), IU Internationale Hochschule

## Daten

Verwendet wird `project_data/complaints_processed.csv` aus dem Repository
[halpert3/complaint-content-classification-nlp](https://github.com/halpert3/complaint-content-classification-nlp)
(ca. 162.000 englischsprachige Beschwerden aus der CFPB Consumer Complaint Database, Spalten `product` und `narrative`).

Die Datei ist ca. 100 MB groß und deshalb nicht in diesem Repository enthalten.
Herunterladen und in den Projektordner legen:

```
git clone --depth 1 https://github.com/halpert3/complaint-content-classification-nlp.git
copy complaint-content-classification-nlp\project_data\complaints_processed.csv .     (Windows)
cp complaint-content-classification-nlp/project_data/complaints_processed.csv .        (macOS/Linux)
```

Alternativ auf der GitHub-Seite die Datei `project_data/complaints_processed.csv` öffnen und über „Download raw file“ speichern.

## Installation

Python 3.10 oder neuer (getestet mit Python 3.13).

**Windows**
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux**
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Beim ersten Start lädt das Skript die NLTK-Stoppwortliste automatisch herunter (Internetverbindung nötig).

## Nutzung

```
python nlp_analysis.py
```

Optionale Parameter:

| Parameter | Standard | Bedeutung |
|---|---|---|
| `--data` | `complaints_processed.csv` | Pfad zur CSV-Datei |
| `--sample` | `20000` | Größe der Zufallsstichprobe |
| `--k` | `5` | Anzahl der Themen |

Beispiel: `python nlp_analysis.py --sample 50000 --k 8`

Laufzeit mit den Standardwerten: ca. 1 Minute.

## Ablauf

1. Daten laden, leere Texte entfernen, Zufallsstichprobe ziehen (`random_state=42`)
2. Bereinigung: Kleinbuchstaben, CFPB-Platzhalter „XXXX“, Zahlen, Sonderzeichen und englische Stoppwörter entfernen
3. Vektorisierung mit TF-IDF und CountVectorizer (gleiche Parameter: 5.000 Merkmale, Uni- und Bigramme)
4. Themenmodelle: NMF auf TF-IDF, NMF auf Count, LDA auf Count
5. Themenanteile: Anteil der Dokumente, deren dominantes Thema das jeweilige Thema ist
6. Vergleich der Vektorisierungen über die Überschneidung der Top-Wörter (Jaccard)
7. Bewertung mit UMass-Kohärenz und Vergleich von k = 5, 8 und 10

## Ausgabe

Konsolenausgabe mit Top-Wörtern und Anteilen je Thema sowie CSV-Dateien im Ordner `results/`:

| Datei | Inhalt |
|---|---|
| `top_words_nmf_tfidf.csv`, `top_words_nmf_count.csv`, `top_words_lda_count.csv` | Top-10-Wörter und Anteil je Thema |
| `topic_shares_*.csv` | Themenanteile in % |
| `vectorizer_comparison.csv` | Zuordnung und Überschneidung TF-IDF- vs. Count-Themen |
| `model_coherence.csv` | UMass-Kohärenz der drei Modelle |
| `k_comparison.csv` | Kohärenz und Rekonstruktionsfehler für k = 5, 8, 10 |

## Ergebnisse (Standardlauf)

NMF auf TF-IDF, k = 5:

| Thema | Top-Wörter (Auszug) | Anteil |
|---|---|---|
| Zahlungen, Kredite, Bankkonten | payment, bank, loan, card, late | 47,0 % |
| Kreditberichte und Inkasso | credit report, bureau, inquiry, debt, collection | 38,1 % |
| Identitätsdiebstahl | identity theft, victim, fraudulent account | 9,2 % |
| Bearbeitung von Widersprüchen | dispute, response, deleted, day | 4,0 % |
| Kontoeröffnungen und Salden | acct opened, balance, charge | 1,7 % |

Die vollständigen Ergebnisse stehen im Ordner `results/`.
