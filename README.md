# NLP-Analyse von Verbraucherbeschwerden (CFPB)

Dieses Repository enthält den Code für die Analyse von Beschwerdetexten mit NLP-Techniken (TF-IDF, CountVectorizer, NMF, LDA).

**Autor:** Niklas Lazar (Matrikelnummer: IU14136580)  
**Kurs:** Projekt: Data Analysis (DLBDSEDA02_D)

## Daten
Die Datei `complaints_processed.csv` muss im selben Verzeichnis liegen.  
Quelle: https://github.com/halpert3/complaint-content-classification-nlp (basierend auf CFPB-Daten)

## Installation und Nutzung

### Windows
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python nlp_analysis.py
```

### macOS / Linux
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python nlp_analysis.py
```

## Ausgabe
- Konsolenausgabe mit Top-Wörtern und Themenanteilen in %
- CSV-Dateien im Ordner `results/`

Laufzeit: ca. 1–2 Minuten (Stichprobe 20.000 Dokumente).  
Python 3.10+ empfohlen.
