# Legal Text Classification

Classify U.S. Supreme Court opinions from the textacy SupremeCourt dataset.
The project compares a TF-IDF baseline with LDA and Doc2Vec representations.
All experiments use `random_state=42`, a stratified 80/20 split, accuracy, and
macro-F1.

## Setup

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train models

Run these commands from the repository root. The first run downloads the
dataset to `data/raw/` and caches labeled text in `data/processed/`.

```powershell
python src/train_baseline.py --target issue_area
python src/train_topics.py --target issue_area
python src/train_doc2vec.py --target issue_area
```

Use `--target issue` for the more detailed 279-class task. Use `--limit 500`
for a quick smoke test. Metrics are written to `results/` and models to
`models/`.

## Similarity search and demo

```powershell
python src/similarity_search.py "freedom of speech and the First Amendment"
streamlit run app.py
```

The Streamlit demo predicts an issue area with the baseline and shows similar
opinions when the Doc2Vec model has been trained.
