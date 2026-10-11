# Classification of US Supreme Court Opinions

This project classifies US Supreme Court opinions into issue areas using
natural language processing and machine learning. It is based on the paper
_Classification of Legal Text_ by Krithika Iyer and compares traditional
text-representation methods for legal opinion classification.

## Problem statement

Given the text of a Supreme Court opinion, predict its issue area. The project
uses 13 issue-area classes, including Criminal Procedure, Civil Rights, First
Amendment, Economic Activity, and Judicial Power.

## Dataset

The project uses the `textacy` `SupremeCourt` dataset, which is derived from
the [Supreme Court Database](http://scdb.wustl.edu/).

- 8,419 opinions are downloaded from the dataset.
- 8,393 opinions remain after cleaning: 23 unlabeled cases, 2 empty texts, and
  1 case from a class with fewer than 5 samples are removed.
- A stratified 80/20 train/test split is created with `random_state=42`.
- Training set: 6,714 opinions.
- Test set: 1,679 opinions.

The data downloads automatically and is not stored on GitHub. The generated
data files are recreated by the data-loading and preprocessing commands.

## Approach

The project evaluates three approaches:

1. **TF-IDF + Logistic Regression**: the added baseline model.
2. **LDA + Logistic Regression**: a 50-topic Gensim representation followed by
   Logistic Regression.
3. **Doc2Vec + Logistic Regression**: 100-dimensional Gensim document vectors
   followed by Logistic Regression.

All classifiers use `class_weight="balanced"`. BERT was not used because the
opinions average about 7,000 words, which is far beyond its 512-token limit.

## Results

Results are reported on the 1,679-case test set.

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Majority-class guess | 0.229 | 0.029 |
| LDA + LR | 0.631 | 0.522 |
| Doc2Vec + LR | 0.630 | 0.539 |
| TF-IDF + LR | 0.783 | 0.706 |

TF-IDF performs best, while LDA and Doc2Vec are about tied.

## Project structure

```text
.
├── .github/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── load_data.py
│   ├── preprocess.py
│   ├── train_baseline.py
│   ├── train_lda.py
│   ├── train_doc2vec.py
│   └── evaluate.py
├── models/
├── results/
├── report/
├── app.py
├── requirements.txt
└── README.md
```

## Setup

The commands below are for Windows PowerShell and Python 3.11.

```powershell
git clone <repository-url>
cd legal-text-classification
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## How to run

Run the commands from the repository root in the following order.

Download and save the Supreme Court dataset:

```powershell
python -m src.load_data
```

Clean the opinions and create the stratified train/test files:

```powershell
python -m src.preprocess
```

Train the TF-IDF + Logistic Regression baseline:

```powershell
python -m src.train_baseline
```

Train the LDA + Logistic Regression model:

```powershell
python -m src.train_lda
```

Train the Doc2Vec + Logistic Regression model:

```powershell
python -m src.train_doc2vec
```

Create the model comparison table and evaluation plots:

```powershell
python -m src.evaluate
```

Start the Streamlit demo:

```powershell
streamlit run app.py
```

The `data/` and `models/` directories are not stored on GitHub. They are
recreated by the commands above.

## Demo

The Streamlit app supports two input options:

1. Click **Load random test case** to load an opinion from the test set and
   display its true issue area.
2. Paste an opinion into the text area.

Click **Classify** to view the predicted issue area and a chart of the top five
predicted class probabilities.

## Limitations

- The issue-area classes are imbalanced.
- Some classes are very small; Miscellaneous has only 4 test cases.
- Results use a single train/test split and no cross-validation.
- The task is single-label classification, although individual cases can cover
  several topics.
- Results are not directly comparable with the paper because the dataset
  version differs.

## Team

| Name | GitHub username |
|---|---|
| GOWTHAM S | gowth139am-lab |
| Harsh VenkatSai Kalekar| Harsh Kalekar |

## Reference

- `textacy` library: the source library used to download and read the
  `SupremeCourt` dataset.
- [Supreme Court Database](http://scdb.wustl.edu/).
- Krithika Iyer, _Classification of Legal Text_.
