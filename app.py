"""Small Streamlit demo for topic prediction and Doc2Vec similarity search."""

from pathlib import Path

import joblib
import streamlit as st
from gensim.models.doc2vec import Doc2Vec

from src.data import PROJECT_ROOT, load_dataset, tokenize


st.set_page_config(page_title="Supreme Court Topic Classifier", layout="wide")
st.title("Supreme Court Opinion Classifier")
st.caption("TF-IDF + Logistic Regression with optional Doc2Vec similarity search")

model_path = PROJECT_ROOT / "models" / "baseline_issue_area.joblib"
if not model_path.exists():
    st.warning("Train the baseline first: python src/train_baseline.py")
    st.stop()

model = joblib.load(model_path)
text = st.text_area("Paste an opinion or legal passage", height=240)
if st.button("Predict topic") and text.strip():
    st.success(f"Predicted issue area: {model.predict([text])[0]}")

st.subheader("Similar opinions")
doc2vec_path = PROJECT_ROOT / "models" / "doc2vec_issue_area.model"
if doc2vec_path.exists() and text.strip():
    d2v = Doc2Vec.load(str(doc2vec_path))
    data = load_dataset()
    vector = d2v.infer_vector(tokenize(text))
    scores = [
        (float(score), data.iloc[index]["text"])
        for index, score in enumerate(d2v.dv.cosine_similarities(vector, d2v.dv.vectors))
        if index < len(data)
    ]
    for score, opinion in sorted(scores, reverse=True)[:3]:
        st.write(f"**Similarity: {score:.3f}**")
        st.write(opinion[:700] + "...")
else:
    st.info("Train Doc2Vec and enter text to see similar opinions.")
