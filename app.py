"""Streamlit interface for classifying Supreme Court opinions."""

from pathlib import Path

import joblib
import altair as alt
import pandas as pd
import streamlit as st

from src.preprocess import ISSUE_AREA_NAMES, clean_text


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "baseline.joblib"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"


@st.cache_resource
def load_model():
    """Load the trained baseline model once per Streamlit process."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Baseline model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_test_data() -> pd.DataFrame:
    """Load the test data once and cache it between reruns."""
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"Test data not found: {TEST_DATA_PATH}")
    return pd.read_csv(TEST_DATA_PATH)


def label_name(label: int) -> str:
    """Return a readable issue-area name for a model label."""
    return ISSUE_AREA_NAMES.get(int(label), str(label))


st.set_page_config(page_title="Supreme Court Opinion Classifier", layout="wide")
st.title("Supreme Court Opinion Classifier")

try:
    model = load_model()
    test_data = load_test_data()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

if "opinion_text" not in st.session_state:
    st.session_state.opinion_text = ""
if "random_true_label" not in st.session_state:
    st.session_state.random_true_label = None

if st.button("Load random test case"):
    random_case = test_data.sample(n=1).iloc[0]
    st.session_state.opinion_text = random_case["text"]
    st.session_state.random_true_label = random_case["issue_area"]

opinion_text = st.text_area(
    "Paste an opinion",
    key="opinion_text",
    height=300,
)

if st.session_state.random_true_label is not None:
    st.caption(
        "True issue area: "
        f"{label_name(st.session_state.random_true_label)}"
    )

if st.button("Classify"):
    cleaned_opinion = clean_text(opinion_text)
    if not cleaned_opinion:
        st.warning("Please enter an opinion before classifying.")
    else:
        prediction = model.predict([cleaned_opinion])[0]
        probabilities = model.predict_proba([cleaned_opinion])[0]
        classes = model.classes_

        st.success(f"Predicted issue area: {label_name(prediction)}")

        probability_table = pd.DataFrame(
            {
                "Class": [label_name(label) for label in classes],
                "Probability": probabilities,
            }
        ).sort_values("Probability", ascending=False).head(5)

        st.subheader("Top 5 predicted class probabilities")
        chart = (
            alt.Chart(probability_table)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Class:N",
                    sort=None,
                    title=None,
                    axis=alt.Axis(labelAngle=-30),
                ),
                y=alt.Y("Probability:Q", axis=alt.Axis(format="%")),
                tooltip=["Class", "Probability"],
            )
        )
        st.altair_chart(chart, use_container_width=True)
