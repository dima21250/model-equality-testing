# Analytic VADER KS Test (Streamlit page)

"""Streamlit page that runs the VADER sentiment KS test using an analytical p‑value.
It reports the test statistic, p‑value, wall‑clock time, and memory usage.
"""

import streamlit as st
from model_equality_testing.utils import Stopwatch, MemoryWatch
from model_equality_testing.pvalue import AnalyticalKSPvalueCalculator
from scipy.stats import ks_2samp
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Ensure data is loaded in the session state
if "loaded_data" not in st.session_state:
    st.warning("Please upload a data file on the main page first.")
    st.stop()

data = st.session_state["loaded_data"]

samples = data["samples"]
model_keys = list(samples.keys())

st.header("Analytic VADER KS Test")
st.subheader("Select two models to compare")
col1, col2 = st.columns(2)
with col1:
    model_a = st.selectbox("Model A", model_keys, index=0, key="model_a")
with col2:
    model_b = st.selectbox("Model B", model_keys, index=1, key="model_b")
# Display sample sizes for the selected models
size_a = samples[model_a].N
size_b = samples[model_b].N
st.write(f"Sample size per model: {model_a}: {size_a}, {model_b}: {size_b}")

if st.button("Run Test"):
    st.info(f"Running VADER‑KS test: **{model_a}** vs **{model_b}**")
    # ---- Timing & memory block ----
    with Stopwatch() as sw:
        # Compute reference VADER scores for model A
        analyzer = SentimentIntensityAnalyzer()
        def _vader_scores(sample):
            data_arr = sample.completion_sample.cpu().numpy()
            scores = []
            for row in data_arr:
                txt = "".join(chr(c) for c in row if c != -1)
                scores.append(analyzer.polarity_scores(txt)["compound"])
            return scores

        # Compute VADER scores for both selected models
        scores_a = _vader_scores(samples[model_a])
        scores_b = _vader_scores(samples[model_b])
        # Compute KS statistic and p-value directly
        test_stat, pvalue = ks_2samp(scores_a, scores_b)
    # ---- End timing block ----

    st.success("Test completed!")
    st.metric(label="VADER‑KS statistic (D)", value=f"{test_stat:.5f}")
    st.metric(label="Analytic p‑value", value=f"{pvalue:.5f}")
    st.metric(label="Wall‑clock time (s)", value=f"{sw.time:.2f}")
    st.metric(label="Sample size per model", value=f"{model_a}: {size_a}, {model_b}: {size_b}")
  

    if pvalue < 0.05:
        st.error("Result: REJECT null hypothesis – the two models differ in sentiment distribution.")
    else:
        st.info("Result: FAIL TO REJECT – cannot distinguish the models based on sentiment.")
