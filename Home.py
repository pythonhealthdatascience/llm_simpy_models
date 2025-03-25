import streamlit as st

st.title("LLM-Generated Simulation Apps")
st.markdown(
    """
**Discrete-event simulation (DES) model web applications generated using Large
Language Models (LLMs).**"""
)

st.divider()
st.markdown(
    """
This application is complementary to:

> Thomas Monks, Alison Harper, and Amy Heather. **Using Large
Language Models to support researchers reproduce and reuse unpublished health
care discrete-event simulation computer models: a feasibility and pilot study
in Python**. https://github.com/pythonhealthdatascience/llm_simpy.

It deploys the streamlit applications generated using Perplexity as a single
app via stlite and GitHub pages.

Use the sidebar to navigate to each app.
"""
)

st.divider()
st.markdown(
    """
Source code: https://github.com/pythonhealthdatascience/llm_simpy_models."""
)
