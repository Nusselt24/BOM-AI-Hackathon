import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from teams.team_1.retrieval.vector import load_vectorstore
from teams.team_1.agent.adaptive_agent import answer_query

st.set_page_config(layout="wide")

@st.cache_resource
def init():
    return load_vectorstore()

vectorstore = init()

st.title("Adaptive Cancer Information Agent")

query = st.text_input("Ask a question about cancer:")

mode = st.radio(
    "Information focus",
    ["understanding", "evidence", "population"]
)

if query:
    answer, docs, strategy = answer_query(query, vectorstore, mode)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Why this answer?")
    st.json(strategy)

    with st.expander("Sources used"):
        for d in docs:
            st.markdown(
                f"- **{d.metadata.get('source')}** — {d.metadata.get('file', '')}"
            )

