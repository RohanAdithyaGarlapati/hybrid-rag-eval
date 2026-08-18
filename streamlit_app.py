import sys, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
from hybridrag.dataset import load_dataset
from hybridrag.embeddings import build_embedder
from hybridrag.retriever import HybridRetriever


@st.cache_resource
def get_retriever():
    ds = load_dataset(Path(__file__).parent / "data")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emb = build_embedder("auto")
    return HybridRetriever(ds["corpus"], emb, strategy="overlapping", abstain_threshold=0.30), emb


retriever, emb = get_retriever()

st.title("hybrid-rag-eval")
st.caption(f"Embedder: {emb.name}  |  is_semantic={emb.is_semantic}")

query = st.text_input("Query", "how does bm25 rank documents")
mode = st.radio("Mode", ["lexical", "dense", "hybrid-rrf", "hybrid-normalized"], index=2, horizontal=True)
k = st.slider("k", 1, 10, 5)

if st.button("Search") and query.strip():
    res = retriever.retrieve(query, k=k, mode=mode)
    st.write(f"Max dense cosine: {res.max_dense_sim:.3f}  |  Abstained: {res.abstained}")
    if res.abstained:
        st.warning("Guardrail abstained: confidence below threshold.")
    for rc in res.chunks:
        st.markdown(f"**{rc.rank}. {rc.chunk.title}** — doc `{rc.chunk.doc_id}`, score {rc.score:.4f}")
        st.write(rc.chunk.text[:250])
