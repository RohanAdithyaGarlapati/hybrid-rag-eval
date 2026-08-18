import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st

# Bridge Streamlit Cloud secrets into environment variables so the provider
# agnostic LLM layer (_llm.py) can pick up the key. Safe when no secrets exist.
try:
    _secrets = dict(st.secrets)
except Exception:
    _secrets = {}
for _key in ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "GROQ_MODEL", "ANTHROPIC_MODEL"):
    if _key in _secrets and _secrets[_key]:
        os.environ[_key] = str(_secrets[_key])

from hybridrag.dataset import load_dataset
from hybridrag.embeddings import build_embedder
from hybridrag.generator import AnswerGenerator
from hybridrag.pipeline import answer_question
from hybridrag.retriever import HybridRetriever


@st.cache_resource
def get_retriever():
    ds = load_dataset(Path(__file__).parent / "data")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emb = build_embedder("auto")
    return HybridRetriever(ds["corpus"], emb, strategy="overlapping", abstain_threshold=0.30), emb


retriever, emb = get_retriever()
gen_available = AnswerGenerator().available

st.title("hybrid-rag-eval")
st.caption(f"Embedder: {emb.name}  |  is_semantic={emb.is_semantic}  |  generation: {'on' if gen_available else 'off (no API key)'}")

query = st.text_input("Query", "how does bm25 rank documents")
mode = st.radio("Mode", ["lexical", "dense", "hybrid-rrf", "hybrid-normalized"], index=2, horizontal=True)
k = st.slider("k", 1, 10, 5)
do_generate = st.checkbox("Generate a grounded answer and judge it", value=True)

if st.button("Search") and query.strip():
    ret = retriever.retrieve(query, k=k, mode=mode)
    st.write(f"Max dense cosine: {ret.max_dense_sim:.3f}  |  Abstained: {ret.abstained}")

    if do_generate:
        res = answer_question(retriever, query, k=k, mode=mode)
        st.subheader("Answer")
        if res.abstained:
            st.warning(res.answer)
        elif res.answer is None:
            st.info("Generation is not configured on this deployment (no GROQ_API_KEY / ANTHROPIC_API_KEY). Retrieval is shown below.")
        else:
            st.success(res.answer)
            st.caption(
                f"provider={res.provider} · model={res.model} · "
                f"faithfulness={res.faithfulness} · answer_relevance={res.answer_relevance}"
            )

    st.subheader("Retrieved documents")
    for rc in ret.chunks:
        st.markdown(f"**{rc.rank}. {rc.chunk.title}** — doc `{rc.chunk.doc_id}`, score {rc.score:.4f}")
        st.write(rc.chunk.text[:250])
