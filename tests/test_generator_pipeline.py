import warnings

import pytest

from hybridrag.embeddings import HashingEmbedder
from hybridrag.generator import AnswerGenerator
from hybridrag.judge import AnthropicJudge
from hybridrag.pipeline import ABSTAIN_MESSAGE, answer_question
from hybridrag.retriever import HybridRetriever

CORPUS = [
    {"id": "doc-bm25", "title": "BM25", "text": "BM25 scores a document from term frequency, inverse document frequency, and length normalization."},
    {"id": "doc-rrf", "title": "RRF", "text": "Reciprocal rank fusion combines ranked lists by summing one over k plus rank."},
    {"id": "doc-cache", "title": "Caching", "text": "Caching stores an expensive result so repeated requests skip the work."},
]


class _Block:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Msg:
    def __init__(self, text):
        self.content = [_Block(text)]


class _Messages:
    def __init__(self, text):
        self._text = text

    def create(self, **kwargs):
        return _Msg(self._text)


class _AnthropicFake:
    def __init__(self, text):
        self.messages = _Messages(text)


class _ChoiceMsg:
    def __init__(self, text):
        self.content = text


class _Choice:
    def __init__(self, text):
        self.message = _ChoiceMsg(text)


class _CompletionsResp:
    def __init__(self, text):
        self.choices = [_Choice(text)]


class _Completions:
    def __init__(self, text):
        self._text = text

    def create(self, **kwargs):
        return _CompletionsResp(self._text)


class _Chat:
    def __init__(self, text):
        self.completions = _Completions(text)


class _OpenAIFake:
    def __init__(self, text):
        self.chat = _Chat(text)


def _inject(obj, fake_client, provider="anthropic", model="claude-3-5-sonnet-20241022"):
    obj._llm.client = fake_client
    obj._llm.provider = provider
    obj._llm.model = model


@pytest.fixture
def retriever():
    return HybridRetriever(CORPUS, HashingEmbedder(dimension=256), strategy="fixed", abstain_threshold=0.30)


def test_generator_skips_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    gen = AnswerGenerator()
    assert gen.available is False
    assert gen.generate("q", "context") is None
    assert gen.reason_unavailable is not None


def test_generator_parses_fake_client_response(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    gen = AnswerGenerator()
    _inject(gen, _AnthropicFake("BM25 ranks by term frequency and length."), provider="anthropic", model="claude-3-5-sonnet-20241022")
    out = gen.generate("how does bm25 rank", "BM25 scores documents ...")
    assert out is not None
    assert "BM25" in out.text
    assert out.provider == "anthropic"
    assert out.model == "claude-3-5-sonnet-20241022"
    assert out.prompt_version == "grounded-answer-v1"


def test_generator_groq_path(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    gen = AnswerGenerator()
    _inject(gen, _OpenAIFake("BM25 ranks by term frequency."), provider="groq", model="llama-3.3-70b-versatile")
    out = gen.generate("how does bm25 rank", "BM25 scores documents ...")
    assert out is not None
    assert "BM25" in out.text
    assert out.provider == "groq"
    assert out.model == "llama-3.3-70b-versatile"


def test_pipeline_abstains_on_out_of_corpus(retriever):
    res = answer_question(retriever, "quantum chromodynamics gluon lattice confinement", k=3)
    assert res.abstained is True
    assert res.answer == ABSTAIN_MESSAGE
    assert res.answer_source == "abstained"


def test_pipeline_generator_unavailable_note(monkeypatch, retriever):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    res = answer_question(retriever, "how does bm25 score a document from term frequency", k=3)
    assert res.abstained is False
    assert res.answer is None
    assert res.answer_source == "generator-unavailable"
    assert any("generator unavailable" in n for n in res.notes)


def test_pipeline_generated_and_judged_with_fakes(monkeypatch, retriever):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    gen = AnswerGenerator()
    _inject(gen, _AnthropicFake("BM25 scores a document using term frequency, IDF, and length normalization."), provider="anthropic", model="claude-3-5-sonnet-20241022")
    judge = AnthropicJudge()
    _inject(judge, _AnthropicFake('{"faithfulness": 0.9, "answer_relevance": 0.85}'), provider="anthropic", model="claude-3-5-sonnet-20241022")

    res = answer_question(
        retriever,
        "how does bm25 score a document from term frequency",
        k=3,
        generator=gen,
        judge=judge,
    )
    assert res.answer_source == "generated"
    assert "BM25" in res.answer
    assert res.provider == "anthropic"
    assert res.model == "claude-3-5-sonnet-20241022"
    assert res.faithfulness == 0.9
    assert res.answer_relevance == 0.85


def test_generate_endpoint_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    from fastapi.testclient import TestClient

    from hybridrag import api

    api.get_generator.cache_clear()
    api.get_judge.cache_clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        client = TestClient(api.app)
    resp = client.post("/generate", json={"query": "how does bm25 rank documents", "k": 5, "mode": "hybrid-rrf"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer_source"] in ("generated", "abstained", "generator-unavailable")
    assert "doc_ids" in body
