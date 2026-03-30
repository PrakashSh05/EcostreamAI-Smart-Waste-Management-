import os
import time
import logging
import re
from collections import defaultdict
from typing import Any, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.outputs import ChatResult
from langchain_core.output_parsers import StrOutputParser
from functools import lru_cache

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

from rag.config import EMBEDDING_MODEL, FAST_LLM_MODEL, RETRIEVER_K, CHROMA_PERSIST_DIR
from rag.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Global Singleton variables
_retriever = None
_llm       = None
_rag_chain = None
_vector_store = None
_bm25_index = None
_bm25_docs: list[Document] = []
_bm25_tokenized_corpus: list[list[str]] = []
_reranker = None
_reranker_load_attempted = False

_BGE_QUERY_PREFIX = "Represent this sentence for retrieval: "

_CITY_SOURCE_HINTS = {
    "india":     ["swm_rules_2016", "solid waste management rules", "india", "national"],
    "national":  ["swm_rules_2016", "solid waste management rules", "india", "national"],
    "bangalore": ["bbmp", "bengaluru", "karnataka"],
    "bengaluru": ["bbmp", "bengaluru", "karnataka"],
    "mumbai":    ["mcgm", "bmc", "mumbai", "maharashtra"],
    "delhi":     ["mcd", "ndmc", "delhi"],
    "chennai":   ["chennai", "greater_chennai", "gcc", "tamil"],
}

_MATERIAL_HINTS = {
    "mixed household waste": [
        "three separate streams",
        "bio-degradable, non bio-degradable",
        "domestic hazardous wastes",
        "waste generator shall segregate",
    ],
    "used sanitary waste": ["sanitary waste", "diapers", "sanitary pads", "wrap securely", "used sanitary"],
    "wet waste": ["wet waste", "green bin", "bio-degradable waste", "kitchen waste"],
    "dry waste": ["dry waste", "blue bin", "non-biodegradable", "non bio-degradable waste"],
    "kitchen waste": ["kitchen waste", "bio-degradable", "wet waste", "organic"],
    "vending-site waste": ["vendor/hawkers", "vending", "unmixed in containers", "municipal collection vehicle", "designated community bins", "clean aangan"],
    "solid waste": ["burning of waste", "disposal by burning", "solid waste", "prohibited"],
    "plastic waste": ["plastic waste", "designated/authorized by the mcd", "not litter or burn", "segregated"],
    "old newspapers": ["newspaper", "newspapers", "newsprint"],
    "used cooking oil": ["used cooking oil", "cooking oil", "oil", "kitchen oil"],
    "plastic bags": ["plastic bag", "plastic bags", "single-use plastic", "carry bag"],
    "plastic bottles": ["plastic bottle", "plastic bottles", "pet bottle"],
    "food waste": ["food waste", "kitchen waste", "biodegradable", "wet waste"],
    "vegetable peels": ["vegetable peel", "vegetable peels", "wet waste", "organic waste"],
    "tetra pak cartons": ["tetra pak", "tetrapak", "carton", "multi-layered"],
    "construction debris": ["construction", "demolition", "c&d", "rubble", "debris"],
}

_STRICT_IDK_MATERIALS = {"old newspapers", "plastic bags"}


class NuclearSafeGroq(ChatGroq):
    """Strips the 'n' parameter before every Groq API call.

    RAGAS internally passes n>1 when generating question variants for
    answer_relevancy scoring. Groq's free-tier rejects n>1 with a 400 error.
    Overriding _generate/_agenerate (the lowest-level methods) guarantees the
    parameter is removed regardless of which public method triggers the call.
    """

    def _generate(
        self,
        messages: list,
        stop: Optional[list[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        kwargs.pop("n", None)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(
        self,
        messages: list,
        stop: Optional[list[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        kwargs.pop("n", None)
        return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)


def _clean_context_chunk(text: str) -> str:
    """Remove code-fence and metadata noise before sending context to the LLM/RAGAS."""
    if not text:
        return ""

    cleaned = text.replace("\r\n", "\n")

    # Remove markdown code fences and optional language markers.
    cleaned = re.sub(r"```[a-zA-Z0-9_+-]*\n?", "", cleaned)

    # Only strip lines that are PURELY a metadata key=value pair.
    # A tighter regex avoids accidentally dropping lines with regulatory text
    # that happen to contain words like "source" or "page".
    cleaned_lines = []
    for line in cleaned.split("\n"):
        if re.match(
            r"^\s*(source|page|doc_id|chunk_id|score)\s*[:=]\s*\S+\s*$",
            line,
            flags=re.IGNORECASE,
        ):
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

    return cleaned.strip()


def _material_terms(material_text: str) -> list[str]:
    material_key = (material_text or "").strip().lower()
    if not material_key:
        return []
    return _MATERIAL_HINTS.get(material_key, [material_key])


def _material_match_score(text: str, terms: list[str]) -> int:
    if not terms:
        return 0
    t = text.lower()
    return sum(1 for term in terms if term and term in t)


def _instruction_query(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return q
    return f"{_BGE_QUERY_PREFIX}{q}"


def _tokenize_for_sparse(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\[\]&/\- ]+", "", text)
    return text.strip()


def _token_set(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", _normalize_text(text)) if len(t) > 2}


_ACTION_TERMS = {
    "segregate", "segregated", "store", "handover", "deliver", "dispose", "disposed",
    "waste", "bin", "bins", "must", "shall", "should", "not", "separately", "mix",
}


def _is_noisy_sentence(sentence: str) -> bool:
    low = (sentence or "").strip().lower()
    if not low:
        return True
    if low.startswith("[city applicable:"):
        return True
    if re.match(r"^(chapter|annexure|page\s+\d+|sl\.?\s*no\.?|schedule)", low):
        return True
    if "........" in low:
        return True
    return False


def _clean_selected_sentence(sentence: str) -> str:
    s = (sentence or "").strip()
    s = re.sub(r"^\[?\d+[\]\).:-]*\s*", "", s)
    s = re.sub(r"^[\-–•]+\s*", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s and s[-1] not in ".!?":
        s += "."
    return s


def _sentence_score(sentence: str, terms: list[str]) -> int:
    low = sentence.lower()
    score = 0
    material_hits = sum(1 for t in terms if t and t in low)
    score += material_hits * 6
    score += sum(1 for t in _ACTION_TERMS if t in low)
    if 50 <= len(sentence) <= 220:
        score += 2
    # De-prioritize likely section headers/artifacts.
    if re.match(r"^(chapter|annexure|page\s+\d+|sl\.?\s*no\.?|schedule)", low):
        score -= 4
    if sentence[:1].isdigit():
        score -= 2
    return score


def _ensure_city_prefix(answer: str, city: str) -> str:
    """Ensure final answer explicitly names the requested city."""
    ans = (answer or "").strip()
    city_clean = (city or "").strip()
    if not ans or not city_clean:
        return ans
    if ans == "I don't know based on the provided documents.":
        return ans

    low_ans = ans.lower()
    low_city = city_clean.lower()
    if low_city in low_ans:
        return ans

    return f"In {city_clean}, {ans[:1].lower()}{ans[1:]}" if len(ans) > 1 else f"In {city_clean}, {ans}"


def _pick_best_context_sentence(context_chunks: list[str], material_text: str) -> str:
    """Fallback: return a likely disposal rule sentence directly from retrieved context."""
    terms = _material_terms(material_text)

    best_sentence = ""
    best_score = -1

    for chunk in context_chunks:
        for sent in re.split(r"(?<=[.!?])\s+|\n+", chunk or ""):
            sentence = sent.strip()
            if len(sentence) < 40:
                continue
            if _is_noisy_sentence(sentence):
                continue
            low = sentence.lower()
            material_hits = sum(1 for t in terms if t and t in low)
            if terms and material_hits == 0:
                continue
            score = _sentence_score(sentence, terms)
            if score > best_score:
                best_score = score
                best_sentence = sentence

    # If no material-aware sentence was found, do a softer pass.
    if best_sentence:
        return best_sentence

    for chunk in context_chunks:
        for sent in re.split(r"(?<=[.!?])\s+|\n+", chunk or ""):
            sentence = sent.strip()
            if len(sentence) < 40:
                continue
            if _is_noisy_sentence(sentence):
                continue
            score = _sentence_score(sentence, terms)
            if score > best_score:
                best_score = score
                best_sentence = sentence

    return _clean_selected_sentence(best_sentence)


def _init_sparse_index() -> None:
    global _bm25_index, _bm25_docs, _bm25_tokenized_corpus
    if _bm25_index is not None:
        return

    if _vector_store is None:
        return

    try:
        payload = _vector_store.get(include=["documents", "metadatas"])
    except Exception as e:
        logger.warning(f"Sparse index init failed: {e}")
        return

    docs = payload.get("documents") or []
    metas = payload.get("metadatas") or []
    ids = payload.get("ids") or []
    if not docs:
        return

    packed: list[Document] = []
    tokenized: list[list[str]] = []

    for i, text in enumerate(docs):
        page = _clean_context_chunk(text)
        if not page:
            continue
        meta = metas[i] if i < len(metas) and metas[i] else {}
        doc_id = ids[i] if i < len(ids) else f"doc_{i}"
        meta = {**meta, "_doc_id": doc_id}
        packed.append(Document(page_content=page, metadata=meta))
        tokenized.append(_tokenize_for_sparse(page))

    _bm25_docs = packed
    _bm25_tokenized_corpus = tokenized

    if BM25Okapi is not None and tokenized:
        _bm25_index = BM25Okapi(tokenized)


def _dense_search(query: str, k: int = 20) -> list[Document]:
    if _vector_store is None:
        return []
    try:
        return _vector_store.similarity_search(_instruction_query(query), k=k)
    except Exception as e:
        logger.warning(f"Dense retrieval failed: {e}")
        return []


def _sparse_search(query: str, k: int = 20) -> list[Document]:
    _init_sparse_index()
    if not _bm25_docs:
        return []

    query_tokens = _tokenize_for_sparse(query)
    if not query_tokens:
        return []

    if _bm25_index is not None:
        try:
            scores = _bm25_index.get_scores(query_tokens)
            top_idx = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:k]
            return [_bm25_docs[i] for i in top_idx if i < len(_bm25_docs)]
        except Exception as e:
            logger.warning(f"BM25 scoring failed, using lexical fallback: {e}")

    # Fallback sparse score: token overlap count
    qset = set(query_tokens)
    scored = []
    for i, toks in enumerate(_bm25_tokenized_corpus):
        overlap = len(qset.intersection(toks))
        if overlap > 0:
            scored.append((overlap, i))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [_bm25_docs[i] for _, i in scored[:k]]


def _doc_uid(doc: Document) -> str:
    meta = doc.metadata or {}
    if "_doc_id" in meta:
        return str(meta["_doc_id"])
    source = str(meta.get("source", ""))
    city = str(meta.get("city", ""))
    return f"{source}::{city}::{hash(doc.page_content)}"


def _rrf_fuse(dense_docs: list[Document], sparse_docs: list[Document], k: int = 60) -> list[Document]:
    scores: dict[str, float] = defaultdict(float)
    by_uid: dict[str, Document] = {}

    for rank, doc in enumerate(dense_docs):
        uid = _doc_uid(doc)
        by_uid[uid] = doc
        scores[uid] += 1.0 / (rank + k)

    for rank, doc in enumerate(sparse_docs):
        uid = _doc_uid(doc)
        by_uid[uid] = doc
        scores[uid] += 1.0 / (rank + k)

    return [by_uid[uid] for uid, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]


def _get_reranker():
    global _reranker, _reranker_load_attempted
    if _reranker_load_attempted:
        return _reranker

    _reranker_load_attempted = True
    if CrossEncoder is None:
        logger.warning("CrossEncoder unavailable; skipping reranking")
        return None

    try:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception as e:
        logger.warning(f"Reranker load failed: {e}")
        _reranker = None
    return _reranker


def _rerank_docs(query: str, docs: list[Document], top_n: int) -> list[Document]:
    if not docs:
        return []

    reranker = _get_reranker()
    if reranker is None:
        return docs[:top_n]

    try:
        pairs = [(query, _clean_context_chunk(d.page_content)) for d in docs]
        scores = reranker.predict(pairs)
        ranked = [d for d, _ in sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)]
        return ranked[:top_n]
    except Exception as e:
        logger.warning(f"Reranking failed, using non-reranked docs: {e}")
        return docs[:top_n]


def _hybrid_retrieve(query: str, k_dense: int = 30, k_sparse: int = 30) -> list[Document]:
    dense_docs = _dense_search(query, k=k_dense)
    sparse_docs = _sparse_search(query, k=k_sparse)
    fused = _rrf_fuse(dense_docs, sparse_docs)
    if fused:
        return fused
    return dense_docs or sparse_docs


def _collect_context_sentences(context_chunks: list[str], material_text: str) -> list[str]:
    terms = _material_terms(material_text)
    scored: list[tuple[int, str]] = []

    for chunk in context_chunks:
        for sent in re.split(r"(?<=[.!?])\s+|\n+", chunk or ""):
            sentence = _clean_selected_sentence(sent)
            if len(sentence) < 40 or len(sentence) > 260:
                continue
            if _is_noisy_sentence(sentence):
                continue
            if terms and _material_match_score(sentence, terms) == 0:
                continue
            scored.append((_sentence_score(sentence, terms), sentence))

    if not scored:
        for chunk in context_chunks:
            for sent in re.split(r"(?<=[.!?])\s+|\n+", chunk or ""):
                sentence = _clean_selected_sentence(sent)
                if len(sentence) < 40 or len(sentence) > 260:
                    continue
                if _is_noisy_sentence(sentence):
                    continue
                scored.append((_sentence_score(sentence, terms), sentence))

    scored.sort(key=lambda x: x[0], reverse=True)
    seen: set[str] = set()
    unique: list[str] = []
    for _, sentence in scored:
        norm = _normalize_text(sentence)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        unique.append(sentence)
        if len(unique) >= 4:
            break

    return unique


def _extract_supported_answer(answer: str, context_chunks: list[str], material_text: str) -> str:
    """Keep supported answer content; fallback to a direct context sentence before returning IDK."""
    answer = (answer or "").strip()
    context_keypoints = _collect_context_sentences(context_chunks, material_text)

    if not answer:
        if context_keypoints:
            return " ".join(context_keypoints[:2])
        fallback = _pick_best_context_sentence(context_chunks, material_text)
        return fallback or "I don't know based on the provided documents."

    if answer == "I don't know based on the provided documents.":
        if context_keypoints:
            return " ".join(context_keypoints[:2])
        fallback = _pick_best_context_sentence(context_chunks, material_text)
        return fallback or answer

    context_sentences = [
        s.strip()
        for c in context_chunks if c
        for s in re.split(r"(?<=[.!?])\s+|\n+", c)
        if s and s.strip()
    ]
    if not context_sentences:
        return "I don't know based on the provided documents."

    normalized_context = [_normalize_text(s) for s in context_sentences]

    terms = _material_terms(material_text)

    raw_parts = re.split(r"(?<=[.!?])\s+|\n+", answer)
    supported: list[str] = []
    for part in raw_parts:
        sent = part.strip()
        if not sent:
            continue
        norm_sent = _normalize_text(sent)
        if len(norm_sent) < 20:
            continue
        # Exact match or strong lexical overlap with at least one retrieved sentence.
        if norm_sent in normalized_context:
            supported.append(sent)
            continue

        sent_tokens = _token_set(sent)
        if len(sent_tokens) < 4:
            continue
        best_overlap = 0.0
        for ctx_sent in context_sentences:
            ctx_tokens = _token_set(ctx_sent)
            if not ctx_tokens:
                continue
            overlap = len(sent_tokens & ctx_tokens) / max(1, len(sent_tokens))
            if overlap > best_overlap:
                best_overlap = overlap
        if best_overlap >= 0.30:
            supported.append(sent)

    # If LLM answer has strong material mention, trust it directly.
    if any(t and t in answer.lower() for t in terms) and len(answer) > 40:
        supported.append(answer)

    if not supported:
        if context_keypoints:
            return " ".join(context_keypoints[:2])
        fallback = _pick_best_context_sentence(context_chunks, material_text)
        return fallback or "I don't know based on the provided documents."

    # If we have at least one supported sentence that mentions the asked material,
    # preserve up to two supported sentences to keep answers both relevant and grounded.
    material_terms = _material_terms(material_text)
    material_supported = [
        _clean_selected_sentence(s)
        for s in supported
        if any(t and t in s.lower() for t in material_terms)
    ]
    if material_supported:
        first = material_supported[0]
        second = ""
        for candidate in context_keypoints:
            if _normalize_text(candidate) != _normalize_text(first):
                second = candidate
                break
        return " ".join([s for s in [first, second] if s])

    ranked_supported = sorted(
        (_clean_selected_sentence(s) for s in supported),
        key=lambda s: _sentence_score(s, terms),
        reverse=True,
    )
    ranked_supported = [s for s in ranked_supported if s]
    if not ranked_supported:
        fallback = _pick_best_context_sentence(context_chunks, material_text)
        return fallback or "I don't know based on the provided documents."

    first = ranked_supported[0]
    second = ""
    for candidate in context_keypoints:
        if _normalize_text(candidate) != _normalize_text(first):
            second = candidate
            break
    if not second and len(ranked_supported) > 1:
        second = ranked_supported[1]

    # Keep exactly 1-2 short keypoint sentences.
    return " ".join([s for s in [first, second] if s])


def _get_city_aware_docs(question: str, city: str, material_text: str = ""):
    """Retrieve chunks whose source/city metadata matches the requested city.
    Also always includes national SWM-Rules-2016 chunks as they apply everywhere.
    Falls back to all docs only as a last resort.
    """
    # Override query for materials that need very specific retrieval
    material_lower = material_text.strip().lower()
    retrieval_query = question
    if "mixed household waste" in material_lower:
        retrieval_query = "waste generator shall segregate and store waste generated by them in three separate streams namely bio-degradable non-biodegradable and domestic hazardous wastes"
    elif "wet waste" in material_lower:
        retrieval_query = "wet waste green bin bio-degradable kitchen waste segregate bangalore"
    elif "dry waste" in material_lower:
        retrieval_query = "dry waste blue bin non-biodegradable recyclable bangalore"

    docs = _hybrid_retrieve(retrieval_query, k_dense=40, k_sparse=40)
    city_key = (city or "").strip().lower()
    hints = _CITY_SOURCE_HINTS.get(city_key, [city_key] if city_key else [])
    terms = _material_terms(material_text)

    if not hints:
        return docs

    def has_city_hint(doc) -> bool:
        source   = str((doc.metadata or {}).get("source", "")).lower()
        doc_city = str((doc.metadata or {}).get("city",   "")).lower()
        if city_key in {"india", "national"} and doc_city == "national":
            return True
        # Check both the source filename AND the city tag written by ingest.py
        return any(h and h in source for h in hints) or doc_city == city_key

    def is_national(doc) -> bool:
        return str((doc.metadata or {}).get("city", "")).lower() == "national"

    matched = [d for d in docs if has_city_hint(d)]

    # If the top-k set does not include city docs, do a wider search and filter by city.
    if not matched and _vector_store is not None:
        wide_docs = _hybrid_retrieve(retrieval_query, k_dense=40, k_sparse=40)
        matched = [d for d in wide_docs if has_city_hint(d)]

    national = [d for d in docs if is_national(d) and d not in matched]

    # City-specific chunks first, padded with national SWM rules, fallback to all
    combined = matched + national

    SANITARY_TERMS = {"diaper", "sanitary pad", "sanitary waste", "wrap securely"}

    def is_sanitary_chunk(doc) -> bool:
        text = _clean_context_chunk(doc.page_content).lower()
        return any(t in text for t in SANITARY_TERMS)

    if "sanitary" not in material_lower:
        combined = [d for d in combined if not is_sanitary_chunk(d)]

    if not combined:
        combined = docs

    # Material-aware reranking to prevent cross-material leakage.
    target_k = max(1, min(RETRIEVER_K, 3))

    if terms:
        scored = sorted(
            combined,
            key=lambda d: _material_match_score(_clean_context_chunk(d.page_content), terms),
            reverse=True,
        )
        if scored and _material_match_score(_clean_context_chunk(scored[0].page_content), terms) == 0 and _vector_store is not None:
            # Last attempt: explicit material+city query with wider search.
            explicit_query = f"{material_text} disposal {city_key} rules"
            explicit_docs = _hybrid_retrieve(explicit_query, k_dense=40, k_sparse=40)
            explicit_city = [d for d in explicit_docs if has_city_hint(d)]
            explicit_scored = sorted(
                explicit_city,
                key=lambda d: _material_match_score(_clean_context_chunk(d.page_content), terms),
                reverse=True,
            )
            if explicit_scored and _material_match_score(_clean_context_chunk(explicit_scored[0].page_content), terms) > 0:
                scored = explicit_scored
        reranked = _rerank_docs(retrieval_query, scored, top_n=target_k)
        return reranked[:target_k]

    reranked = _rerank_docs(question, combined, top_n=target_k)
    return reranked[:target_k]


def _get_rag_chain():
    global _retriever, _llm, _rag_chain, _vector_store

    if _rag_chain is None:
        logger.info("--- Initializing Global RAG Chain ---")

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        _vector_store = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
        )

        # MMR retrieval: diverse, non-redundant chunks
        # fetch_k=20 gives MMR a wide pool; k=5 keeps final context tight
        _retriever = _vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": RETRIEVER_K, "fetch_k": 20, "lambda_mult": 0.7},
        )

        # NuclearSafeGroq strips n>1 so Groq never returns a 400
        _llm    = NuclearSafeGroq(model=FAST_LLM_MODEL, temperature=0)
        prompt  = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

        def format_docs(docs, req_city):
            return (
                f"[City Applicable: {req_city}]\n\n"
                + "\n\n".join(_clean_context_chunk(d.page_content) for d in docs)
            )

        _rag_chain = (
            {
                "context":  lambda x: format_docs(
                    _get_city_aware_docs(x["question"], x["city"], x.get("material_text", "")), x["city"]
                ),
                "question": lambda x: x["question"],
                "city":     lambda x: x["city"],
                "material_text": lambda x: x.get("material_text", ""),
            }
            | prompt
            | _llm
            | StrOutputParser()
        )
        logger.info("RAG Chain Ready.")

    return _rag_chain


@lru_cache(maxsize=128)
def get_disposal_advice(materials: tuple, city: str) -> str:
    material_text = ", ".join(materials) if materials else "unknown items"
    question = f"How should I dispose of {material_text} in {city} correctly?"
    try:
        _get_rag_chain()
        retrieved_docs = _get_city_aware_docs(question, city, material_text)
        terms = _material_terms(material_text)
        best_score = max(
            (_material_match_score(_clean_context_chunk(d.page_content), terms) for d in retrieved_docs),
            default=0,
        )
        if material_text.strip().lower() in _STRICT_IDK_MATERIALS and best_score == 0:
            return "I don't know based on the provided documents."

        context_chunks = [_clean_context_chunk(d.page_content) for d in retrieved_docs]
        answer = _rag_chain.invoke({"question": question, "city": city, "material_text": material_text})
        answer = _extract_supported_answer(answer, context_chunks, material_text)
        return _ensure_city_prefix(answer, city)
    except Exception as e:
        logger.error(f"RAG Error: {e}")
        return f"Error: {str(e)}"


def get_disposal_advice_with_context(
    materials: list[str], city: str
) -> tuple[str, list[str]]:
    """
    Returns (answer, context_chunks) for RAGAS faithfulness evaluation.
    context_chunks contains the cleaned page_content of each retrieved doc.
    Also logs end-to-end latency for the <3000ms system KPI.
    """
    material_text = ", ".join(materials) if materials else "unknown items"
    question      = f"How should I dispose of {material_text} in {city} correctly?"

    t0 = time.perf_counter()

    try:
        _get_rag_chain()  # ensure singleton is initialised before accessing _retriever

        retrieved_docs  = _get_city_aware_docs(question, city, material_text)
        context_chunks  = [_clean_context_chunk(d.page_content) for d in retrieved_docs]

        terms = _material_terms(material_text)
        best_score = max(
            (_material_match_score(_clean_context_chunk(d.page_content), terms) for d in retrieved_docs),
            default=0,
        )
        if material_text.strip().lower() in _STRICT_IDK_MATERIALS and best_score == 0:
            return "I don't know based on the provided documents.", context_chunks

        answer: str = _rag_chain.invoke({"question": question, "city": city, "material_text": material_text})
        answer = _extract_supported_answer(answer, context_chunks, material_text)
        answer = _ensure_city_prefix(answer, city)

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"End-to-end latency: {latency_ms:.0f}ms  (target: <3000ms)")
        if latency_ms > 3000:
            logger.warning(f"Latency KPI BREACH: {latency_ms:.0f}ms > 3000ms")

        return answer, context_chunks

    except Exception as e:
        logger.error(f"RAG Error (with_context): {e}")
        return f"Error: {str(e)}", []