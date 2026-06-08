"""Hybrid config — rag_complex pipeline with hybridRAG retrieval swapped in.

Pipeline shell is identical to rag_complex.py (rewrite -> HyDE -> retrieve ->
grade -> [retry] -> candidates -> CoT select). The retrieval module is
replaced with the hybridRAG approach from hybridRAG/test2.ipynb:
    1. multilingual-e5-base (fine-tuned if available) with E5 query/passage prefixes.
    2. FAISS IndexFlatIP over chunk embeddings (cosine via L2-normalized vectors).
    3. ICD knowledge graph: parent-child edges (E78 -> E78.0) weight=2,
       co-occurrence edges from protocols' icd_codes lists weight=+1 per shared protocol.
    4. Hybrid candidate scoring = 0.7 * semantic (cosine-weighted) + 0.3 * graph-expanded.
    5. Domain keyword boosts (беремен / давление-гипертенз / hellp) applied at the code level.

Run: `python rag_hybrid.py`
"""
import json
import math
import os
import re
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import TypedDict

import numpy as np
from openai import OpenAI
from tqdm.auto import tqdm

BASE_DIR = Path(os.environ.get("RAG_BASE_DIR") or Path(__file__).resolve().parent)
CORPUS_PATH = BASE_DIR / "protocols_corpus.jsonl"
TEST_ZIP = BASE_DIR / "test_set.zip"
TEST_DIR = BASE_DIR / "test_set"
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)
EVAL_OUT = BASE_DIR / "eval_results_hybrid.jsonl"
SUMMARY_OUT = BASE_DIR / "eval_summary_hybrid.json"
HYBRID_FT_DIR = BASE_DIR / "hybridRAG" / "fine_tuned_model"

HF_CACHE = CACHE_DIR / "hf"
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE / "hub"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(HF_CACHE / "sentence_transformers"))

GEMMA_URL = os.environ.get("GEMMA_URL", "http://92.46.110.71:8887/v1")
GEMMA_MODEL = os.environ.get("GEMMA_MODEL", "/models/gemma-4-26B-A4B-it")

import torch  # noqa: E402
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if TEST_DIR.exists() and len(list(TEST_DIR.glob("**/*.json"))) == 0 and TEST_ZIP.exists():
    with zipfile.ZipFile(TEST_ZIP) as zf:
        zf.extractall(TEST_DIR)
elif not TEST_DIR.exists() and TEST_ZIP.exists():
    TEST_DIR.mkdir(exist_ok=True)
    with zipfile.ZipFile(TEST_ZIP) as zf:
        zf.extractall(TEST_DIR)

test_files = sorted(TEST_DIR.glob("**/*.json"))
TEST_LIMIT = int(os.environ.get("RAG_TEST_LIMIT", "0"))
if TEST_LIMIT > 0:
    test_files = test_files[:TEST_LIMIT]
client = OpenAI(base_url=GEMMA_URL, api_key="no")

# ---------- Corpus (matches hybridRAG word-based chunking: size=150, overlap=30) ----------
CHUNK_WORDS = 150
CHUNK_OVERLAP_WORDS = 30


def _chunk_words(text, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP_WORDS):
    words = text.split()
    out, i = [], 0
    while i < len(words):
        out.append(" ".join(words[i:i + size]))
        if i + size >= len(words):
            break
        i += size - overlap
    return out


def load_corpus():
    protos = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            protos.append(json.loads(line))
    chunks = []
    for pi, p in enumerate(protos):
        for ch in _chunk_words(p["text"]):
            chunks.append({
                "text": ch, "protocol_id": p["protocol_id"],
                "protocol_idx": pi, "source_file": p["source_file"],
            })
    return protos, chunks


CYR2LAT = str.maketrans({"А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H",
                         "К": "K", "М": "M", "О": "O", "Р": "P", "Т": "T", "Х": "X"})
CODE_RE_LATIN = re.compile(r"[A-Z]\d{2}(?:\.\d{1,2})?")
RANGE_RE_LATIN = re.compile(r"([A-Z]\d{2}(?:\.\d{1,2})?)\s*-\s*([A-Z]\d{2}(?:\.\d{1,2})?)")
SECTION_BREAK_RE = re.compile(r"\s(?:1\.\d+|\d+\.\s*[А-Я])")
ICD_CODE_RE = re.compile(r"\b[A-Z]\d{2}(?:\.\d{1,2})?(?:[A-Z0-9]{1,3})?\b")


def extract_icd_descriptions(protocols):
    desc = {}
    for p in protocols:
        t = p["text"]
        i = t.find("МКБ-10")
        if i < 0:
            continue
        section = t[i:i + 2500].translate(CYR2LAT)
        cut = SECTION_BREAK_RE.search(section, 50)
        if cut:
            section = section[:cut.start()]
        matches = list(CODE_RE_LATIN.finditer(section))
        ranges = list(RANGE_RE_LATIN.finditer(section))
        for mi, m in enumerate(matches):
            code = m.group(0)
            start = m.end()
            end = matches[mi + 1].start() if mi + 1 < len(matches) else len(section)
            d = re.sub(r"\s+", " ", section[start:end].strip(" .,;:-–—\t"))
            if 5 <= len(d) <= 120 and re.search(r"[а-яА-ЯёЁa-zA-Z]", d) \
                    and not any(t in d.lower() for t in ["мкб", "клинический протокол"]):
                desc.setdefault(code, d)
        for rm in ranges:
            start = rm.end()
            nxt = next((m for m in matches if m.start() >= start), None)
            end = nxt.start() if nxt else len(section)
            d = re.sub(r"\s+", " ", section[start:end].strip(" .,;:-–—\t"))
            if 5 <= len(d) <= 120 and re.search(r"[а-яА-ЯёЁa-zA-Z]", d):
                prefix = rm.group(1).split(".")[0]
                desc.setdefault(prefix, d)
    return desc


# ---------- E5 dense index + ICD graph (hybridRAG style) ----------
def build_dense_index(chunks):
    """Load multilingual-e5-base (fine-tuned if HYBRID_FT_DIR exists) and build chunk embeddings."""
    from sentence_transformers import SentenceTransformer

    if HYBRID_FT_DIR.exists() and (HYBRID_FT_DIR / "model.safetensors").exists():
        model_id = str(HYBRID_FT_DIR)
        cache_name = "e5_ft_chunks.npy"
        print(f"Loading fine-tuned E5 from {HYBRID_FT_DIR}")
    else:
        model_id = "intfloat/multilingual-e5-base"
        cache_name = "e5_base_chunks.npy"
        print(f"Loading base {model_id}")
    model = SentenceTransformer(model_id, device=DEVICE)

    cache = CACHE_DIR / cache_name
    if cache.exists():
        embs = np.load(cache)
        if embs.shape[0] == len(chunks):
            return model, embs.astype(np.float32)
    # E5: passages need "passage: " prefix
    texts = ["passage: " + c["text"] for c in chunks]
    embs = model.encode(texts, batch_size=16, show_progress_bar=True,
                        normalize_embeddings=True, convert_to_numpy=True)
    np.save(cache, embs)
    return model, embs.astype(np.float32)


def build_faiss_index(embs):
    import faiss
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.ascontiguousarray(embs))
    return index


def build_icd_graph(protocols):
    """Build ICD knowledge graph from corpus.
       Edges: parent-child (E78 -> E78.0) weight=2; co-occurrence weight=+1 per shared protocol.
    """
    import networkx as nx
    G = nx.Graph()
    for p in protocols:
        codes = list(p.get("icd_codes") or [])
        for code in codes:
            G.add_node(code)
            parts = code.split(".")
            if len(parts) > 1:
                parent = parts[0]
                G.add_node(parent)
                if G.has_edge(code, parent):
                    G[code][parent]["weight"] += 2
                else:
                    G.add_edge(code, parent, weight=2)
        for c1, c2 in combinations(codes, 2):
            if G.has_edge(c1, c2):
                G[c1][c2]["weight"] += 1
            else:
                G.add_edge(c1, c2, weight=1)
    return G


def faiss_chunk_search(index, query_emb, k=20):
    q = np.ascontiguousarray(query_emb.reshape(1, -1).astype(np.float32))
    scores, idxs = index.search(q, k)
    return list(zip(idxs[0].tolist(), scores[0].tolist()))


def chunks_to_protocols(chunks, chunk_hits):
    best = {}
    for ci, score in chunk_hits:
        pi = chunks[ci]["protocol_idx"]
        if pi not in best or score > best[pi]:
            best[pi] = score
    return sorted(best.items(), key=lambda x: -x[1])


def graph_expand(graph, seed_codes, top_n=20):
    """Expand seed codes via graph neighbors weighted by edge weight."""
    scores = defaultdict(float)
    for code in seed_codes:
        if code not in graph:
            continue
        for neighbor in graph.neighbors(code):
            scores[neighbor] += graph[code][neighbor].get("weight", 1)
    for code in seed_codes:
        scores.pop(code, None)
    return sorted(scores.items(), key=lambda x: -x[1])[:top_n]


def retrieve(search_query, hyde_text, *, dense_model, faiss_index, chunks, protocols):
    """hybridRAG retrieval: E5 query embedding + FAISS chunk search.
       Returns (top5_protos, chunk_hits) so that downstream candidate scoring
       can use the per-chunk semantic scores."""
    # E5 needs "query: " prefix
    qe = dense_model.encode(["query: " + search_query], normalize_embeddings=True,
                            convert_to_numpy=True)[0].astype(np.float32)
    chunk_hits = faiss_chunk_search(faiss_index, qe, k=20)
    if hyde_text:
        # HyDE doc embedded as a passage (it's a synthesized protocol excerpt)
        he = dense_model.encode(["passage: " + hyde_text], normalize_embeddings=True,
                                convert_to_numpy=True)[0].astype(np.float32)
        hyde_hits = faiss_chunk_search(faiss_index, he, k=20)
        merged = {}
        for ci, s in chunk_hits + hyde_hits:
            if ci not in merged or s > merged[ci]:
                merged[ci] = s
        chunk_hits = sorted(merged.items(), key=lambda x: -x[1])[:30]
    top5 = chunks_to_protocols(chunks, chunk_hits)[:5]
    return top5, chunk_hits


# ---------- LLM ----------
def llm(messages, max_tokens=512, temperature=0.0, retries=2):
    for attempt in range(retries + 1):
        try:
            r = client.chat.completions.create(
                model=GEMMA_MODEL, messages=messages,
                max_tokens=max_tokens, temperature=temperature,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return r.choices[0].message.content or ""
        except Exception:
            if attempt == retries:
                raise
            time.sleep(1.0 * (attempt + 1))
    return ""


def llm_json(messages, max_tokens=512, temperature=0.0):
    raw = llm(messages, max_tokens, temperature)
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return None, raw
    try:
        return json.loads(m.group(0)), raw
    except Exception:
        return None, raw


# ---------- Agentic nodes (identical to rag_complex.py) ----------
def node_rewrite(query):
    sys_msg = (
        "Ты медицинский ассистент. Извлеки из жалоб пациента структурированную информацию. "
        'Верни СТРОГО JSON: {"search_query": "...", "symptoms": "...", "body_part": "..."}. '
        "search_query — короткая медицинская формулировка (3-8 слов) для поиска клинического протокола."
    )
    data, _ = llm_json([{"role": "system", "content": sys_msg},
                       {"role": "user", "content": query}], max_tokens=250)
    if data and data.get("search_query", "").strip():
        return data["search_query"].strip()
    return query[:200]


def node_hyde(query):
    sys_msg = (
        "Ты — клиницист. Напиши 3-4 предложения из клинического протокола на русском, "
        "описывающих диагноз и ведение пациента с такими жалобами. Используй медицинскую терминологию, "
        "упомяни вероятный диагноз и код МКБ-10 если уместно."
    )
    try:
        return llm([{"role": "system", "content": sys_msg},
                   {"role": "user", "content": query}],
                  max_tokens=220, temperature=0.2).strip()
    except Exception:
        return ""


def node_grade(query, top5, protocols):
    lines = []
    for i, (pi, _) in enumerate(top5):
        p = protocols[pi]
        title = (p.get("source_file") or "").replace(".pdf", "")
        snippet = p["text"][:300].replace("\n", " ")
        lines.append(f"{i+1}. {title}\n   {snippet}")
    sys_msg = (
        "Ты оцениваешь качество поиска. Отвечают ли найденные клинические протоколы на жалобу пациента? "
        'Верни СТРОГО JSON: {"verdict": "GOOD"|"BAD", "reason": "..."}.'
    )
    user_msg = f"Жалоба:\n{query}\n\nНайденные протоколы:\n" + "\n\n".join(lines)
    data, _ = llm_json([{"role": "system", "content": sys_msg},
                       {"role": "user", "content": user_msg}], max_tokens=200)
    if data and data.get("verdict") in ("GOOD", "BAD"):
        return data["verdict"], data.get("reason", "")
    return "GOOD", "parse failed"


def node_rewrite_with_context(query, history, reason):
    hist = "\n".join(
        f"Попытка {i+1}: '{h['query']}' → {', '.join(h['titles'][:3])}"
        for i, h in enumerate(history)
    )
    sys_msg = (
        "Предыдущий поиск не нашёл подходящего протокола. Предложи ДРУГУЮ формулировку (3-8 слов), "
        'не повторяющую предыдущие. Верни СТРОГО JSON: {"search_query": "..."}.'
    )
    user_msg = f"Жалоба:\n{query}\n\nНеудачные попытки:\n{hist}\n\nПричина: {reason}"
    data, _ = llm_json([{"role": "system", "content": sys_msg},
                       {"role": "user", "content": user_msg}], max_tokens=150)
    if data and data.get("search_query", "").strip():
        return data["search_query"].strip()
    return query[:200]


# ---------- hybridRAG candidate aggregation: semantic + graph + keyword boosts ----------
SEMANTIC_WEIGHT = 0.7
GRAPH_WEIGHT = 0.3


def collect_candidates_hybrid(query, chunk_hits, chunks, protocols, icd_graph,
                              icd_desc, cap=40):
    """hybridRAG candidate scoring: cosine-weighted semantic + graph-expanded + keyword boosts."""
    # 1. Semantic: per chunk, distribute its FAISS score to all icd_codes of its protocol.
    sem_scores = defaultdict(float)
    code_titles = defaultdict(list)
    seen_title = defaultdict(set)
    for ci, score in chunk_hits:
        ch = chunks[ci]
        proto = protocols[ch["protocol_idx"]]
        title = (proto.get("source_file") or "").replace(".pdf", "")
        s = max(0.0, float(score))
        for code in proto.get("icd_codes") or []:
            sem_scores[code] += s
            if title not in seen_title[code]:
                code_titles[code].append(title)
                seen_title[code].add(title)
    sem_total = sum(sem_scores.values()) or 1.0
    sem_norm = {c: s / sem_total for c, s in sem_scores.items()}

    # 2. Graph: expand the seed codes via the ICD knowledge graph.
    seed_codes = list(sem_scores.keys())
    graph_scores = defaultdict(float)
    for code, w in graph_expand(icd_graph, seed_codes, top_n=20):
        graph_scores[code] += w
    for code in seed_codes:
        if code in icd_graph:
            graph_scores[code] += icd_graph.degree(code, weight="weight")
    graph_total = sum(graph_scores.values()) or 1.0
    graph_norm = {c: s / graph_total for c, s in graph_scores.items()}

    all_codes = set(sem_norm) | set(graph_norm)
    combined = {
        c: SEMANTIC_WEIGHT * sem_norm.get(c, 0.0) + GRAPH_WEIGHT * graph_norm.get(c, 0.0)
        for c in all_codes
    }

    # 3. Keyword boosts (verbatim from hybridRAG/test2.ipynb)
    q = (query or "").lower()
    if "беремен" in q:
        for c in list(combined):
            if c.startswith("O"):
                combined[c] *= 2
    if "давление" in q or "гипертенз" in q:
        for c in list(combined):
            if c.startswith("O14"):
                combined[c] *= 3
    if "hellp" in q:
        for c in list(combined):
            if c.startswith("O14.2"):
                combined[c] *= 4

    ranked = sorted(combined.items(), key=lambda x: -x[1])[:cap]
    return [
        {"code": code,
         "description": icd_desc.get(code) or icd_desc.get(code.split(".")[0], "(описание недоступно)"),
         "titles": code_titles.get(code, []),
         "score": round(score, 6)}
        for code, score in ranked
    ]


def node_select(query, top5, candidates, protocols):
    ctx = []
    for i, (pi, _) in enumerate(top5[:3]):
        p = protocols[pi]
        title = (p.get("source_file") or "").replace(".pdf", "")
        ctx.append(f"=== Протокол {i+1}: {title} ===\n{p['text'][:1500]}")
    groups = {}
    for c in candidates:
        groups.setdefault(c["code"].split(".")[0], []).append(c)
    cand_lines = []
    for i, c in enumerate(candidates):
        pfx = c["code"].split(".")[0]
        gsize = len(groups[pfx])
        sib_note = f"  [{gsize} кода в группе {pfx}]" if gsize > 1 else ""
        cand_lines.append(f"{i+1}. {c['code']} — {c['description']}{sib_note}")
    sys_msg = (
        "Ты — эксперт по кодированию МКБ-10. Твоя задача — точно выбрать ОДИН код из списка. "
        "Особенно внимательно различай близкие коды в одной группе (X22.0 vs X22.1 vs X22.9) — "
        "мелкие различия в описании важны."
    )
    user_msg = (
        f"Жалоба пациента:\n{query}\n\n"
        f"Контекст клинических протоколов:\n" + "\n\n".join(ctx) + "\n\n"
        f"Кандидаты МКБ-10:\n" + "\n".join(cand_lines) + "\n\n"
        f"ШАГ 1. ДИАГНОЗ: Сформулируй наиболее вероятный клинический диагноз (1-2 предложения).\n\n"
        f"ШАГ 2. БЛИЖАЙШИЕ КОДЫ: Выбери 2-4 ближайших к диагнозу кода. "
        f"Если есть группа соседей (J04.0, J04.1, J04.2), включи ВСЕХ.\n\n"
        f"ШАГ 3. СРАВНЕНИЕ: Для каждого ближайшего кода — ПОЧЕМУ подходит или нет. "
        f"Какое слово совпадает/не совпадает с симптомами.\n\n"
        f"ШАГ 4. ВЫБОР: Какой код точнее. 1 предложение — обоснование vs ближайший сосед.\n\n"
        f"ШАГ 5. В ПОСЛЕДНЕЙ строке: FINAL: <CODE>\n\n"
        f"Код ОБЯЗАТЕЛЬНО из списка кандидатов."
    )
    raw = llm([{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
             max_tokens=900, temperature=0.0)
    cand_set = {c["code"] for c in candidates}
    for line in raw.strip().splitlines()[::-1]:
        m = re.search(r"FINAL\s*:?\s*([A-Z]\d{2}(?:\.\d{1,2})?(?:[A-Z0-9]{1,3})?)", line)
        if m and m.group(1) in cand_set:
            return m.group(1), raw
    for c in reversed(ICD_CODE_RE.findall(raw)):
        if c in cand_set:
            return c, raw
    return (candidates[0]["code"] if candidates else None), raw


# ---------- LangGraph (CRAG retry loop, identical shape to rag_complex) ----------
class PState(TypedDict, total=False):
    query: str
    search_query: str
    hyde_text: str
    top5: list
    chunk_hits: list
    retries: int
    retrieval_history: list
    grader_verdict: str
    grader_reason: str
    candidates: list
    pred: str
    select_raw: str


MAX_RETRIES = 2


def build_graph(deps):
    from langgraph.graph import END, StateGraph
    protocols = deps["protocols"]
    chunks = deps["chunks"]
    faiss_index = deps["faiss_index"]
    dense_model = deps["dense_model"]
    icd_graph = deps["icd_graph"]
    icd_desc = deps["icd_desc"]

    def s_rewrite(state):
        return {"search_query": node_rewrite(state["query"]),
                "retries": 0, "retrieval_history": []}

    def s_hyde(state):
        return {"hyde_text": node_hyde(state["query"])}

    def s_retrieve(state):
        top5, chunk_hits = retrieve(state["search_query"], state.get("hyde_text", ""),
                                    dense_model=dense_model, faiss_index=faiss_index,
                                    chunks=chunks, protocols=protocols)
        titles = [(protocols[pi].get("source_file") or "").replace(".pdf", "") for pi, _ in top5]
        history = state.get("retrieval_history", []) + [
            {"query": state["search_query"], "titles": titles}]
        return {"top5": top5, "chunk_hits": chunk_hits, "retrieval_history": history}

    def s_grade(state):
        if state.get("retries", 0) >= MAX_RETRIES:
            return {"grader_verdict": "GOOD", "grader_reason": "max retries"}
        verdict, reason = node_grade(state["query"], state["top5"], protocols)
        return {"grader_verdict": verdict, "grader_reason": reason}

    def s_rewrite_retry(state):
        nq = node_rewrite_with_context(state["query"], state["retrieval_history"],
                                       state.get("grader_reason", ""))
        return {"search_query": nq, "retries": state.get("retries", 0) + 1}

    def s_candidates(state):
        cands = collect_candidates_hybrid(state["query"], state["chunk_hits"],
                                          chunks, protocols, icd_graph, icd_desc)
        return {"candidates": cands}

    def s_select(state):
        if not state["candidates"]:
            return {"pred": None, "select_raw": ""}
        pred, raw = node_select(state["query"], state["top5"], state["candidates"], protocols)
        return {"pred": pred, "select_raw": raw}

    def route(state):
        return ("rewrite_retry"
                if state["grader_verdict"] == "BAD" and state.get("retries", 0) < MAX_RETRIES
                else "candidates")

    b = StateGraph(PState)
    b.add_node("rewrite", s_rewrite)
    b.add_node("hyde", s_hyde)
    b.add_node("retrieve", s_retrieve)
    b.add_node("grade", s_grade)
    b.add_node("rewrite_retry", s_rewrite_retry)
    b.add_node("candidates", s_candidates)
    b.add_node("select", s_select)
    b.set_entry_point("rewrite")
    b.add_edge("rewrite", "hyde")
    b.add_edge("hyde", "retrieve")
    b.add_edge("retrieve", "grade")
    b.add_conditional_edges("grade", route,
                            {"rewrite_retry": "rewrite_retry", "candidates": "candidates"})
    b.add_edge("rewrite_retry", "retrieve")
    b.add_edge("candidates", "select")
    b.add_edge("select", END)
    return b.compile()


# ---------- Eval ----------
def run_one(graph, protocols, test_file):
    t0 = time.time()
    try:
        test = json.load(open(test_file, encoding="utf-8"))
    except Exception as e:
        return {"test_file": test_file.name, "error": f"load: {e!r}", "correct": False,
                "gt": None, "gt_protocol_id": None, "pred": None, "top5_pids": [],
                "candidates": [], "retries": 0, "duration_s": 0.0, "query_snippet": ""}
    q, gt = test.get("query"), test.get("gt")
    if not q or not isinstance(q, str):
        return {"test_file": test_file.name, "error": "empty query", "correct": False,
                "gt": gt, "gt_protocol_id": test.get("protocol_id"), "pred": None,
                "top5_pids": [], "candidates": [], "retries": 0,
                "duration_s": 0.0, "query_snippet": ""}
    pred, top5, cands, retries, err = None, [], [], 0, None
    try:
        state = graph.invoke({"query": q})
        pred = state.get("pred")
        top5 = state.get("top5", []) or []
        cands = state.get("candidates", []) or []
        retries = state.get("retries", 0)
    except Exception as e:
        err = repr(e)
    return {
        "test_file": test_file.name,
        "gt": gt,
        "gt_protocol_id": test.get("protocol_id"),
        "pred": pred,
        "correct": bool(pred and gt and pred == gt),
        "top5_pids": [protocols[pi]["protocol_id"] for pi, _ in top5],
        "candidates": [c["code"] for c in cands][:25],
        "retries": retries,
        "duration_s": round(time.time() - t0, 2),
        "query_snippet": q[:220],
        "error": err,
    }


def run_eval(graph, protocols, test_files, *, workers=4):
    done = set()
    if EVAL_OUT.exists():
        with open(EVAL_OUT, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["test_file"])
                except Exception:
                    pass
    todo = [f for f in test_files if f.name not in done]
    print(f"Resuming: {len(done)} done, {len(todo)} to process")
    with open(EVAL_OUT, "a", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(run_one, graph, protocols, f): f for f in todo}
            for fut in tqdm(as_completed(futs), total=len(futs)):
                try:
                    row = fut.result()
                except Exception as e:
                    fp = futs[fut]
                    row = {"test_file": fp.name, "error": f"future: {e!r}",
                           "correct": False, "gt": None, "gt_protocol_id": None, "pred": None,
                           "top5_pids": [], "candidates": [], "retries": 0,
                           "duration_s": 0.0, "query_snippet": ""}
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()


def compute_metrics():
    results = [json.loads(l) for l in open(EVAL_OUT, encoding="utf-8")]
    n = len(results)
    if n == 0:
        return
    strict = sum(r["correct"] for r in results)
    lenient = sum(1 for r in results if r["pred"] and r["gt"]
                  and r["pred"].split(".")[0] == r["gt"].split(".")[0])
    proto = sum(1 for r in results if r.get("gt_protocol_id") and r["gt_protocol_id"] in r.get("top5_pids", []))
    cand = sum(1 for r in results if r["gt"] in r.get("candidates", []))
    recall3 = mrr = ndcg = 0.0
    nw = 0
    for r in results:
        gt = r.get("gt")
        if not gt:
            continue
        nw += 1
        pred = r.get("pred")
        cands = r.get("candidates", []) or []
        ranked = ([pred] if pred else []) + [c for c in cands if c != pred]
        if gt in ranked[:3]:
            recall3 += 1
        if gt in ranked:
            rk = ranked.index(gt) + 1
            mrr += 1.0 / rk
            if rk <= 10:
                ndcg += 1.0 / math.log2(rk + 1)
    print(f"\nN = {n}")
    print(f"Strict@1         {strict/n:.4f}  ({strict}/{n})")
    print(f"Lenient@1        {lenient/n:.4f}  ({lenient}/{n})")
    print(f"Recall@3         {recall3/nw:.4f}")
    print(f"MRR              {mrr/nw:.4f}")
    print(f"NDCG@10          {ndcg/nw:.4f}")
    print(f"Protocol rec@5   {proto/n:.4f}  ({proto}/{n})")
    print(f"Candidate recall {cand/n:.4f}  ({cand}/{n})")
    summary = {
        "n": n,
        "strict1": strict / n,
        "lenient1": lenient / n,
        "recall3": recall3 / nw if nw else 0.0,
        "mrr": mrr / nw if nw else 0.0,
        "ndcg10": ndcg / nw if nw else 0.0,
        "protocol_recall_at_5": proto / n,
        "candidate_recall": cand / n,
    }
    json.dump(summary, open(SUMMARY_OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nWrote {SUMMARY_OUT}")


def main():
    print("Ping:", client.chat.completions.create(
        model=GEMMA_MODEL, messages=[{"role": "user", "content": "ping"}],
        max_tokens=5, temperature=0,
    ).choices[0].message.content)
    protocols, chunks = load_corpus()
    print(f"{len(protocols)} protocols, {len(chunks)} chunks")
    icd_desc = extract_icd_descriptions(protocols)
    print(f"ICD descriptions (corpus-harvested): {len(icd_desc)}")
    dense_model, chunk_embs = build_dense_index(chunks)
    print(f"E5 chunk embeddings: {chunk_embs.shape}")
    faiss_index = build_faiss_index(chunk_embs)
    print(f"FAISS chunk index: {faiss_index.ntotal} vectors")
    icd_graph = build_icd_graph(protocols)
    print(f"ICD graph: {icd_graph.number_of_nodes()} nodes, {icd_graph.number_of_edges()} edges")
    deps = dict(
        protocols=protocols, chunks=chunks, faiss_index=faiss_index,
        dense_model=dense_model, icd_graph=icd_graph, icd_desc=icd_desc,
    )
    graph = build_graph(deps)
    run_eval(graph, protocols, test_files, workers=4)
    compute_metrics()


if __name__ == "__main__":
    main()
