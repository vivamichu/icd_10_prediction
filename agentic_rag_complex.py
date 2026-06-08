"""Complex config — RAG with HyDE and CRAG retry loop.

Pipeline (up to 9 stages, ~4-7 LLM calls per query):
    rewrite → HyDE → retrieve (dense-query + dense-HyDE + BM25, RRF, reranker)
          → grade → [if BAD and retries<2: rewrite_with_context → retrieve]
          → collect candidates → CoT select

Run: `python rag_complex.py`
"""
import json
import math
import os
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
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
EVAL_OUT = BASE_DIR / "eval_results_complex.jsonl"
SUMMARY_OUT = BASE_DIR / "eval_summary_complex.json"

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

# ---------- Corpus + ICD descriptions ----------
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150


def _chunk(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        if i + size >= len(text):
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
        for ci, ch in enumerate(_chunk(p["text"])):
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


# ---------- BM25 ----------
TOKEN_RE = re.compile(r"[А-Яа-яЁёA-Za-z0-9]+(?:\.\d+)?")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, texts, k1=1.5, b=0.75):
        from sklearn.feature_extraction.text import CountVectorizer
        self.k1, self.b = k1, b
        self.vec = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        self.X = self.vec.fit_transform(texts)
        self.N = self.X.shape[0]
        self.doc_lens = np.asarray(self.X.sum(axis=1)).flatten().astype(np.float32)
        self.avgdl = float(self.doc_lens.mean()) if self.N > 0 else 1.0
        df = np.asarray((self.X > 0).sum(axis=0)).flatten().astype(np.float32)
        self.idf = np.log((self.N - df + 0.5) / (df + 0.5) + 1).astype(np.float32)

    def top_k(self, query, k=30):
        q_toks = tokenize(query)
        vocab = self.vec.vocabulary_
        q_ids = [vocab[t] for t in q_toks if t in vocab]
        if not q_ids:
            return []
        X_q = self.X[:, q_ids].toarray().astype(np.float32)
        idf_q = self.idf[q_ids]
        K = self.k1 * (1 - self.b + self.b * self.doc_lens / self.avgdl)
        scores = ((X_q * (self.k1 + 1) / (X_q + K[:, None])) * idf_q).sum(axis=1)
        if scores.max() == 0:
            return []
        idx = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]


# ---------- Dense index + reranker ----------
def build_dense_index(chunks):
    from sentence_transformers import SentenceTransformer
    cache = CACHE_DIR / "bge_m3_chunks.npy"
    model = SentenceTransformer("BAAI/bge-m3", device=DEVICE)
    if cache.exists():
        embs = np.load(cache)
        if embs.shape[0] == len(chunks):
            return model, embs
    texts = [c["text"] for c in chunks]
    embs = model.encode(texts, batch_size=16, show_progress_bar=True,
                        normalize_embeddings=True, convert_to_numpy=True)
    np.save(cache, embs)
    return model, embs


def build_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder("BAAI/bge-reranker-v2-m3", device=DEVICE, max_length=1024)


def dense_search(chunk_embs, query_emb, k=30):
    sims = chunk_embs @ query_emb
    idx = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
    idx = idx[np.argsort(-sims[idx])]
    return [(int(i), float(sims[i])) for i in idx]


def chunks_to_protocols(chunks, chunk_hits):
    best = {}
    for ci, score in chunk_hits:
        pi = chunks[ci]["protocol_idx"]
        if pi not in best or score > best[pi]:
            best[pi] = score
    return sorted(best.items(), key=lambda x: -x[1])


def rrf_fuse(ranked_lists, k0=60, top_k=20):
    rrf = {}
    for ranks in ranked_lists:
        for rank, (pi, _) in enumerate(ranks):
            rrf[pi] = rrf.get(pi, 0) + 1.0 / (k0 + rank + 1)
    return sorted(rrf.items(), key=lambda x: -x[1])[:top_k]


def rerank(reranker_model, protocols, query, cand_idxs, top_n=5):
    if not cand_idxs:
        return []
    pairs = []
    for pi in cand_idxs:
        p = protocols[pi]
        title = (p.get("source_file") or "").replace(".pdf", "")
        pairs.append([query, f"{title}\n{p['text'][:2000]}"])
    scores = reranker_model.predict(pairs, batch_size=8, show_progress_bar=False)
    order = np.argsort(-np.asarray(scores))[:top_n]
    return [(cand_idxs[int(i)], float(scores[int(i)])) for i in order]


def retrieve(search_query, hyde_text, *, dense_model, chunk_embs, chunks, bm25,
             reranker_model, protocols):
    """Hybrid dense(query) + dense(HyDE) + BM25(query) → RRF → rerank."""
    qe = dense_model.encode([search_query], normalize_embeddings=True, convert_to_numpy=True)[0]
    dense_q = chunks_to_protocols(chunks, dense_search(chunk_embs, qe, k=30))
    dense_h = []
    if hyde_text:
        he = dense_model.encode([hyde_text], normalize_embeddings=True, convert_to_numpy=True)[0]
        dense_h = chunks_to_protocols(chunks, dense_search(chunk_embs, he, k=30))
    sparse_hits = bm25.top_k(search_query, k=30)
    fused_inputs = [dense_q, sparse_hits] + ([dense_h] if dense_h else [])
    fused = rrf_fuse(fused_inputs, top_k=20)
    cand = [pi for pi, _ in fused]
    return rerank(reranker_model, protocols, search_query, cand, top_n=5)


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


# ---------- Agentic nodes ----------
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
    """HyDE — hypothetical clinical protocol excerpt (bridges patient-speak → doctor-speak)."""
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
    """Grader — do retrieved protocols match the complaint? JSON verdict."""
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
    """On retry: produce a NEW search_query given what already failed."""
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


def collect_candidates(top5, protocols, icd_desc, cap=40):
    srcs = {}
    for pi, _ in top5:
        p = protocols[pi]
        title = (p.get("source_file") or "").replace(".pdf", "")
        for c in (p.get("icd_codes") or []):
            srcs.setdefault(c, []).append(title)
        for c in set(ICD_CODE_RE.findall(p["text"][:10000])):
            srcs.setdefault(c, []).append(title)
    for c in list(srcs.keys()):
        seen, uniq = set(), []
        for t in srcs[c]:
            if t not in seen:
                uniq.append(t)
                seen.add(t)
        srcs[c] = uniq
    ranked = sorted(srcs.items(), key=lambda x: -len(x[1]))[:cap]
    return [
        {"code": code,
         "description": icd_desc.get(code) or icd_desc.get(code.split(".")[0], "(описание недоступно)"),
         "titles": titles}
        for code, titles in ranked
    ]


def node_select(query, top5, candidates, protocols):
    """Structured 5-step CoT selector, single shot at temperature 0."""
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


# ---------- LangGraph with CRAG retry loop ----------
class PState(TypedDict, total=False):
    query: str
    search_query: str
    hyde_text: str
    top5: list
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
    chunk_embs = deps["chunk_embs"]
    dense_model = deps["dense_model"]
    bm25 = deps["bm25"]
    reranker_model = deps["reranker_model"]
    icd_desc = deps["icd_desc"]

    def s_rewrite(state):
        return {"search_query": node_rewrite(state["query"]),
                "retries": 0, "retrieval_history": []}

    def s_hyde(state):
        return {"hyde_text": node_hyde(state["query"])}

    def s_retrieve(state):
        top5 = retrieve(state["search_query"], state.get("hyde_text", ""),
                       dense_model=dense_model, chunk_embs=chunk_embs, chunks=chunks,
                       bm25=bm25, reranker_model=reranker_model, protocols=protocols)
        titles = [(protocols[pi].get("source_file") or "").replace(".pdf", "") for pi, _ in top5]
        history = state.get("retrieval_history", []) + [
            {"query": state["search_query"], "titles": titles}]
        return {"top5": top5, "retrieval_history": history}

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
        return {"candidates": collect_candidates(state["top5"], protocols, icd_desc)}

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
        test = json.load(open(test_file))
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
        with open(EVAL_OUT) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["test_file"])
                except Exception:
                    pass
    todo = [f for f in test_files if f.name not in done]
    print(f"Resuming: {len(done)} done, {len(todo)} to process")
    with open(EVAL_OUT, "a") as fout:
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
    results = [json.loads(l) for l in open(EVAL_OUT)]
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
    print(f"BGE-M3 chunk embeddings: {chunk_embs.shape}")
    bm25 = BM25([p["text"] for p in protocols])
    print(f"BM25 index: {bm25.N} docs")
    reranker_model = build_reranker()
    print("Cross-encoder reranker loaded.")
    deps = dict(
        protocols=protocols, chunks=chunks, chunk_embs=chunk_embs,
        dense_model=dense_model, bm25=bm25, reranker_model=reranker_model,
        icd_desc=icd_desc,
    )
    graph = build_graph(deps)
    run_eval(graph, protocols, test_files, workers=4)
    compute_metrics()


if __name__ == "__main__":
    main()
