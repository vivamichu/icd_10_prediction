"""Advanced config — rag_complex pipeline with advancedRAG retrieval swapped in.

Pipeline shell is identical to rag_complex.py (rewrite -> HyDE -> retrieve ->
grade -> [retry] -> candidates -> CoT select). Only the retrieval module is
replaced: instead of dense+BM25 RRF fusion, we use the advancedRAG approach:
    1. FAISS-indexed chunk-level dense retrieval (BGE-M3, IndexFlatIP).
    2. Cross-encoder rerank on chunk pairs.
    3. Protocol_first ICD aggregation: codes literally mentioned in retrieved
       chunk text get the full chunk score, codes only listed in protocol
       metadata get a weak (5%) contribution.

Run: `python rag_advanced.py`
"""
import json
import math
import os
import re
import time
import zipfile
from collections import defaultdict
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
EVAL_OUT = BASE_DIR / "eval_results_advanced.jsonl"
SUMMARY_OUT = BASE_DIR / "eval_summary_advanced.json"

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

# ---------- Corpus ----------
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


# ---------- Dense FAISS index + reranker (advancedRAG style) ----------
def build_dense_index(chunks):
    from sentence_transformers import SentenceTransformer
    cache = CACHE_DIR / "bge_m3_chunks.npy"
    model = SentenceTransformer("BAAI/bge-m3", device=DEVICE)
    if cache.exists():
        embs = np.load(cache)
        if embs.shape[0] == len(chunks):
            return model, embs.astype(np.float32)
    texts = [c["text"] for c in chunks]
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


def build_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder("BAAI/bge-reranker-v2-m3", device=DEVICE, max_length=1024)


def faiss_chunk_search(index, query_emb, k=40):
    q = np.ascontiguousarray(query_emb.reshape(1, -1).astype(np.float32))
    scores, idxs = index.search(q, k)
    return list(zip(idxs[0].tolist(), scores[0].tolist()))


def cross_encoder_rerank_chunks(reranker_model, chunks, query, chunk_hits, top_n=20):
    """Re-score (query, chunk_text) pairs with cross-encoder; return top_n chunk hits."""
    if not chunk_hits:
        return []
    pairs = [[query, chunks[ci]["text"]] for ci, _ in chunk_hits]
    ce_scores = reranker_model.predict(pairs, batch_size=16, show_progress_bar=False)
    order = np.argsort(-np.asarray(ce_scores))[:top_n]
    return [(chunk_hits[int(i)][0], float(ce_scores[int(i)])) for i in order]


def chunks_to_protocols(chunks, chunk_hits):
    """Aggregate per-chunk scores to per-protocol, taking the best chunk score per protocol."""
    best = {}
    for ci, score in chunk_hits:
        pi = chunks[ci]["protocol_idx"]
        if pi not in best or score > best[pi]:
            best[pi] = score
    return sorted(best.items(), key=lambda x: -x[1])


def retrieve(search_query, hyde_text, *, dense_model, faiss_index, chunks,
             reranker_model, protocols):
    """advancedRAG retrieval: chunk-level FAISS + optional HyDE merge + cross-encoder rerank.

    Returns:
      top5_protos: list[(protocol_idx, score)] - top-5 protocols (for CoT context)
      reranked_chunks: list[(chunk_idx, ce_score)] - top-20 chunks for ICD aggregation
    """
    qe = dense_model.encode([search_query], normalize_embeddings=True,
                            convert_to_numpy=True)[0].astype(np.float32)
    chunk_hits = faiss_chunk_search(faiss_index, qe, k=40)
    if hyde_text:
        he = dense_model.encode([hyde_text], normalize_embeddings=True,
                                convert_to_numpy=True)[0].astype(np.float32)
        hyde_hits = faiss_chunk_search(faiss_index, he, k=40)
        # merge by max-score per chunk
        merged = {}
        for ci, s in chunk_hits + hyde_hits:
            if ci not in merged or s > merged[ci]:
                merged[ci] = s
        chunk_hits = sorted(merged.items(), key=lambda x: -x[1])[:60]
    reranked_chunks = cross_encoder_rerank_chunks(reranker_model, chunks,
                                                  search_query, chunk_hits, top_n=20)
    top5_protos = chunks_to_protocols(chunks, reranked_chunks)[:5]
    return top5_protos, reranked_chunks


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


def collect_candidates_protocol_first(reranked_chunks, chunks, protocols, icd_desc, cap=40):
    """advancedRAG protocol_first aggregation: codes mentioned in chunk text get full
    chunk score, codes only in protocol metadata get a weak (5%) contribution.

    Returns ranked candidates with their evidence titles.
    """
    code_scores = defaultdict(float)
    code_titles = defaultdict(list)
    seen_title = defaultdict(set)
    for ci, score in reranked_chunks:
        ch = chunks[ci]
        proto = protocols[ch["protocol_idx"]]
        title = (proto.get("source_file") or "").replace(".pdf", "")
        s = max(0.0, float(score))
        proto_codes = list(proto.get("icd_codes") or [])
        text_codes = set(ICD_CODE_RE.findall(ch["text"]))
        # proto-listed codes
        for code in proto_codes:
            if code in text_codes:
                code_scores[code] += s
            else:
                code_scores[code] += s * 0.05
            if title not in seen_title[code]:
                code_titles[code].append(title)
                seen_title[code].add(title)
        # codes that appear in chunk text but aren't in proto's listed codes
        for code in text_codes - set(proto_codes):
            code_scores[code] += s * 0.5
            if title not in seen_title[code]:
                code_titles[code].append(title)
                seen_title[code].add(title)

    ranked = sorted(code_scores.items(), key=lambda x: -x[1])[:cap]
    return [
        {"code": code,
         "description": icd_desc.get(code) or icd_desc.get(code.split(".")[0], "(описание недоступно)"),
         "titles": code_titles[code],
         "score": round(score, 4)}
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
    reranked_chunks: list
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
    reranker_model = deps["reranker_model"]
    icd_desc = deps["icd_desc"]

    def s_rewrite(state):
        return {"search_query": node_rewrite(state["query"]),
                "retries": 0, "retrieval_history": []}

    def s_hyde(state):
        return {"hyde_text": node_hyde(state["query"])}

    def s_retrieve(state):
        top5, rchunks = retrieve(state["search_query"], state.get("hyde_text", ""),
                                 dense_model=dense_model, faiss_index=faiss_index,
                                 chunks=chunks, reranker_model=reranker_model,
                                 protocols=protocols)
        titles = [(protocols[pi].get("source_file") or "").replace(".pdf", "") for pi, _ in top5]
        history = state.get("retrieval_history", []) + [
            {"query": state["search_query"], "titles": titles}]
        return {"top5": top5, "reranked_chunks": rchunks, "retrieval_history": history}

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
        cands = collect_candidates_protocol_first(state["reranked_chunks"],
                                                  chunks, protocols, icd_desc)
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
    print(f"BGE-M3 chunk embeddings: {chunk_embs.shape}")
    faiss_index = build_faiss_index(chunk_embs)
    print(f"FAISS chunk index: {faiss_index.ntotal} vectors")
    reranker_model = build_reranker()
    print("Cross-encoder reranker loaded.")
    deps = dict(
        protocols=protocols, chunks=chunks, faiss_index=faiss_index,
        dense_model=dense_model, reranker_model=reranker_model,
        icd_desc=icd_desc,
    )
    graph = build_graph(deps)
    run_eval(graph, protocols, test_files, workers=4)
    compute_metrics()


if __name__ == "__main__":
    main()
