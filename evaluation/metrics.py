"""Basic lexical/statistical proxies adapted from the original RAG evaluation tools."""
import math
import re
from collections import Counter


# ──────────────────────────────────────────────────────────────────────────────
# 1. Distinct-1 / Distinct-2  (Li et al., 2016)
# ──────────────────────────────────────────────────────────────────────────────

def distinct_n(utterances: list[str], n: int) -> float:
    """
    Distinct-N: 전체 토큰 쌍 중 고유 n-gram 비율.
    값이 높을수록 어휘/표현 다양성이 높음.
    reference: Li et al. (2016) "A Diversity-Promoting Objective Function
               for Neural Conversation Models", NAACL-HLT.
    """
    if n < 1:
        raise ValueError("n must be positive")
    all_ngrams: list[tuple] = []
    for text in utterances:
        tokens = text.lower().split()
        all_ngrams.extend(zip(*[tokens[i:] for i in range(n)]))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Turn-Taking Entropy
# ──────────────────────────────────────────────────────────────────────────────

def turn_entropy(speakers: list[str]) -> float:
    """
    Shannon entropy of speaker distribution.
    최대값: log2(N)  — 모든 참가자가 동등하게 발언
    최소값: 0        — 1명이 독점
    reference: Shannon (1948); 대화 분석 연구에서 표준적으로 사용.
    """
    if not speakers:
        return 0.0
    freq = Counter(speakers)
    total = len(speakers)
    return -sum((c / total) * math.log2(c / total) for c in freq.values())


def normalized_turn_entropy(speakers: list[str]) -> float:
    """[0, 1] 범위로 정규화된 엔트로피. 1 = 완전히 균등한 분배."""
    n_speakers = len(set(speakers))
    if n_speakers <= 1:
        return 0.0
    return turn_entropy(speakers) / math.log2(n_speakers)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Conversation Move Diversity
# ──────────────────────────────────────────────────────────────────────────────

VALID_MOVES = {
    "AGREE_AND_EXTEND", "DISAGREE_SPECIFIC", "ASK_QUESTION",
    "SHARE_ANECDOTE", "REDIRECT", "BUILD_ON", "REACT_EMOTIONALLY",
}

def move_diversity(moves: list[str]) -> dict:
    """
    conversation move 분포의 Shannon entropy + 각 move 비율.
    논문에서 "대화 행위 다양성(Dialogue Act Diversity)" 지표로 보고.
    """
    valid = [m for m in moves if m in VALID_MOVES]
    if not valid:
        return {"entropy": 0.0, "normalized": 0.0, "distribution": {}}
    freq = Counter(valid)
    total = len(valid)
    h = -sum((c / total) * math.log2(c / total) for c in freq.values())
    norm = h / math.log2(len(VALID_MOVES))  # 최대 log2(7)로 정규화
    return {
        "entropy": round(h, 4),
        "normalized": round(norm, 4),
        "distribution": {k: round(v / total, 3) for k, v in freq.most_common()},
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. Direct Address Rate
# ──────────────────────────────────────────────────────────────────────────────

def direct_address_rate(utterances: list[str], agent_names: list[str]) -> float:
    """
    발화 중 다른 참가자 이름을 직접 언급하는 비율.
    '개별 미니 에세이 나열' 문제(problem.md §5)를 정량화.
    reference: Levinson (1983) Pragmatics; 대화 분석에서 addressee 지정 연구.
    """
    if not utterances:
        return 0.0
    # 화자 이름(콜론 앞)은 제외하고 발화 본문만 검색
    bodies = list(utterances)
    n_direct = sum(
        1 for body in bodies
        if any(re.search(r"(?<!\w)" + re.escape(name.replace("_", " ")) + r"(?!\w)",
                         body.replace("_", " "), re.IGNORECASE) for name in agent_names)
    )
    return round(n_direct / len(bodies), 4)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Rebuttal Rate
# ──────────────────────────────────────────────────────────────────────────────

def rebuttal_rate(moves: list[str]) -> float:
    """
    DISAGREE_SPECIFIC move 비율.
    '실질적 충돌 없는 안전한 의견 교환' 문제(discussion_problem.md §3)를 정량화.
    reference: Walton (1996) "Argumentation Schemes for Presumptive Reasoning."
    """
    if not moves:
        return 0.0
    valid = [m for m in moves if m in VALID_MOVES]
    if not valid:
        return 0.0
    return round(valid.count("DISAGREE_SPECIFIC") / len(valid), 4)


# ──────────────────────────────────────────────────────────────────────────────
# 6. Q→A Chain Completion Rate
# ──────────────────────────────────────────────────────────────────────────────

def qa_chain_completion(moves: list[str]) -> float:
    """
    ASK_QUESTION 이후 다음 발화가 BUILD_ON / AGREE_AND_EXTEND / DISAGREE_SPECIFIC
    (질문에 응답하는 행위) 중 하나인 비율.
    reference: van Kuppevelt (1995) "Discourse structure, topicality and
               questioning", Journal of Linguistics.
    """
    if len(moves) < 2:
        return 0.0
    answer_moves = {"BUILD_ON", "AGREE_AND_EXTEND", "DISAGREE_SPECIFIC",
                    "SHARE_ANECDOTE", "REACT_EMOTIONALLY"}
    total_q = sum(1 for m in moves if m == "ASK_QUESTION")
    if total_q == 0:
        return 0.0  # 질문이 없으면 측정 불가 → 별도 표기
    answered = sum(
        1 for i, m in enumerate(moves[:-1])
        if m == "ASK_QUESTION" and moves[i + 1] in answer_moves
    )
    return round(answered / total_q, 4)


# ──────────────────────────────────────────────────────────────────────────────
# 7. Topic Drift Score  (TF-IDF 기반 rolling cosine similarity)
# ──────────────────────────────────────────────────────────────────────────────

def _tfidf_vector(text: str, idf: dict[str, float]) -> dict[str, float]:
    tokens = re.findall(r"[a-z]+", text.lower())
    tf = Counter(tokens)
    total = max(1, sum(tf.values()))
    return {w: (tf[w] / total) * idf.get(w, 1.0) for w in tf}


def _cosine(v1: dict, v2: dict) -> float:
    keys = set(v1) | set(v2)
    dot = sum(v1.get(k, 0) * v2.get(k, 0) for k in keys)
    n1 = math.sqrt(sum(x ** 2 for x in v1.values()))
    n2 = math.sqrt(sum(x ** 2 for x in v2.values()))
    return dot / (n1 * n2) if n1 * n2 > 0 else 0.0


def topic_drift_score(utterances: list[str], window: int = 3) -> dict:
    """
    연속 발화 윈도우 간 TF-IDF cosine similarity 평균.
    - 높을수록 주제 일관성 유지 (drift 없음)
    - 낮을수록 토픽이 빠르게 이동 (discussion_problem.md §1 문제)

    window: 각 비교 단위의 발화 수
    reference: Manning & Schütze (1999) "Foundations of Statistical NLP."
    """
    if len(utterances) < window * 2:
        return {"mean_similarity": None, "min_similarity": None,
                "drift_variance": None, "note": "insufficient data"}

    bodies = list(utterances)

    # IDF 계산
    all_tokens = [re.findall(r"[a-z]+", b.lower()) for b in bodies]
    doc_freq: Counter = Counter()
    for tokens in all_tokens:
        doc_freq.update(set(tokens))
    N = len(bodies)
    idf = {w: math.log(N / (df + 1)) + 1 for w, df in doc_freq.items()}

    sims: list[float] = []
    for i in range(len(bodies) - window):
        chunk1 = " ".join(bodies[i: i + window])
        chunk2 = " ".join(bodies[i + 1: i + 1 + window])
        v1 = _tfidf_vector(chunk1, idf)
        v2 = _tfidf_vector(chunk2, idf)
        sims.append(_cosine(v1, v2))

    mean_sim = sum(sims) / len(sims)
    variance = sum((s - mean_sim) ** 2 for s in sims) / len(sims)

    return {
        "mean_similarity": round(mean_sim, 4),
        "min_similarity": round(min(sims), 4),
        "drift_variance": round(variance, 4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 8. Grounding Score  (책 원문 근거 언급 비율)
# ──────────────────────────────────────────────────────────────────────────────

# 구체적 장면/인물 참조를 나타내는 패턴 (정규식)
_GROUNDING_PATTERNS = [
    r"\bscene\b", r"\bchapter\b", r"\bpassage\b",
    r"\bwhen\b.{0,30}\b(she|he|they|pinocchio|cinderella|prince|geppetto|fairy)\b",
    r"\"[^\"]{5,80}\"",        # 인용 부호 안 텍스트
    r"'[^']{5,80}'",
    r"\bin the (book|story|text|novel)\b",
    r"\bspecific(ally)?\b",
    r"\bexact(ly)?\b",
]
_GROUNDING_RE = re.compile("|".join(_GROUNDING_PATTERNS), re.IGNORECASE)


def grounding_score(utterances: list[str], rag_contexts: list[str]) -> dict:
    """
    발화가 RAG가 제공한 원문 구절 또는 구체적 장면 참조를 포함하는 비율.
    - rag_contexts: 각 발화에 대응하는 retrieve() 결과
    '텍스트 근거 없는 추상적 발언(Evidence-free discussion)' 문제를
    정량화 (discussion_problem.md §2).

    두 가지 하위 지표:
    - pattern_grounded: 정규식으로 구체적 언급 감지
    - rag_overlap: 발화 단어가 RAG 구절에 등장하는 비율 (어휘 겹침)
    """
    if len(utterances) != len(rag_contexts):
        raise ValueError("Each utterance needs its matching retrieval context")
    n = len(utterances)
    if n == 0:
        return {"pattern_grounded_rate": 0.0, "rag_overlap_rate": 0.0}

    bodies = list(utterances)
    pattern_count = sum(1 for b in bodies if _GROUNDING_RE.search(b))

    overlap_scores: list[float] = []
    for body, ctx in zip(bodies, rag_contexts):
        if not ctx:
            overlap_scores.append(0.0)
            continue
        body_words = set(re.findall(r"[a-z]+", body.lower())) - _STOPWORDS
        ctx_words  = set(re.findall(r"[a-z]+", ctx.lower()))
        if not body_words:
            overlap_scores.append(0.0)
        else:
            overlap_scores.append(len(body_words & ctx_words) / len(body_words))

    return {
        "pattern_grounded_rate": round(pattern_count / n, 4),
        "rag_overlap_rate": round(sum(overlap_scores) / len(overlap_scores), 4),
    }


_STOPWORDS = {
    "the", "a", "an", "is", "it", "in", "of", "and", "to", "that",
    "this", "was", "are", "for", "on", "with", "as", "at", "by",
    "from", "or", "but", "not", "be", "have", "had", "her", "his",
    "she", "he", "they", "we", "you", "i", "me", "my", "our", "their",
    "been", "has", "do", "did", "can", "will", "would", "could",
    "should", "which", "who", "what", "when", "where", "how", "if",
}


# ──────────────────────────────────────────────────────────────────────────────
# 9. Lexical Diversity (Type-Token Ratio)
# ──────────────────────────────────────────────────────────────────────────────

def lexical_diversity(utterances: list[str]) -> float:
    """
    Type-Token Ratio (TTR): 전체 어휘 중 고유 단어 비율.
    긴 텍스트에서는 Moving-Average TTR (MATTR) 로 보완 가능.
    reference: Templin (1957); Richards (1987) "The context of language learning."
    """
    all_tokens = []
    for u in utterances:
        body = u.split(":", 1)[-1] if ":" in u else u
        all_tokens.extend(re.findall(r"[a-z]+", body.lower()))
    if not all_tokens:
        return 0.0
    return round(len(set(all_tokens)) / len(all_tokens), 4)


def moving_average_ttr(utterances: list[str], window: int = 50) -> float:
    """
    MATTR: 텍스트 길이에 덜 민감한 TTR 변형.
    reference: Covington & McFall (2010) "Cutting the Gordian knot: The
               moving-average type-token ratio (MATTR)", JQLA.
    """
    all_tokens: list[str] = []
    for u in utterances:
        body = u.split(":", 1)[-1] if ":" in u else u
        all_tokens.extend(re.findall(r"[a-z]+", body.lower()))
    if len(all_tokens) < window:
        return lexical_diversity(utterances)
    ttrs = [
        len(set(all_tokens[i: i + window])) / window
        for i in range(len(all_tokens) - window + 1)
    ]
    return round(sum(ttrs) / len(ttrs), 4)


# ──────────────────────────────────────────────────────────────────────────────
# 10. Average Response Length
# ──────────────────────────────────────────────────────────────────────────────

def avg_response_length(utterances: list[str]) -> dict:
    """발화 길이 통계 (단어 수 기준)."""
    if not utterances:
        return {"mean": 0, "min": 0, "max": 0, "std": 0.0}
    bodies = list(utterances)
    lengths = [len(b.split()) for b in bodies]
    mean_l = sum(lengths) / len(lengths)
    variance = sum((l - mean_l) ** 2 for l in lengths) / len(lengths)
    return {
        "mean": round(mean_l, 1),
        "min": min(lengths),
        "max": max(lengths),
        "std": round(variance ** 0.5, 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 통합 계산 함수
# ──────────────────────────────────────────────────────────────────────────────

