from zengziagent.alignment import TextAligner, exact_only_aligner, normalize_for_match
from zengziagent.tokenization import RegexTokenizer, char_span_to_token_range, token_iou

SRC = "The paper is interesting, but the experimental validation is insufficient. The paper is interesting, though."


def test_exact_match_prefers_occurrence_after_prev_end():
    al = TextAligner(tokenizer=RegexTokenizer())
    r1 = al.align(SRC, "The paper is interesting")
    assert r1.status == "exact" and r1.start == 0
    r2 = al.align(SRC, "The paper is interesting", prev_end=10)
    assert r2.status == "exact" and r2.start == SRC.index("The paper is interesting, though")


def test_hint_breaks_ties():
    al = TextAligner(tokenizer=RegexTokenizer())
    r = al.align(SRC, "The paper is interesting", hint_pos=80)
    assert r.start == SRC.index("The paper is interesting, though")


def test_normalized_match_handles_quotes_and_case():
    al = TextAligner(tokenizer=RegexTokenizer())
    src = "Overall, the “results” are weak – see Table 2."
    r = al.align(src, 'overall, the "results" are weak - see table 2.')
    assert r.status == "normalized"
    assert src[r.start : r.end] == src


def test_fuzzy_recovery_of_paraphrase_and_rejection_of_hallucination():
    al = TextAligner(tokenizer=RegexTokenizer())
    r = al.align(SRC, "experimental validation is not sufficient")
    assert r.status == "fuzzy"
    assert r.recovered_text == "experimental validation is insufficient"
    assert r.token_start is not None and r.token_end > r.token_start
    bad = al.align(SRC, "The supplementary appendix is extensive and praised unanimously.")
    assert bad.status == "rejected"


def test_exact_only_aligner_rejects_non_verbatim():
    ex = exact_only_aligner(tokenizer=RegexTokenizer())
    assert ex.align(SRC, "experimental validation is not sufficient").status == "rejected"
    assert ex.align(SRC, "Experimental validation is insufficient").status == "rejected"  # case differs
    assert ex.align(SRC, "experimental validation is insufficient").status == "exact"


def test_normalize_index_map():
    norm, idx = normalize_for_match("A  “B”\tc")
    assert norm == 'a "b" c'
    assert [("A  “B”\tc")[i] for i in idx] == ["A", " ", "“", "B", "”", "\t", "c"]


def test_token_iou():
    tok = RegexTokenizer().spans("a b c d e")
    assert token_iou(tok, (0, 5), (0, 9)) == 3 / 5
    assert token_iou(tok, (0, 1), (8, 9)) == 0.0
    assert char_span_to_token_range(tok, 2, 5) == (1, 3)
