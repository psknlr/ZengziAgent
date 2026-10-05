import numpy as np

from zengziagent.evaluation.agreement import cohen_kappa, fleiss_kappa, icc_oneway, inter_run_agreement
from zengziagent.evaluation.metrics import PredSpan, aggregate, evaluate_review, metrics_from_counts
from zengziagent.evaluation.stats import holm, mcnemar, paired_bootstrap
from zengziagent.schema import GoldSpan, ReviewRecord
from zengziagent.tokenization import RegexTokenizer

TEXT = "The paper is well written. The experiments are weak because only one dataset is used. Overall I recommend rejection."
# gold: Eval_pos [0,26), Eval_neg [27,51), Jus_neg [52,85), Major_Claim [86,116)
GOLD = [GoldSpan(0, 26, "Eval_pos"), GoldSpan(27, 51, "Eval_neg"), GoldSpan(52, 85, "Jus_neg"), GoldSpan(86, 116, "Major_Claim")]
REV = ReviewRecord("r1", "toy", TEXT, GOLD)


def test_perfect_prediction():
    preds = [PredSpan(g.start, g.end, g.label, "exact") for g in GOLD]
    ev = evaluate_review(REV, preds, tokenizer=RegexTokenizer())
    assert (ev.n_units, ev.n_correct, ev.tp, ev.fp, ev.fn) == (4, 4, 4, 0, 0)
    agg = aggregate([ev])
    assert agg["f1"] == 1.0 and agg["exact_span_match_rate"] == 1.0 and agg["mean_token_iou"] == 1.0


def test_partial_overlap_label_error_and_rejected():
    preds = [
        PredSpan(0, 26, "Eval_pos", "exact"),  # TP
        PredSpan(27, 51, "Eval_pos", "exact"),  # wrong label -> FP + FN
        PredSpan(52, 68, "Jus_neg", "fuzzy"),  # partial: 'because only one' (3 of 7 tokens) -> IoU 0.43 < 0.5 -> FP + FN, but unit label correct
        PredSpan(None, None, "Major_Claim", "rejected"),  # unrecoverable -> FP
    ]
    ev = evaluate_review(REV, preds, tau=0.5, tokenizer=RegexTokenizer())
    assert ev.n_units == 4
    assert ev.n_correct == 2  # Eval_pos unit and Jus_neg unit (max-overlap label correct)
    assert ev.tp == 1
    assert ev.fp == 3  # wrong label, low-IoU span, rejected span
    assert ev.fn == 3
    assert ev.n_rejected == 1
    m = metrics_from_counts(ev.n_units, ev.n_correct, ev.tp, ev.fp, ev.fn)
    assert abs(m["f1"] - 2 * 0.25 * 0.25 / 0.5) < 1e-9
    ev2 = evaluate_review(REV, preds, tau=0.5, tokenizer=RegexTokenizer(), rejected_policy="exclude")
    assert ev2.fp == 2


def test_tau_sensitivity_changes_tp():
    preds = [PredSpan(52, 68, "Jus_neg", "fuzzy")]
    low = evaluate_review(REV, preds, tau=0.3, tokenizer=RegexTokenizer())
    high = evaluate_review(REV, preds, tau=0.9, tokenizer=RegexTokenizer())
    assert low.tp == 1 and high.tp == 0


def test_paired_bootstrap_and_mcnemar_and_holm():
    rng = np.random.default_rng(0)
    R = 60
    a = np.column_stack([np.full(R, 10), rng.integers(7, 10, R), rng.integers(7, 10, R), rng.integers(0, 3, R), rng.integers(0, 3, R)])
    b = np.column_stack([np.full(R, 10), rng.integers(4, 8, R), rng.integers(4, 8, R), rng.integers(2, 5, R), rng.integers(2, 5, R)])
    res = paired_bootstrap(a, b, "f1", n_resamples=2000, seed=1)
    assert res["delta"] > 0 and res["ci_low"] > 0 and res["p_value"] < 0.05
    same = paired_bootstrap(a, a, "accuracy", n_resamples=500, seed=1)
    assert same["delta"] == 0.0
    ca = np.array([1, 1, 1, 1, 0, 1, 1, 0, 1, 1])
    cb = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 1])
    mc = mcnemar(ca, cb)
    assert mc["b"] == 5 and mc["c"] == 0 and mc["p_exact"] < 0.1
    adj = holm([0.01, 0.04, 0.03])
    assert adj == [0.03, 0.06, 0.06]


def test_agreement_functions():
    assert abs(cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) - 1.0) < 1e-9
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0
    mat = np.array([[3, 0], [0, 3], [3, 0], [0, 3]])
    assert abs(fleiss_kappa(mat) - 1.0) < 1e-9
    ratings = np.array([[1.0, 1.1], [2.0, 2.1], [3.0, 2.9], [4.0, 4.2]])
    assert icc_oneway(ratings) > 0.9
    ira = inter_run_agreement([["a", "b", "c"], ["a", "b", "c"], ["a", "b", "d"]])
    assert abs(ira["pairwise_agreement"] - (1 + 2 / 3 + 2 / 3) / 3) < 1e-9
