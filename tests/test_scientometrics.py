import numpy as np

from zengziagent.scientometrics.analysis import synthetic_features, top_decile_test, winsorize
from zengziagent.scientometrics.features import features_from_counts
from zengziagent.scientometrics.openalex import five_year_citations, window_coverage


def test_rank_biserial_direction():
    import pandas as pd

    df = pd.DataFrame({"citations": list(range(1, 31)), "x": list(range(1, 31))})  # top decile has the largest x
    res = top_decile_test(df, ["x"])[0]
    assert res["rank_biserial"] == 1.0 and res["median_top"] > res["median_rest"]


def test_winsor_cap_is_an_observed_count():
    x = np.array([1, 2, 3, 4, 5, 100, 250], dtype=float)
    w, cap = winsorize(x, 0.99)
    assert cap in set(x.tolist()) and float(cap).is_integer() and w.max() == cap


def test_window_coverage_and_five_year_sum():
    work = {"publication_year": 2018, "retrieved_at": "2026-09-04T00:00:00Z", "counts_by_year": [{"year": y, "cited_by_count": 10} for y in range(2017, 2027)]}
    assert window_coverage(work)[1] is True and five_year_citations(work) == 50
    old = {"publication_year": 2016, "retrieved_at": "2026-09-04T00:00:00Z", "counts_by_year": [{"year": y, "cited_by_count": 10} for y in range(2017, 2027)]}
    assert window_coverage(old)[1] is False and five_year_citations(old) is None
    # uncited work inside the horizon: OpenAlex returns no years -> complete window with 0 citations
    uncited = {"publication_year": 2018, "retrieved_at": "2026-09-04T00:00:00Z", "counts_by_year": [], "cited_by_count": 0}
    assert window_coverage(uncited) == (2018, True) and five_year_citations(uncited) == 0
    # zero citations in the publication year itself must not make the window incomplete
    late = {"publication_year": 2018, "retrieved_at": "2026-09-04T00:00:00Z", "counts_by_year": [{"year": y, "cited_by_count": 3} for y in range(2019, 2027)]}
    assert five_year_citations(late) == 12


def test_feature_formulas():
    from collections import Counter

    f = features_from_counts(Counter({"Eval_pos": 4, "Eval_neg": 6, "Jus_pos": 2, "Jus_neg": 5, "Major_Claim": 1}))
    assert f["r_substan"] == 7 / 17 and f["r_neg_val"] == 0.6 and f["r_jus"] == 7 / 18 and f["T"] == 18
    assert f["r_crit"] == 6 / 18 and f["r_neg_jus"] == 5 / 7 and f["b_pos"] == 4 / 7 and f["c_MC"] == 1
    assert len(synthetic_features(5)) == 5
