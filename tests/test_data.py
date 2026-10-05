import json
from pathlib import Path

from zengziagent.data.elife import extract_reviewer_reports, parse_article
from zengziagent.data.units import build_units, corpus_statistics, dataset_hash, load_records_jsonl
from zengziagent.schema import canonical_label

FIXTURE_XML = b"""<?xml version="1.0"?><article><front><article-meta>
<article-id pub-id-type="publisher-id">12345</article-id><article-id pub-id-type="doi">10.7554/eLife.12345</article-id>
<title-group><article-title>Test article</article-title></title-group>
<pub-date date-type="pub"><day>1</day><month>2</month><year>2018</year></pub-date></article-meta></front>
<body/><sub-article article-type="decision-letter" id="SA1"><front-stub><title-group><article-title>Decision letter</article-title></title-group></front-stub>
<body><p>In the interests of transparency, eLife includes the editorial decision letter.</p><p>Thank you for submitting your article.</p>
<p>Reviewer #1:</p><p>The manuscript is interesting. However, the data are weak.</p><p>Reviewer #2:</p><p>Overall this is a solid study.</p>
<p>Reviewer #3 (General assessment): The statistics need work.</p></body></sub-article>
<sub-article article-type="reply" id="SA2"><body><p>We thank the reviewers.</p></body></sub-article></article>"""


def test_canonical_labels():
    assert canonical_label("Jus_neg_3") == ("Jus_neg", 3)
    assert canonical_label("Major_claim") == ("Major_Claim", None)
    assert canonical_label("negative evidence") == ("Jus_neg", None)
    assert canonical_label("Eval_neutral") == (None, None)


def test_substanreview_style_loader(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    row = {"id": 7, "review": "Good paper. Weak eval.", "label": [[0, 11, "Eval_pos_1"], [12, 22, "Eval_neg_1"], [0, 4, "Bogus_9"]]}
    p.write_text(json.dumps(row) + "\n")
    recs = load_records_jsonl(p, dataset="substanreview")
    assert len(recs) == 1 and [g.label for g in recs[0].gold_spans] == ["Eval_pos", "Eval_neg"]
    assert recs[0].gold_spans[0].occurrence == 1
    units = build_units(recs[0], "gold_span")
    assert len(units) == 2 and units[1].text == "Weak eval."
    sent_units = build_units(recs[0], "sentence")
    assert [u.gold_label for u in sent_units] == ["Eval_pos", "Eval_neg"]
    stats = corpus_statistics(recs)
    assert stats["total_units"] == 2 and stats["label_counts"]["Eval_pos"] == 1
    assert len(dataset_hash(recs)) == 64


def test_elife_extraction():
    art = parse_article(FIXTURE_XML)
    assert art["article_id"] == "12345" and art["year"] == 2018
    reports = extract_reviewer_reports(art)
    by_rev = {r.reviewer: r for r in reports}
    assert set(by_rev) == {"1", "2", "3"}
    assert by_rev["1"].text.startswith("The manuscript is interesting.")
    assert by_rev["3"].text == "The statistics need work."
    assert all(r.sub_article_type == "decision-letter" and r.round_index == 1 for r in reports)
    assert not any("thank you for submitting" in r.text.lower() for r in reports)


def test_canonical_layout_is_validated_and_whitespace_trimmed(tmp_path: Path):
    p = tmp_path / "gold.jsonl"
    row = {"review_id": "elife-1", "dataset": "elife", "text": "Good paper. Weak eval. ", "gold_spans": [{"start": 0, "end": 12, "label": "Eval_pos_1"}, {"start": 11, "end": 23, "label": "Major_claim"}, {"start": 0, "end": 4, "label": "Bogus"}], "metadata": {}}
    p.write_text(json.dumps(row) + "\n")
    recs = load_records_jsonl(p, dataset="elife")
    spans = recs[0].gold_spans
    assert [g.label for g in spans] == ["Eval_pos", "Major_Claim"]
    assert (spans[0].start, spans[0].end) == (0, 11) and recs[0].text[spans[0].start:spans[0].end] == "Good paper."
    assert (spans[1].start, spans[1].end) == (12, 22)
    assert spans[0].occurrence == 1


def test_iaa_layout_loads(tmp_path: Path):
    p = tmp_path / "iaa.jsonl"
    row = {"id": 3, "text": "Nice work. ", "rid": "r9", "scores": [1, 2], "label": [[0, 11, "Eval_pos_1"]], "Comments": ""}
    p.write_text(json.dumps(row) + "\n")
    recs = load_records_jsonl(p, dataset="substanreview-iaa")
    assert recs[0].text == "Nice work. " and recs[0].metadata["rid"] == "r9"
    assert (recs[0].gold_spans[0].start, recs[0].gold_spans[0].end) == (0, 10)
