from zengziagent.preprocessor import Preprocessor, segment_sentences

SRC = "The paper is interesting,  but the experimental\r\nvalidation is insufficient. <br/>Overall, I recommend rejection – the “results” are weak.\n\n\nSecond paragraph here.\nThe method\nis novel."


def test_round_trip_mapping_preserves_characters():
    pp = Preprocessor().process(SRC)
    for i, c in enumerate(pp.text):
        if not c.isspace():
            assert SRC[pp.to_original[i]] == c
    assert "<br/>" not in pp.text
    assert "\r" not in pp.text
    assert "  " not in pp.text


def test_map_span_returns_original_offsets():
    pp = Preprocessor().process(SRC)
    idx = pp.text.index("validation is insufficient")
    s, e = pp.map_span(idx, idx + len("validation is insufficient"))
    assert SRC[s:e] == "validation is insufficient"


def test_identity_when_disabled():
    pp = Preprocessor(enabled=False).process(SRC)
    assert pp.text == SRC
    assert pp.map_span(3, 10) == (3, 10)


def test_sentence_segmentation_handles_soft_wraps():
    txt = "The method\nis novel. Second one.\n- Bullet item\nAnother line."
    sents = [txt[s:e] for s, e in segment_sentences(txt)]
    assert sents[0] == "The method\nis novel."
    assert "- Bullet item" in sents
