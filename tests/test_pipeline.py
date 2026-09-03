from zengziagent.actor import build_prompt_bundle, load_fixed_instruction_baseline
from zengziagent.analyzer import parse_xml_annotations
from zengziagent.experiments.configs import ABLATIONS, ORDER, get_config
from zengziagent.llm.mock import MockBackend
from zengziagent.pipeline import ZengziAgentPipeline
from zengziagent.planner import Planner
from zengziagent.schema import AnnotationSchema, GoldSpan, ReviewRecord, SamplingParams
from zengziagent.utils import repo_root

SCHEMA = AnnotationSchema.from_yaml(str(repo_root() / "configs" / "schema" / "five_label.yaml"))
DEMO = ReviewRecord(
    "demo-1",
    "toy",
    "The paper is well written. The experiments are weak because only one dataset is used. Overall I recommend rejection.",
    [GoldSpan(0, 26, "Eval_pos"), GoldSpan(27, 51, "Eval_neg"), GoldSpan(52, 85, "Jus_neg"), GoldSpan(86, 116, "Major_Claim")],
)
TARGET = ReviewRecord("t-1", "toy", "The method is novel and interesting. However, the evaluation is insufficient because no baseline is reported. Overall, I recommend rejection.", [])


def test_parse_xml_blocks_and_issues():
    good = "<annotations>\n<annotation><text>The method is novel</text><index>0-19</index><label>Eval_pos</label></annotation>\n</annotations>"
    p = parse_xml_annotations(good)
    assert p.well_formed and len(p.annotations) == 1 and p.annotations[0].label == "Eval_pos" and p.annotations[0].index_hint == (0, 19)
    bad = "```xml\n<annotations>\n<annotation><text>x</text><label>Eval_neutral_2</label></annotation>\n"
    p2 = parse_xml_annotations(bad)
    assert not p2.well_formed and any(i.startswith("MISSING_ROOT_CLOSE") for i in p2.structural_issues)
    assert p2.annotations[0].label is None and p2.annotations[0].occurrence == 2
    inline = "<Eval_neg_1>the evaluation is insufficient</Eval_neg_1>"
    p3 = parse_xml_annotations(inline)
    assert p3.format == "inline_tags" and p3.annotations[0].label == "Eval_neg"
    assert parse_xml_annotations("").structural_issues[0].startswith("EMPTY_OUTPUT")


def test_planner_specification_and_generic():
    planner = Planner(SCHEMA, n_demonstrations=2)
    spec = planner.build("substanreview", [TARGET], demo_pool=[DEMO])
    assert spec.kind == "planner" and spec.label_names == list(SCHEMA.label_names)
    assert [d.review_id for d in spec.demonstrations] == ["demo-1"]
    assert spec.constraints["computed_profile"]["n_records"] == 1
    assert spec.validation_rules["allowed_labels"] == spec.label_names
    generic = Planner.generic(SCHEMA, "substanreview")
    assert generic.kind == "generic" and not generic.demonstrations and "computed_profile" not in generic.constraints
    assert spec.hash() != generic.hash()


def test_prompt_compilation_is_backend_specific_and_fib_is_fixed():
    spec = Planner(SCHEMA, 1).build("substanreview", [TARGET], demo_pool=[DEMO])
    cfg = get_config("F")
    b_anthropic = MockBackend()
    b_anthropic.family = "anthropic"
    b_openai = MockBackend()
    b_openai.family = "openai"
    pa = build_prompt_bundle(spec, b_anthropic, cfg)
    po = build_prompt_bundle(spec, b_openai, cfg)
    assert pa.prompt_hash != po.prompt_hash and "<labels>" in pa.system_prompt and "## Permitted labels" in po.system_prompt
    assert "<example>" in pa.system_prompt  # demonstrations rendered (Table 9 format)
    fib = build_prompt_bundle(spec, b_anthropic, get_config("A2"))
    assert fib.mode == "fixed" and fib.system_prompt == load_fixed_instruction_baseline()
    assert "<example>" not in fib.system_prompt and "substanreview" not in fib.system_prompt.lower()


def test_ablation_presets_are_well_defined():
    assert ORDER[0] == "F" and set(ORDER) == set(ABLATIONS)
    for cid in ORDER:
        c = ABLATIONS[cid]
        assert c.config_id == cid and c.replacement
    b0 = ABLATIONS["B0"]
    assert not (b0.use_planner or b0.use_adaptive_prompting or b0.use_preprocessing or b0.use_validation or b0.use_alignment)
    assert ABLATIONS["A6"].use_validation and not ABLATIONS["A6"].use_refinement


def _run(cfg_id, profile="noisy", seed=3):
    backend = MockBackend(profile=profile, seed=seed)
    cfg = get_config(cfg_id)
    spec = Planner(SCHEMA, 1).build("toy", [TARGET], demo_pool=[DEMO]) if cfg.use_planner else Planner.generic(SCHEMA, "toy")
    bundle = build_prompt_bundle(spec, backend, cfg)
    pipe = ZengziAgentPipeline(backend, cfg, bundle, allowed_labels=SCHEMA.label_names, sampling=SamplingParams())
    return pipe.run(TARGET, run_index=0)


def test_full_pipeline_refines_and_aligns():
    res = _run("F")
    assert res.error is None and res.round == 1
    assert all(s.status in {"exact", "normalized", "fuzzy", "rejected"} for s in res.spans)
    for s in res.spans:
        if s.recovered:
            assert TARGET.text[s.original_start : s.original_end] == s.recovered_text
    assert res.retries <= get_config("F").max_refinement_retries
    assert res.attempts[0].kind == "initial"
    if res.retries:
        assert res.attempts[1].kind == "refinement"


def test_ablations_change_processing():
    # find a seed where the noisy mock produces a paraphrase so that A5 rejects it
    for seed in range(20):
        full = _run("F", seed=seed)
        no_align = _run("A5", seed=seed)
        if any(s.status == "fuzzy" for s in full.spans):
            assert all(s.status != "fuzzy" for s in no_align.spans)
            break
    no_val = _run("A4", seed=1)
    assert no_val.retries == 0 and len(no_val.attempts) == 1
    b0 = _run("B0", seed=1)
    assert b0.preprocessing == "identity" and b0.retries == 0


def test_reflection_round():
    r1 = _run("F", seed=2)
    backend = MockBackend(profile="noisy", seed=2)
    cfg = get_config("F", reflection_rounds=2)
    spec = Planner(SCHEMA, 1).build("toy", [TARGET], demo_pool=[DEMO])
    bundle = build_prompt_bundle(spec, backend, cfg)
    pipe = ZengziAgentPipeline(backend, cfg, bundle, allowed_labels=SCHEMA.label_names, sampling=SamplingParams(), reflection_prompt="Re-check.")
    r2 = pipe.run(TARGET, run_index=0, previous_xml=r1.final_xml)
    assert r2.round == 2 and r2.attempts[0].kind == "reflection"
