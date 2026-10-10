"""The iterate loop's pure parts: reading prompts/iterate.md, the code-side
core checks, the accept/phase rules and the series order."""
import string

import pytest

from services.iterate import judge
from services.iterate.checks import check_core, head_noun, missing_must_keeps
from services.iterate.roles import validate_brief, validate_edit, validate_report
from services.iterate.runner import SERIES_MAX, _series_members, triggers_for
from services.iterate.spec import ROLES, TEMPLATES, compose_prompt, load_spec, parse_spec


# ── spec ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def spec():
    return load_spec()


def test_spec_has_every_section_and_no_unresolved_block(spec):
    assert set(spec.prompts) == set(ROLES)
    assert set(spec.templates) == set(TEMPLATES)
    for role, text in spec.prompts.items():
        assert "{TASTE}" not in text and "{RUBRIC}" not in text, role
    assert "THE ARTIST'S SIGNATURE" in spec.prompts["CRITIC"]
    assert "SCALE" in spec.prompts["EDITOR"]


def _fields(template: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(template) if name}


def test_templates_take_exactly_what_the_program_fills(spec):
    """A placeholder renamed in the md but not in the code fails a run at its
    first editor call — catch the drift here instead."""
    assert _fields(spec.templates["BRIEF_USER"]) == {"idea", "mode", "source_prompt"}
    assert _fields(spec.templates["CRITIC_USER"]) == {
        "subject", "twist", "must_keep", "mood", "round", "rounds", "fragile_line"}
    assert _fields(spec.templates["DUEL_USER"]) == {"subject", "twist", "must_keep", "mood"}
    assert _fields(spec.templates["EDITOR_USER"]) == {
        "subject", "twist", "must_keep", "free", "mood", "mode", "anchor_head", "anchor_tail",
        "next_round", "rounds", "phase", "plateau", "operators_allowed", "best_round", "best_total",
        "best_core", "best_scores", "best_keep", "round", "accepted_line", "latest_core",
        "latest_observed", "latest_must_keep", "latest_scores", "latest_defects", "latest_expect",
        "history"}


def test_missing_section_or_block_fails_loudly(spec):
    with pytest.raises(RuntimeError, match="PROMPT: DUEL"):
        parse_spec("## PROMPT: BRIEF\nx\n## PROMPT: CRITIC\nx\n## PROMPT: EDITOR\nx\n")
    with pytest.raises(RuntimeError, match="unknown block"):
        parse_spec("## PROMPT: BRIEF\n{NOPE}\n")


def test_signature_anchor_rebuilds_the_published_prompt(spec):
    """Desert Reflections, verbatim from the gallery row."""
    anchor = spec.anchors["signature"]
    assert anchor.mode == "formula"
    triggers = triggers_for([
        {"name": "rustorangeanddimblue_lora_copy.safetensors"}, {"name": "art_vision_ZIG.safetensors"},
        {"name": "zImageT_zidiusArt_melancholy.safetensors"}, {"name": "art_vision_v2_epoch_10.safetensors"},
    ])
    assert triggers == ["rustorangeanddimblue", "art_vision", "zidiusArt"]
    assert compose_prompt(anchor, "extremely long shot of a person in twisted reality", triggers) == (
        "rustorangeanddimblue, art_vision, zidiusArt, Psychologically charged conceptual photograph, "
        "extremely long shot of a person in twisted reality, in the style of contemplative absurdities "
        "and neo-ultra panavision 70, surrealism, weirdcore/oddcore, Rust orange and dim blue, VHS filter")


def test_prose_anchor_without_lora_is_just_the_core(spec):
    assert compose_prompt(spec.anchors["prose"], "A pour of paint.", []) == "A pour of paint"


# ── checks ───────────────────────────────────────────────────────────────────

KW = dict(anchor_head="{triggers}, Psychologically charged conceptual photograph,",
          anchor_tail="in the style of contemplative absurdities and neo-ultra panavision 70, surrealism",
          triggers=["artvis3"])
GOOD = "extremely long shot of a lone lighthouse keeper beneath a lighthouse built from stacked people on a salt flat"


def test_good_core_passes():
    assert check_core(GOOD, mode="formula", must_keep=["lighthouse", "a lighthouse keeper"], **KW) is None


@pytest.mark.parametrize("core,rule", [
    ("a lighthouse", "length"),
    (GOOD.replace("lighthouse built", "tower built").replace("lighthouse keeper", "keeper"), "must_keep"),
    (GOOD + " under rust-orange haze", "formula"),
    ("artvis3, " + GOOD, "anchor"),
    (GOOD + " psychologically charged conceptual photograph", "anchor"),
    (GOOD + " und Leuchtturm", "language"),
    (GOOD + " with no sky", "negation"),
])
def test_each_rule(core, rule):
    v = check_core(core, mode="formula", must_keep=["lighthouse"], **KW)
    assert v is not None and v.rule == rule, v


def test_rust_as_material_is_allowed_in_formula():
    core = "macro shot of rust growing over a single leaf until its veins become a map of a city"
    assert check_core(core, mode="formula", must_keep=["leaf"], **KW) is None


def test_repeat_is_caught_on_normalized_text():
    v = check_core(GOOD + ".", mode="formula", must_keep=[], seen_cores={GOOD.lower(): 3}, **KW)
    assert v is not None and v.rule == "repeat" and "round 3" in v.message


def test_prose_length_band():
    assert check_core(" ".join(["paint"] * 100), mode="prose", must_keep=[], **KW).rule == "length"


def test_head_noun_and_must_keeps():
    assert head_noun("a reaching hand") == "hand"
    assert head_noun("kintsugi seams") == "seam"
    assert head_noun("two bodies") == "bod"
    assert missing_must_keeps("two figures emerge from a poured mass", ["two figures", "paint pour"]) == []
    assert missing_must_keeps("two figures", ["a reaching hand"]) == ["a reaching hand"]


# ── judge ────────────────────────────────────────────────────────────────────

def test_weighted_total_and_gates():
    s = {"intent": 7, "surreal": 6, "signature": 5, "composition": 8, "craft": 9}
    assert judge.weighted_total(s) == 6.7
    rep = {"scores": s, "must_keep": [{"item": "x", "status": "present"}]}
    assert judge.gate_failure(rep) is None
    assert "intent" in judge.gate_failure({**rep, "scores": {**s, "intent": 4}})
    assert "craft" in judge.gate_failure({**rep, "scores": {**s, "craft": 3}})
    assert "missing" in judge.gate_failure({**rep, "must_keep": [{"item": "x", "status": "missing"}]})


def test_triage_bands():
    assert judge.triage(5.0, None) == "first"
    assert judge.triage(6.6, 6.0) == "clear"
    assert judge.triage(6.5, 6.0) == "close"
    assert judge.triage(5.5, 6.0) == "close"
    assert judge.triage(5.4, 6.0) == "reject"


@pytest.mark.parametrize("first,second,out", [
    ("A", "B", "candidate"), ("B", "A", "best"), ("A", "A", "split"),
    ("B", "B", "split"), ("tie", "tie", "split"), ("A", "tie", "split"),
])
def test_duel_counts_only_when_both_orders_agree(first, second, out):
    assert judge.duel_verdict(first, second) == out


def test_phases():
    assert [judge.phase_for(r, 5) for r in range(2, 6)] == ["explore", "refine", "refine", "polish"]
    assert [judge.phase_for(r, 3) for r in range(2, 4)] == ["explore", "polish"]
    assert judge.phase_for(2, 2) == "explore"
    ph = [judge.phase_for(r, 20) for r in range(1, 21)]
    assert ph.count("explore") == 7 and ph.count("polish") == 3 and ph[-1] == "polish"


def test_allowed_operators():
    explore = judge.allowed_operators(mode="formula", phase="explore", plateau=False, recent=[], intent_weak=False)
    assert "LIGHT" not in explore and "CAPTURE" not in explore and "CLARIFY" not in explore
    assert "LIGHT" in judge.allowed_operators(mode="prose", phase="explore", plateau=False, recent=[], intent_weak=False)
    assert judge.allowed_operators(mode="formula", phase="polish", plateau=False, recent=[],
                                   intent_weak=False) == ["CLARIFY", "REDUCE"]
    assert "CLARIFY" in judge.allowed_operators(mode="formula", phase="explore", plateau=False, recent=[],
                                                intent_weak=True)
    jolt = judge.allowed_operators(mode="formula", phase="refine", plateau=True,
                                   recent=["SCALE", "SCALE", "REPEAT", "CAMERA", "MERGE"], intent_weak=False)
    assert not {"SCALE", "REPEAT", "CAMERA", "MERGE"} & set(jolt) and jolt


# ── validation ───────────────────────────────────────────────────────────────

def test_report_validation_lifts_scores_and_fills_must_keeps():
    r = validate_report({"observed": "o", "intent": "7", "surreal": 6.6, "signature": 12, "composition": 0,
                         "craft": 5, "must_keep": [{"item": "Lighthouse", "status": "present"}]},
                        ["lighthouse", "keeper"])
    assert r["scores"] == {"intent": 7, "surreal": 7, "signature": 10, "composition": 1, "craft": 5}
    assert r["must_keep"] == [{"item": "lighthouse", "status": "present"}, {"item": "keeper", "status": "weak"}]
    with pytest.raises(ValueError):
        validate_report({"scores": {"intent": "high"}}, [])


def test_brief_and_edit_validation():
    b = validate_brief({"subject": "s", "twist": "t", "core": "c", "must_keep": "a, b, a", "twist_source": "x"})
    assert b["must_keep"] == ["a", "b"] and b["twist_source"] == "user"
    with pytest.raises(ValueError):
        validate_brief({"subject": "s", "twist": "", "core": "c"})
    with pytest.raises(ValueError, match="not allowed"):
        validate_edit({"core": "c", "operator": "LIGHT"}, ["SCALE"])
    assert validate_edit({"core": "c", "operator": "scale"}, ["SCALE"])["operator"] == "SCALE"


# ── series ───────────────────────────────────────────────────────────────────

def test_series_puts_the_best_first_and_caps_rejected_rounds():
    rounds = [{"round": i, "image_id": f"r{i}", "decision": "accepted" if i in (1, 5) else "rejected"}
              for i in range(1, 26)]
    rounds[0]["decision"] = "first"
    run = {"rounds": rounds, "best_round": 5, "sweep": [{"image_id": "s1"}], "final": None}
    ids = _series_members(run)
    assert ids[0] == "r5" and len(ids) == SERIES_MAX
    assert "r1" in ids and "s1" in ids
    run["final"] = {"image_id": "s1"}
    assert _series_members(run)[0] == "s1"
