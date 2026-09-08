# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Terms carry a weight: how many times the app has seen them.

Every merge that carries a term adds its weight to the one already stored, so a
name that keeps coming up rises above one mentioned once. That ordering is what
decides which terms survive a budget — Whisper's prompt is capped by characters
and Azure's phrase list at fifty entries, and both drop from the tail.
"""

from __future__ import annotations

from transcriber_studio import vocab_bias
from transcriber_studio.config import Settings
from transcriber_studio.glossary import (
    DEFAULT_WEIGHT,
    WEIGHT_KEY,
    by_weight,
    merge_terms,
    term_weight,
)


def _term(canonical: str, weight: int | None = None, kind: str = "product") -> dict:
    entry = {"canonical": canonical, "type": kind}
    if weight is not None:
        entry[WEIGHT_KEY] = weight
    return entry


# ---- reading a weight ------------------------------------------------
def test_a_term_with_no_weight_counts_as_seen_once():
    """Terms written before weights existed, and ones typed by hand, have none.
    Zero would sort them last and quietly drop them from every budget."""
    assert term_weight({"canonical": "Plaud"}) == DEFAULT_WEIGHT == 1


def test_nonsense_weights_fall_back_rather_than_crash():
    assert term_weight({"canonical": "X", WEIGHT_KEY: "lots"}) == 1
    assert term_weight({"canonical": "X", WEIGHT_KEY: None}) == 1
    assert term_weight({"canonical": "X", WEIGHT_KEY: 0}) == 1
    assert term_weight({"canonical": "X", WEIGHT_KEY: -5}) == 1


# ---- accumulating ----------------------------------------------------
def test_seeing_a_term_again_increases_its_weight():
    merged = merge_terms([[_term("Plaud")], [_term("Plaud")], [_term("Plaud")]])
    assert len(merged) == 1
    assert term_weight(merged[0]) == 3


def test_case_differences_are_the_same_term():
    merged = merge_terms([[_term("Plaud")], [_term("plaud")], [_term("PLAUD")]])
    assert len(merged) == 1
    assert term_weight(merged[0]) == 3


def test_an_existing_weight_is_added_to_not_replaced():
    """A term arriving with a weight of nine has been seen nine times already;
    a merge must not flatten that back to one."""
    merged = merge_terms([[_term("Plaud", 9)], [_term("Plaud")]])
    assert term_weight(merged[0]) == 10


def test_weights_survive_repeated_merges():
    merged = merge_terms([[_term("Plaud")], [_term("Plaud")]])
    for _ in range(3):
        merged = merge_terms([merged, [_term("Plaud")]])
    assert term_weight(merged[0]) == 5


def test_a_new_term_starts_at_one():
    merged = merge_terms([[_term("Plaud", 4)], [_term("Ottinger")]])
    weights = {t["canonical"]: term_weight(t) for t in merged}
    assert weights == {"Plaud": 4, "Ottinger": 1}


# ---- ordering --------------------------------------------------------
def test_the_heaviest_term_comes_first():
    merged = merge_terms([
        [_term("Rare")],
        [_term("Common"), _term("Common"), _term("Common")],
        [_term("Sometimes"), _term("Sometimes")],
    ])
    assert [t["canonical"] for t in merged] == ["Common", "Sometimes", "Rare"]


def test_equal_weights_stay_alphabetical():
    """A stable order, so the glossary file does not churn between runs."""
    merged = merge_terms([[_term("Zebra"), _term("apple"), _term("Mango")]])
    assert [t["canonical"] for t in merged] == ["apple", "Mango", "Zebra"]


def test_by_weight_orders_a_plain_list():
    entries = [_term("a", 1), _term("b", 7), _term("c", 3)]
    assert [t["canonical"] for t in by_weight(entries)] == ["b", "c", "a"]


# ---- the rest of the entry is untouched ------------------------------
def test_variants_and_type_still_merge():
    merged = merge_terms([
        [{"canonical": "Plaud", "type": "product", "variants": ["plod"]}],
        [{"canonical": "Plaud", "type": "product", "variants": ["ploud"]}],
    ])
    assert set(merged[0]["variants"]) >= {"plod", "ploud"}
    assert merged[0]["type"] == "product"
    assert term_weight(merged[0]) == 2


def test_an_unresolved_conflict_still_outlives_the_merge():
    from transcriber_studio.glossary_merge import CONFLICT_KEY

    tagged = {"canonical": "Scribe", "type": "product", CONFLICT_KEY: {"field": "type"}}
    merged = merge_terms([[tagged], [_term("Scribe")]])
    assert CONFLICT_KEY in merged[0]


# ---- what the weight is for ------------------------------------------
def _payload(terms: list[dict], speakers: list[dict] | None = None) -> dict:
    return {"terms": terms, "speakers": speakers or []}


def test_the_bias_list_puts_heavier_terms_first():
    """The budget drops from the tail, so order is what decides survival."""
    settings = Settings()
    settings.bias_extra_terms = ""
    terms = vocab_bias.collect_terms(
        settings,
        glossary_id="",
        extra_payloads=[_payload([
            _term("Seldom", 1), _term("Constant", 12), _term("Middling", 5),
        ])],
    )
    assert terms == ["Constant", "Middling", "Seldom"]


def test_typed_terms_still_beat_everything():
    """What the user typed by hand is not a guess and outranks any count."""
    settings = Settings()
    settings.bias_extra_terms = "Handtyped"
    terms = vocab_bias.collect_terms(
        settings, glossary_id="",
        extra_payloads=[_payload([_term("Constant", 99)])],
    )
    assert terms[0] == "Handtyped"


def test_a_name_still_beats_jargon_whatever_the_weights():
    """Getting a person's name wrong is the error that shows."""
    settings = Settings()
    settings.bias_extra_terms = ""
    terms = vocab_bias.collect_terms(
        settings, glossary_id="",
        extra_payloads=[_payload([
            _term("Jargon", 50, kind="concept"),
            _term("Ottinger", 2, kind="person"),
        ])],
    )
    assert terms.index("Ottinger") < terms.index("Jargon")


def test_the_same_term_from_two_glossaries_keeps_its_heaviest_weight():
    settings = Settings()
    settings.bias_extra_terms = ""
    terms = vocab_bias.collect_terms(
        settings, glossary_id="",
        extra_payloads=[
            _payload([_term("Shared", 2), _term("Other", 5)]),
            _payload([_term("Shared", 30)]),
        ],
    )
    assert terms[0] == "Shared"


def test_the_heaviest_terms_are_the_ones_that_fit_a_budget():
    """The whole point: Azure takes fifty hints and Whisper a few hundred
    characters, and both keep the head of the list."""
    settings = Settings()
    settings.bias_extra_terms = ""
    heavy = [_term(f"Heavy{i}", 100 - i) for i in range(5)]
    light = [_term(f"Light{i}", 1) for i in range(50)]
    ordered = vocab_bias.collect_terms(
        settings, glossary_id="", extra_payloads=[_payload(light + heavy)]
    )
    assert ordered[:5] == ["Heavy0", "Heavy1", "Heavy2", "Heavy3", "Heavy4"]


def test_the_azure_hint_list_keeps_the_heaviest_fifty():
    from transcriber_studio import stt_mai

    class _Opts:
        mai_send_phrases = True
        hotwords = ", ".join(
            [f"Heavy{i}" for i in range(10)] + [f"Light{i}" for i in range(60)]
        )

    phrases = stt_mai._phrases(_Opts())
    assert len(phrases) == 50
    assert phrases[:3] == ["Heavy0", "Heavy1", "Heavy2"]
