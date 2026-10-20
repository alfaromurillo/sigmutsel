"""Pure helpers in ``constants``."""

import itertools

import pytest

from sigmutsel import constants
from sigmutsel.constants import extract_context, reverse_complement


@pytest.mark.parametrize(
    ("seq", "expected"),
    [
        ("", ""),
        ("A", "T"),
        ("C", "G"),
        ("ACGT", "ACGT"),  # its own reverse complement
        ("AAC", "GTT"),
        ("GATTACA", "TGTAATC"),
    ],
)
def test_reverse_complement_known_values(seq, expected):
    assert reverse_complement(seq) == expected


def test_reverse_complement_is_an_involution():
    for seq in map("".join, itertools.product("ACGT", repeat=4)):
        assert reverse_complement(reverse_complement(seq)) == seq


def test_reverse_complement_leaves_other_characters():
    # Only upper-case A/C/G/T are complemented; anything else (an
    # ambiguous N, a lower-case soft-masked base) is reversed as is.
    assert reverse_complement("ANC") == "GNT"
    assert reverse_complement("acgT") == "Agca"


def test_reverse_complement_maps_contexts_to_purine_centres():
    # Every canonical (pyrimidine-centred) context reverse-complements
    # to a distinct purine-centred one, and back.
    images = [
        reverse_complement(c)
        for c in constants.canonical_contexts_order
    ]
    assert all(c[1] in "AG" for c in images)
    assert len(set(images)) == len(images) == 32
    assert [reverse_complement(c) for c in images] == (
        constants.canonical_contexts_order
    )


def test_extract_context_known_values():
    assert extract_context("G[C>T]G") == "GCG"
    assert extract_context("A[T>C]A") == "ATA"


def test_extract_context_matches_canonical_orders():
    # The 96 types collapse onto the 32 contexts three to one, in the
    # contexts' own canonical order.
    contexts = [
        extract_context(t) for t in constants.canonical_types_order
    ]
    assert len(contexts) == 96
    assert sorted(set(contexts)) == sorted(
        constants.canonical_contexts_order
    )
    assert all(contexts.count(c) == 3 for c in set(contexts))
