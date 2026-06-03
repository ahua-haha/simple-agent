"""Tests for fractional order keys."""

from simple_agent.fractional_index import key_after, key_between


def test_key_between_creates_lexicographic_middle():
    first = key_between(None, None)
    second = key_after(first)
    middle = key_between(first, second)

    assert first < middle < second


def test_key_after_can_extend_indefinitely():
    key = key_between(None, None)

    for _ in range(100):
        next_key = key_after(key)
        assert key < next_key
        key = next_key


def test_key_between_rejects_out_of_order_bounds():
    first = key_between(None, None)
    second = key_after(first)

    try:
        key_between(second, first)
    except ValueError as exc:
        assert "less than" in str(exc)
    else:
        raise AssertionError("expected ValueError")
