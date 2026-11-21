"""Synthetic only: exact ordered tokenization parity and bounded local reuse."""
import pytest

from research.unified_training.efficient_batches import (
    TokenizationCache,
    TokenizationError,
    TokenizationPin,
    TokenizationRequest,
    synthetic_ablation,
)


PIN = TokenizationPin('0' * 64, '1' * 64, 'a' * 64)


def _encoder(role, text):
    return (('source', 'target', 'context').index(role), *text.encode('utf-8'))


def _request(text, role='source', shareable=True):
    return TokenizationRequest(role, text, shareable)


def test_exact_order_parity_includes_role_unicode_and_repeated_rows():
    rows = [
        _request('café'),
        _request('café', 'target'),
        _request('cafe\u0301'),
        _request('café'),
        _request('café', 'target'),
        _request(' café'),
        _request('café'),
    ]
    off = TokenizationCache(PIN, _encoder)
    on = TokenizationCache(PIN, _encoder, max_entries=10)
    expected = tuple(_encoder(row.role, row.text) for row in rows)
    assert tuple(off.encode_ordered(rows, pin=PIN)) == tuple(on.encode_ordered(iter(rows), pin=PIN)) == expected
    assert off.stats()['encoder_calls'] == len(rows) and off.stats()['retained_entries'] == 0
    assert on.stats()['encoder_calls'] == 4
    assert on.stats()['hits'] == 3
    assert expected[0] != expected[1] != expected[2] and expected[0] != expected[2]
    assert expected[0] != expected[5]


def test_private_and_context_requests_never_seed_or_lookup_cache():
    rows = [
        TokenizationRequest('source', 'synthetic private'),
        TokenizationRequest('source', 'synthetic private'),
        _request('synthetic context', 'context'),
        _request('synthetic context', 'context'),
        _request('synthetic shared'),
        TokenizationRequest('source', 'synthetic shared'),
        _request('synthetic shared'),
    ]
    cache = TokenizationCache(PIN, _encoder, max_entries=64)
    for row, tokens in zip(rows, cache.encode_ordered(rows, pin=PIN), strict=True):
        assert tokens == _encoder(row.role, row.text)
    assert cache.stats()['encoder_calls'] == 6 and cache.stats()['hits'] == 1
    assert cache.stats()['retained_entries'] == 1
    assert 'synthetic private' not in repr(rows[0])


def test_entry_budget_implements_lru_and_clear_releases_charged_bytes():
    cache = TokenizationCache(PIN, _encoder, max_entries=2)
    rows = tuple(_request(value) for value in ('a', 'b', 'a', 'c', 'a', 'b'))
    assert tuple(cache.encode_ordered(rows, pin=PIN)) == tuple(_encoder(row.role, row.text) for row in rows)
    assert cache.stats()['retained_entries'] == 2 and cache.stats()['hits'] == 2
    assert cache.stats()['encoder_calls'] == 4 and cache.stats()['evictions'] == 2
    cache.clear()
    assert cache.stats()['retained_entries'] == cache.stats()['charged_retained_bytes'] == 0
    assert cache.stats()['encoder_calls'] == 4


def test_byte_budget_counts_retained_values_and_does_not_truncate():
    # Measure a platform-specific retained charge without depending on CPython object sizes.
    probe = TokenizationCache(PIN, _encoder, max_entries=3)
    assert tuple(probe.encode_ordered([_request('a')], pin=PIN)) == (_encoder('source', 'a'),)
    charge = probe.stats()['charged_retained_bytes']
    cache = TokenizationCache(PIN, _encoder, max_entries=3, max_retained_bytes=charge)
    rows = [_request('a'), _request('b'), _request('longer'), _request('b')]
    assert tuple(cache.encode_ordered(rows, pin=PIN)) == tuple(_encoder(row.role, row.text) for row in rows)
    assert cache.stats()['retained_entries'] == cache.stats()['evictions'] == cache.stats()['hits'] == 1
    assert cache.stats()['charged_retained_bytes'] == charge


def test_pin_mismatch_is_rejected_before_cache_or_encoder_use():
    variants = (
        TokenizationPin('f' * 64, PIN.tokenizer_sha256, PIN.protocol_sha256),
        TokenizationPin(PIN.model_sha256, 'f' * 64, PIN.protocol_sha256),
        TokenizationPin(PIN.model_sha256, PIN.tokenizer_sha256, 'f' * 64),
    )
    cache = TokenizationCache(PIN, _encoder, max_entries=64)
    one = [_request('sample')]
    assert tuple(cache.encode_ordered(one, pin=PIN)) == (_encoder('source', 'sample'),)
    for wrong in variants:
        with pytest.raises(TokenizationError, match='pin mismatch'):
            tuple(cache.encode_ordered(one, pin=wrong))
    assert cache.stats()['requests'] == cache.stats()['encoder_calls'] == 1
    with pytest.raises(TokenizationError, match='SHA-256'):
        TokenizationPin('fake-model-name', PIN.tokenizer_sha256, PIN.protocol_sha256)


def test_oversize_uncached_tokens_pass_through_and_errors_redact_text():
    row = _request('sample')
    huge = 9000 * (1,)
    cache = TokenizationCache(PIN, lambda role, text: huge, max_entries=8)
    assert tuple(cache.encode_ordered([row, row], pin=PIN)) == (huge, huge)
    assert cache.stats()['encoder_calls'] == 2 and cache.stats()['retained_entries'] == 0

    def echo_call_failure(role, text):
        raise ValueError(text)
    cache = TokenizationCache(PIN, echo_call_failure)
    with pytest.raises(TokenizationError, match='tokenizer callback failed') as exc:
        tuple(cache.encode_ordered([TokenizationRequest('context', 'private-ish')], pin=PIN))
    assert 'private-ish' not in str(exc.value)


def test_reproducible_ablation_counts_not_latency_or_training_quality():
    assert synthetic_ablation() == {
        'scope': 'synthetic_tokenizer_operation_counts', 'ordered_output_parity': True,
        'requests': 32, 'baseline_encoder_calls': 32, 'cached_encoder_calls': 18,
        'cache_hits': 14, 'retained_entries': 2,
    }


def test_unbounded_encoder_output_is_rejected_not_truncated():
    from itertools import repeat
    cache = TokenizationCache(PIN, lambda _role, _text: repeat(1), max_entries=1)
    with pytest.raises(TokenizationError, match="refuse instead of truncating"):
        list(cache.encode_ordered([_request("synthetic")], pin=PIN))
    assert cache.stats()["retained_entries"] == 0
