"""Optional process-local token reuse; no trainer, disk, model or defaults touched.

The reviewed #42 CPU trainer already provides opt-in LENGTH_BUCKET_SIZE and
four microbatches of four examples per optimizer update. This helper does not
reorder, pad, truncate, batch, change gradient accumulation or enable that
trainer. Integrators must bind a deterministic, unbatched encoder (no random
tokenization or batch-dependent padding) and keep all semantic options pinned.
"""
from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
from itertools import islice
import json
import re
import sys


_SHA256 = re.compile(r'[a-f0-9]{64}')
_ROLES = frozenset(('source', 'target', 'context'))
_MAX_ENTRIES = 16_384
_MAX_BYTES = 64 * 1024 * 1024
_MAX_CACHE_TEXT_CHARS = 16_384
_MAX_CACHE_TOKENS = 8192
_MAX_OUTPUT_TOKENS = 131_072
# A conservative accounting charge for tuple keys, mapping/order links and
# value records. This is a cache retention budget, not process RSS.
_ENTRY_OVERHEAD = 384


class TokenizationError(ValueError):
    """The supplied tokenizer, pin or request is not supported."""


def _instruction(condition, message):
    if not condition:
        raise TokenizationError(message)


@dataclass(frozen=True)
class TokenizationPin:
    """Declared digests of verified model, tokenizer and tokenization protocol.

    The caller verifies actual local bytes. The protocol digest must include
    language direction, special tokens, normalization/truncation settings,
    tokenizer code/version and any other setting that changes emitted IDs.
    This helper cannot attest to the identity or purity of a Python callback.
    """

    model_sha256: str
    tokenizer_sha256: str
    protocol_sha256: str

    def __post_init__(self):
        _instruction(all(type(pin) is str and _SHA256.fullmatch(pin)
                         for pin in (self.model_sha256, self.tokenizer_sha256, self.protocol_sha256)),
                     'expected pinned model, tokenizer and protocol SHA-256 digests')


@dataclass(frozen=True)
class TokenizationRequest:
    """A text request; only explicitly permitted public rows may be retained."""

    role: str
    text: str = field(repr=False)
    shareable: bool = False


class TokenizationCache:
    """Bounded per-run LRU; disabled unless max_entries is explicitly positive.

    Encoder signature is `encoder(role, text) -> iterable of token IDs`.
    Use a fresh cache for any encoder/pin change, and use from one thread.
    Private/default requests and all context bypass both lookup and insertion
    even if the same text was cached for a shareable request. This utility never
    writes cache contents, requests, or input hashes to files or reports.
    """

    def __init__(self, pin, encoder, *, max_entries=0, max_retained_bytes=4 * 1024 * 1024):
        _instruction(type(pin) is TokenizationPin and callable(encoder), 'expected a pinned tokenizer callback')
        _instruction(type(max_entries) is int and 0 <= max_entries <= _MAX_ENTRIES, 'invalid cache entry budget')
        _instruction(type(max_retained_bytes) is int and 0 <= max_retained_bytes <= _MAX_BYTES,
                     'invalid cache retention budget')
        self._pin = pin
        self._encoder = encoder
        self._max_entries = max_entries
        self._max_bytes = max_retained_bytes
        self._cache = OrderedDict()
        self._requests = self._calls = self._hits = self._evictions = 0
        self._bytes = 0

    def _encode(self, request):
        self._calls += 1
        try:
            tokens = tuple(islice(self._encoder(request.role, request.text), _MAX_OUTPUT_TOKENS + 1))
        except Exception:
            # Do not surface tokenizer errors that may contain source text.
            raise TokenizationError('tokenizer callback failed') from None
        _instruction(len(tokens) <= _MAX_OUTPUT_TOKENS,
                     'tokenizer output limit exceeded; refuse instead of truncating')
        _instruction(all(type(token) is int and 0 <= token < 2**32 for token in tokens),
                     'tokenizer must yield nonnegative integer token IDs')
        return tokens

    def encode_ordered(self, requests, *, pin):
        """Yield one exact immutable token tuple per request in supplied order."""
        _instruction(type(pin) is TokenizationPin and pin == self._pin, 'tokenization pin mismatch')
        for request in requests:
            _instruction(type(request) is TokenizationRequest and type(request.role) is str
                         and request.role in _ROLES and type(request.text) is str
                         and type(request.shareable) is bool, 'invalid tokenization request')
            self._requests += 1
            if (not self._max_entries or not self._max_bytes or not request.shareable
                    or request.role == 'context' or len(request.text) > _MAX_CACHE_TEXT_CHARS):
                yield self._encode(request)
                continue

            key = (request.role, request.text)  # Exact strings; role-specific tokenization.
            if key in self._cache:
                self._hits += 1
                self._cache.move_to_end(key)
                yield self._cache[key][0]
                continue

            tokens = self._encode(request)
            if len(tokens) <= _MAX_CACHE_TOKENS:
                cost = (_ENTRY_OVERHEAD + sys.getsizeof(request.text) + sys.getsizeof(tokens)
                        + sum(sys.getsizeof(token) for token in tokens))
                if cost <= self._max_bytes:
                    while self._cache and (len(self._cache) >= self._max_entries
                                           or self._bytes + cost > self._max_bytes):
                        _, (_, old_cost) = self._cache.popitem(last=False)
                        self._bytes -= old_cost
                        self._evictions += 1
                    self._cache[key] = (tokens, cost)
                    self._bytes += cost
            yield tokens

    def stats(self):
        """Return redacted counts and budget accounting; no model timings."""
        return {
            'requests': self._requests,
            'encoder_calls': self._calls,
            'hits': self._hits,
            'evictions': self._evictions,
            'retained_entries': len(self._cache),
            'charged_retained_bytes': self._bytes,
        }

    def clear(self):
        """Release retained inputs/results; cumulative operation counts remain."""
        self._cache.clear()
        self._bytes = 0


def synthetic_ablation():
    """Same ordered role/strings with cache off/on; no weights or personal data."""
    def digest(value):
        return hashlib.sha256(value.encode('ascii')).hexdigest()

    pin = TokenizationPin(digest('synthetic-model'), digest('synthetic-tokenizer'), digest('synthetic-encoder-v1'))

    def synthetic_encoder(role, text):
        return (('source', 'target', 'context').index(role), *text.encode('utf-8'))

    workload = tuple(
        request
        for _ in range(8)
        for request in (
            TokenizationRequest('source', 'example shared', True),
            TokenizationRequest('target', 'example shared', True),
            TokenizationRequest('context', 'example context', True),
            TokenizationRequest('source', 'example default bypass'),
        )
    )
    baseline = TokenizationCache(pin, synthetic_encoder)
    candidate = TokenizationCache(pin, synthetic_encoder, max_entries=8)
    unchanged = tuple(baseline.encode_ordered(workload, pin=pin)) == tuple(candidate.encode_ordered(workload, pin=pin))
    b, c = baseline.stats(), candidate.stats()
    return {'scope': 'synthetic_tokenizer_operation_counts', 'ordered_output_parity': unchanged,
            'requests': b['requests'], 'baseline_encoder_calls': b['encoder_calls'],
            'cached_encoder_calls': c['encoder_calls'], 'cache_hits': c['hits'],
            'retained_entries': c['retained_entries']}


if __name__ == '__main__':
    print(json.dumps(synthetic_ablation(), sort_keys=True))
