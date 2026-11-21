"""Append-only local ledger for immutable research plan pins and aggregate receipts.

The SQLite ledger preserves canonical manifest and aggregate envelope bytes, not
phase payloads, config paths, text, translations, weights, or credentials. A
reservation is spent even when its process fails or is interrupted; another
process cannot take it over or silently rerun it.
"""
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import stat


DB_NAME = 'workflow-ledger.sqlite3'
_SCHEMA_VERSION = 1
_APPLICATION_ID = 0x42434C57
_PLAN_KEYS = frozenset(('schema_version', 'kind', 'plan_sha256'))
_RECEIPT_KEYS = _PLAN_KEYS | {'status', 'phases'}
_PHASES = frozenset(('data', 'checkpoint', 'comparison'))
_STATUSES = frozenset(('pass', 'fail', 'not_verified'))
_MAX_RECORD_BYTES = 4096
_SHA256 = re.compile(r'[a-f0-9]{64}')
_RUN_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}')


class StoreError(ValueError):
    """Local ledger could not prove or safely preserve the requested bundle."""


class DuplicatePlanError(StoreError):
    """An exact plan byte pin has already been reserved by another run ID."""

    def __init__(self, existing_run_id):
        self.existing_run_id = existing_run_id
        super().__init__(f'exact plan already reserved by run {existing_run_id}')


def _require(condition, message):
    if not condition:
        raise StoreError(message)


def _valid_id(value):
    return isinstance(value, str) and _RUN_ID.fullmatch(value) is not None


def _is_sha256(value):
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _encode(record):
    return json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')


def _exact(record, keys, name):
    _require(isinstance(record, Mapping) and record.keys() == keys, f'invalid {name} fields')


def _common(record, keys, kind):
    _exact(record, keys, kind)
    schema, observed_kind, plan_pin = record['schema_version'], record['kind'], record['plan_sha256']
    _require(type(schema) is int and schema == _SCHEMA_VERSION,
             'unsupported local workflow record schema')
    _require(type(observed_kind) is str and observed_kind == kind, 'invalid local workflow record kind')
    _require(_is_sha256(plan_pin), 'invalid exact plan byte pin')
    return {'schema_version': schema, 'kind': kind, 'plan_sha256': plan_pin}


def _plan_bytes(record):
    return _encode(_common(record, _PLAN_KEYS, 'local_research_workflow'))


def _receipt_bytes(record):
    envelope = _common(record, _RECEIPT_KEYS, 'local_research_workflow_receipt')
    phases = record['phases']
    _exact(phases, _PHASES, 'receipt phases')
    out = {}
    for name in _PHASES:
        phase = phases[name]
        _exact(phase, {'status', 'digest'}, 'phase')
        status = phase['status']
        digest = phase['digest']
        _require(type(status) is str and status in _STATUSES, 'invalid phase status')
        _require(_is_sha256(digest) or (digest is None and status == 'not_verified'),
                 'invalid or missing aggregate phase digest')
        out[name] = {'status': status, 'digest': digest}
    statuses = {phase['status'] for phase in out.values()}
    expected = 'fail' if 'fail' in statuses else 'pass' if statuses == {'pass'} else 'not_verified'
    status = record['status']
    _require(type(status) is str and status == expected, 'receipt status disagrees with phase outcomes')
    return _encode({**envelope, 'status': status, 'phases': out})


def _stored_record(raw, expected_sha, canonicalize):
    _require(type(raw) is bytes and 0 < len(raw) <= _MAX_RECORD_BYTES, 'invalid local record bytes')
    _require(_is_sha256(expected_sha) and _sha(raw) == expected_sha, 'local record checksum mismatch')
    try:
        record = json.loads(raw)
    except (UnicodeError, ValueError, RecursionError):
        raise StoreError('invalid local record JSON') from None
    # Comparing exact canonical bytes also rejects duplicate JSON keys.
    _require(raw == canonicalize(record), 'local record is not in canonical form')
    return record


@dataclass(frozen=True)
class RunReservation:
    run_id: str
    plan_sha256: str
    created: bool
    _owner: bytes = field(default=b'', repr=False, compare=False)


@dataclass(frozen=True)
class RunAudit:
    run_id: str
    plan_sha256: str
    manifest_sha256: str
    receipt_sha256: str | None
    status: str | None
    _plan: bytes = field(repr=False)
    _receipt: bytes | None = field(repr=False)

    @property
    def state(self):
        return 'reserved' if self._receipt is None else 'completed'

    @property
    def plan(self):
        """Return a fresh copy of the verified plan pin envelope."""
        return json.loads(self._plan)

    @property
    def receipt(self):
        """Return a fresh copy of the verified aggregate envelope, if present."""
        return None if self._receipt is None else json.loads(self._receipt)


class WorkflowStore:
    """Preserve safe metadata under a dedicated, trusted local directory."""

    def __init__(self, root):
        self.root = Path(root).absolute()
        self.database = self.root / DB_NAME

    @staticmethod
    def _mode(path):
        try:
            return path.lstat().st_mode
        except FileNotFoundError:
            return None

    def _check_location(self, *, create):
        # A trusted local ancestor is required, but avoid accidentally targeting
        # another store through any existing symbolic link in the supplied path.
        _require(all(not path.is_symlink() for path in (self.root, *self.root.parents)),
                 'local ledger path may not contain a symlink')
        if create:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        mode = self._mode(self.root)
        _require(mode is not None and stat.S_ISDIR(mode), 'local ledger directory does not exist')
        if create:
            try:
                fd = os.open(self.database, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                pass
            else:
                os.close(fd)
        for suffix in ('', '-journal', '-wal', '-shm'):
            observed = self._mode(self.database.with_name(DB_NAME + suffix))
            _require((observed is None and suffix) or (observed is not None and stat.S_ISREG(observed)),
                     'local ledger requires an existing regular database and ordinary sidecars')

    @contextmanager
    def _transaction(self, *, write=False, create=False):
        db = None
        try:
            self._check_location(create=create)
            # Open an existing file only: reserve above is the sole creator.
            # If SQLite needs interrupted-journal recovery, audit fails closed
            # rather than mutating the file from a read-only inspection.
            mode = 'rw' if write else 'ro'
            db = sqlite3.connect(self.database.as_uri() + f'?mode={mode}', uri=True,
                                 isolation_level=None, timeout=15)
            db.execute('PRAGMA foreign_keys = ON')
            db.execute('PRAGMA trusted_schema = OFF')
            if write:
                db.execute('PRAGMA recursive_triggers = ON')
                db.execute('PRAGMA synchronous = FULL')
            else:
                db.execute('PRAGMA query_only = ON')
            db.execute('BEGIN IMMEDIATE' if write else 'BEGIN')
            app_id = db.execute('PRAGMA application_id').fetchone()[0]
            schema_version = db.execute('PRAGMA user_version').fetchone()[0]
            if create and app_id == 0 and schema_version == 0:
                _require(not db.execute('SELECT 1 FROM sqlite_schema LIMIT 1').fetchone(),
                         'refusing to initialize over an unrelated local database')
                self._initialize(db)
            else:
                _require(app_id == _APPLICATION_ID and schema_version == _SCHEMA_VERSION,
                         'uninitialized or incompatible local workflow ledger')
            yield db
            db.execute('COMMIT')
        except (sqlite3.Error, OSError):
            # Do not include local filenames, database text, or private paths.
            raise StoreError('unable to access or preserve the local workflow ledger') from None
        finally:
            if db is not None:
                db.close()  # Rolls back any active transaction on failure.

    @staticmethod
    def _initialize(db):
        db.execute('''
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                plan_sha256 TEXT NOT NULL UNIQUE,
                manifest_sha256 TEXT NOT NULL,
                manifest BLOB NOT NULL,
                owner_sha256 TEXT NOT NULL
            ) WITHOUT ROWID
        ''')
        db.execute('''
            CREATE TABLE receipts (
                run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
                receipt_sha256 TEXT NOT NULL,
                receipt BLOB NOT NULL
            ) WITHOUT ROWID
        ''')
        # The application never issues updates or deletes. These triggers also
        # catch accidental SQL writes, not edits by someone controlling the file.
        for table in ('runs', 'receipts'):
            for operation in ('update', 'delete'):
                db.execute(f'''
                    CREATE TRIGGER {table}_no_{operation} BEFORE {operation} ON {table}
                    BEGIN SELECT RAISE(ABORT, 'workflow ledger is append-only'); END
                ''')
        db.execute(f'PRAGMA application_id = {_APPLICATION_ID}')
        db.execute(f'PRAGMA user_version = {_SCHEMA_VERSION}')

    @staticmethod
    def _row(db, run_id):
        return db.execute('''
            SELECT runs.run_id, plan_sha256, manifest_sha256, manifest, owner_sha256,
                   receipt_sha256, receipt
            FROM runs LEFT JOIN receipts USING (run_id)
            WHERE runs.run_id = ?
        ''', (run_id,)).fetchone()

    @staticmethod
    def _verified(row):
        run_id, plan_pin, manifest_sha, manifest, owner_sha, receipt_sha, receipt = row
        _require(_valid_id(run_id) and _is_sha256(owner_sha), 'invalid local reservation metadata')
        plan_obj = _stored_record(manifest, manifest_sha, _plan_bytes)
        _require(plan_pin == plan_obj['plan_sha256'], 'stored reservation plan pin mismatch')
        status = None
        if receipt is not None or receipt_sha is not None:
            receipt_obj = _stored_record(receipt, receipt_sha, _receipt_bytes)
            _require(plan_pin == receipt_obj['plan_sha256'], 'stored receipt plan pin mismatch')
            status = receipt_obj['status']
        return RunAudit(run_id, plan_pin, manifest_sha, receipt_sha, status, manifest, receipt)

    def reserve(self, run_id, *, plan):
        """Reserve an exact input plan pin globally, before phase work starts."""
        _require(_valid_id(run_id), 'invalid opaque local run ID')
        manifest = _plan_bytes(plan)
        plan_pin = json.loads(manifest)['plan_sha256']
        with self._transaction(write=True, create=True) as db:
            row = self._row(db, run_id)
            if row is not None:
                audit = self._verified(row)
                _require(audit._plan == manifest, 'run ID is already bound to a different plan')
                return RunReservation(run_id, plan_pin, False)
            other = db.execute('SELECT run_id FROM runs WHERE plan_sha256 = ?', (plan_pin,)).fetchone()
            if other is not None:
                self._verified(self._row(db, other[0]))
                raise DuplicatePlanError(other[0])
            owner = secrets.token_bytes(32)
            db.execute('''
                INSERT INTO runs (run_id, plan_sha256, manifest_sha256, manifest, owner_sha256)
                VALUES (?, ?, ?, ?, ?)
            ''', (run_id, plan_pin, _sha(manifest), manifest, _sha(owner)))
            return RunReservation(run_id, plan_pin, True, owner)

    def complete(self, reservation, *, receipt):
        """Append an aggregate receipt once; never replace or salvage a run."""
        _require(isinstance(reservation, RunReservation) and _valid_id(reservation.run_id),
                 'local completion requires a run reservation')
        raw = _receipt_bytes(receipt)
        receipt_obj = json.loads(raw)
        with self._transaction(write=True) as db:
            row = self._row(db, reservation.run_id)
            _require(row is not None, 'run is not reserved in this local ledger')
            audit = self._verified(row)
            _require(audit.plan_sha256 == reservation.plan_sha256 == receipt_obj['plan_sha256'],
                     'completion plan pin does not match reservation')
            if audit._receipt is not None:
                _require(raw == audit._receipt, 'run already has a different immutable receipt')
                return audit
            _require(reservation.created and type(reservation._owner) is bytes
                     and len(reservation._owner) == 32 and secrets.compare_digest(row[4], _sha(reservation._owner)),
                     'prior reservation cannot be taken over or retried')
            db.execute('INSERT INTO receipts (run_id, receipt_sha256, receipt) VALUES (?, ?, ?)',
                       (audit.run_id, _sha(raw), raw))
            return self._verified(self._row(db, audit.run_id))

    def audit(self, run_id):
        """Read and recheck canonical hashes and binding without reserving work."""
        _require(_valid_id(run_id), 'invalid opaque local run ID')
        with self._transaction() as db:
            row = self._row(db, run_id)
            _require(row is not None, 'unknown local run ID')
            return self._verified(row)
