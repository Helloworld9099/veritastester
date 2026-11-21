"""Offline aggregate persistence tests; no training, phase inputs or network."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import sqlite3
import threading

import pytest

from research.workflow_store import DB_NAME, DuplicatePlanError, StoreError, WorkflowStore


PLAN_PIN = '1' * 64


def _plan(pin=PLAN_PIN):
    return {'schema_version': 1, 'kind': 'local_research_workflow', 'plan_sha256': pin}


def _receipt(pin=PLAN_PIN, status='pass'):
    phases = {
        'data': {'status': 'pass', 'digest': 'a' * 64},
        'checkpoint': {'status': 'pass', 'digest': 'b' * 64},
        'comparison': {'status': status, 'digest': None if status == 'not_verified' else 'c' * 64},
    }
    return {'schema_version': 1, 'kind': 'local_research_workflow_receipt',
            'plan_sha256': pin, 'status': status, 'phases': phases}


def _hash(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def test_persists_verified_plan_and_receipt_envelopes(tmp_path):
    root = tmp_path / 'ledger'
    store = WorkflowStore(root)
    plan = _plan()
    reservation = store.reserve('local-01', plan=plan)
    assert reservation.created and reservation.plan_sha256 == PLAN_PIN
    plan['plan_sha256'] = '2' * 64
    before = WorkflowStore(root).audit('local-01')
    assert (before.state, before.status, before.receipt, before.receipt_sha256) == ('reserved', None, None, None)
    assert before.plan == _plan() and before.manifest_sha256 == _hash(_plan())
    receipt = _receipt(status='not_verified')
    after = store.complete(reservation, receipt=receipt)
    assert (after.state, after.status, after.plan_sha256) == ('completed', 'not_verified', PLAN_PIN)
    assert after.receipt == receipt and after.receipt_sha256 == _hash(receipt)
    after.receipt['phases']['checkpoint']['digest'] = None
    receipt['status'] = 'fail'
    assert WorkflowStore(root).audit('local-01') == after
    assert after.receipt == _receipt(status='not_verified')
    assert (root / DB_NAME).stat().st_mode & 0o077 == 0


def test_idempotent_reservation_never_retries_or_takes_over(tmp_path):
    store = WorkflowStore(tmp_path)
    first = store.reserve('spent', plan=_plan())
    duplicate = store.reserve('spent', plan=_plan())
    assert first.created and not duplicate.created
    assert 'owner' not in repr(first)
    with pytest.raises(StoreError, match='cannot be taken over'):
        store.complete(duplicate, receipt=_receipt())
    with pytest.raises(StoreError, match='different plan'):
        store.reserve('spent', plan=_plan('2' * 64))
    with pytest.raises(DuplicatePlanError) as caught:
        store.reserve('another-id', plan=_plan())
    assert caught.value.existing_run_id == 'spent'
    assert store.audit('spent').state == 'reserved'
    result = store.complete(first, receipt=_receipt())
    assert store.complete(first, receipt=_receipt()) == result
    assert store.complete(duplicate, receipt=_receipt()) == result
    with pytest.raises(StoreError, match='different immutable receipt'):
        store.complete(first, receipt=_receipt(status='fail'))
    assert store.reserve('fresh', plan=_plan('2' * 64)).created


def test_exact_plan_is_reserved_once_across_concurrent_ids(tmp_path):
    start = threading.Barrier(12)

    def reserve(i):
        start.wait(timeout=5)
        run_id = f'case-{i}'
        try:
            r = WorkflowStore(tmp_path).reserve(run_id, plan=_plan())
        except DuplicatePlanError as exc:
            return False, exc.existing_run_id
        return r.created, run_id

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(reserve, range(12)))
    assert sum(created for created, _ in results) == 1
    assert len({run_id for _, run_id in results}) == 1


def test_exact_plan_is_reserved_once_concurrently_under_same_id(tmp_path):
    start = threading.Barrier(8)

    def reserve(_):
        start.wait(timeout=5)
        return WorkflowStore(tmp_path).reserve('same-id', plan=_plan()).created

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(reserve, range(8))) == 1


def test_rejects_inline_payload_invalid_status_and_wrong_plan(tmp_path):
    store = WorkflowStore(tmp_path)
    with pytest.raises(StoreError):
        store.reserve('job', plan={**_plan(), 'path': '/private/input.json'})
    with pytest.raises(StoreError):
        store.reserve('../escape', plan=_plan())
    assert not (tmp_path / DB_NAME).exists()
    r = store.reserve('job', plan=_plan())
    payload = _receipt()
    payload['phases']['data']['data'] = 'not permissible'
    with pytest.raises(StoreError):
        store.complete(r, receipt=payload)
    wrong_status = _receipt(status='fail')
    wrong_status['status'] = 'pass'
    with pytest.raises(StoreError, match='outcomes'):
        store.complete(r, receipt=wrong_status)
    missing_digest = _receipt(status='fail')
    missing_digest['phases']['comparison']['digest'] = None
    with pytest.raises(StoreError, match='aggregate phase digest'):
        store.complete(r, receipt=missing_digest)
    with pytest.raises(StoreError, match='does not match reservation'):
        store.complete(r, receipt=_receipt(pin='0' * 64))
    assert store.audit('job').state == 'reserved'


def test_audit_rejects_checked_row_corruption(tmp_path):
    store = WorkflowStore(tmp_path)
    r = store.reserve('local', plan=_plan())
    store.complete(r, receipt=_receipt())
    with sqlite3.connect(store.database) as db:
        with pytest.raises(sqlite3.IntegrityError, match='append-only'):
            db.execute('UPDATE receipts SET receipt = ?', (b'{}',))
        # Simulate file tampering outside this API.
        db.execute('DROP TRIGGER receipts_no_update')
        db.execute('UPDATE receipts SET receipt = ?', (b'{}',))
    with pytest.raises(StoreError, match='checksum'):
        store.audit('local')
    with pytest.raises(StoreError, match='checksum'):
        store.reserve('local', plan=_plan())
    with pytest.raises(StoreError, match='checksum'):
        store.reserve('alternate', plan=_plan())


def test_audit_rejects_wrong_binding_even_with_updated_record_checksum(tmp_path):
    store = WorkflowStore(tmp_path)
    r = store.reserve('local', plan=_plan())
    store.complete(r, receipt=_receipt())
    other = _receipt(pin='2' * 64)
    raw = json.dumps(other, sort_keys=True, separators=(',', ':')).encode()
    with sqlite3.connect(store.database) as db:
        db.execute('DROP TRIGGER receipts_no_update')
        db.execute('UPDATE receipts SET receipt = ?, receipt_sha256 = ?', (raw, _hash(other)))
    with pytest.raises(StoreError, match='plan pin mismatch'):
        store.audit('local')


def test_audit_does_not_initialize_and_store_rejects_symlinks(tmp_path):
    missing = tmp_path / 'new'
    with pytest.raises(StoreError):
        WorkflowStore(missing).audit('missing')
    assert not missing.exists()
    real = tmp_path / 'real'
    real.mkdir()
    link = tmp_path / 'link'
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(StoreError, match='symlink'):
        WorkflowStore(link).reserve('local', plan=_plan())
    assert not list(real.iterdir())
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.write_bytes(b'unchanged')
    (real / DB_NAME).symlink_to(elsewhere)
    with pytest.raises(StoreError, match='regular database'):
        WorkflowStore(real).reserve('local', plan=_plan())
    assert elsewhere.read_bytes() == b'unchanged'


def test_refuses_to_repurpose_unrelated_sqlite_file(tmp_path):
    with sqlite3.connect(tmp_path / DB_NAME) as db:
        db.execute('CREATE TABLE existing(x)')
        db.execute('INSERT INTO existing VALUES (42)')
    with pytest.raises(StoreError, match='unrelated'):
        WorkflowStore(tmp_path).reserve('local', plan=_plan())
    with sqlite3.connect(tmp_path / DB_NAME) as db:
        assert db.execute('SELECT x FROM existing').fetchone() == (42,)
