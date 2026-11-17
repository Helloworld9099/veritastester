"""Offline delivery integrity, not translation-quality or live readiness tests."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('research_replay', ROOT/'research/replay_adapter.py')
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def test_research_manifest_matches_all_delivered_files():
    root = ROOT/'research'
    manifest = json.loads((root/'CHECKSUMS.json').read_text())['files']
    actual = {str(p.relative_to(root)) for p in root.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.name != 'CHECKSUMS.json'}
    assert actual == set(manifest)
    for name, entry in manifest.items():
        p = root/name
        assert p.stat().st_size == entry['bytes']
        assert replay.sha256(p) == entry['sha256']


def test_replay_help_requires_no_neural_dependencies():
    result = subprocess.run([sys.executable, '-S', str(ROOT/'research/replay_adapter.py'), '--help'],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0
    assert '--verify-only' in result.stdout


def test_base_verification_rejects_corruption(tmp_path):
    model = tmp_path/'model'; model.mkdir()
    p = model/'config.json'; p.write_text('{}')
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps([{'file':'config.json','size_bytes':2,
                                     'sha256':hashlib.sha256(b'{}').hexdigest()}]))
    replay.verify_base(model, manifest)
    p.write_text('[]')
    with pytest.raises(ValueError, match='mismatched'):
        replay.verify_base(model, manifest)


def test_base_manifest_rejects_traversal(tmp_path):
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps([{'file':'../outside','size_bytes':0,'sha256':'x'}]))
    with pytest.raises(ValueError, match='Invalid base filename'):
        replay.verify_base(tmp_path, manifest)


@pytest.mark.parametrize('records', [[], {}, [{'file': 'x'}, {'file': 'x'}]])
def test_base_rejects_empty_or_duplicate_manifest(tmp_path, records):
    manifest = tmp_path/'manifest.json'; manifest.write_text(json.dumps(records))
    with pytest.raises(ValueError):
        replay.verify_base(tmp_path, manifest)


def test_empty_manifest_cannot_bypass_verification_before_imports(tmp_path):
    manifest = tmp_path/'empty.json'; manifest.write_text('[]')
    result = subprocess.run([sys.executable, '-S', str(ROOT/'research/replay_adapter.py'),
        '--experiment-dir', str(ROOT/'research/neural_continuation_20261116/french'),
        '--base-model', str(tmp_path/'absent'), '--base-manifest', str(manifest), '--verify-only'],
        capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert 'does not match' in result.stderr
    assert 'torch' not in result.stderr and 'transformers' not in result.stderr


@pytest.mark.parametrize('experiment,other', [
    ('neural_continuation_20261116/french', 'translation_training_20261116'),
    ('neural_continuation_20261116/spanish', 'translation_fr_training_20261116')])
def test_cross_language_manifest_rejected_before_imports(tmp_path, experiment, other):
    result = subprocess.run([sys.executable, '-S', str(ROOT/'research/replay_adapter.py'),
        '--experiment-dir', str(ROOT/'research'/experiment), '--base-model', str(tmp_path),
        '--base-manifest', str(ROOT/'research'/other/'model/download_manifest.json'), '--verify-only'],
        capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert 'does not match' in result.stderr
    assert 'torch' not in result.stderr and 'transformers' not in result.stderr


@pytest.mark.parametrize('extra', ['model.safetensors', 'generation_config.json', 'tokenizer.json'])
def test_base_rejects_unpinned_loader_files(tmp_path, extra):
    model = tmp_path/'model'; model.mkdir()
    (model/'config.json').write_text('{}')
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps([{'file':'config.json','size_bytes':2,
        'sha256':hashlib.sha256(b'{}').hexdigest()}]))
    (model/extra).write_text('{}')
    with pytest.raises(ValueError, match='Unexpected base files'):
        replay.verify_base(model, manifest)


def test_base_rejects_symlink(tmp_path):
    model = tmp_path/'model'; model.mkdir()
    external = tmp_path/'external'; external.write_text('{}')
    (model/'config.json').symlink_to(external)
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps([{'file':'config.json','size_bytes':2,
        'sha256':hashlib.sha256(b'{}').hexdigest()}]))
    with pytest.raises(ValueError, match='Symlinked base file'):
        replay.verify_base(model, manifest)


def test_unknown_experiment_rejected(tmp_path):
    root = tmp_path/'research'; root.mkdir()
    unknown = root/'other'; unknown.mkdir()
    with pytest.raises(ValueError, match='Unknown research experiment'):
        replay.verify_inputs(root, unknown, tmp_path, tmp_path/'manifest')


def test_checked_research_file_rejects_tampering(tmp_path):
    path = tmp_path/'protocol.json'; path.write_text('{}')
    checks = {'protocol.json': {'bytes':2, 'sha256':hashlib.sha256(b'{}').hexdigest()}}
    assert replay.checked_file(tmp_path, 'protocol.json', checks) == path
    path.write_text('[]')
    with pytest.raises(ValueError, match='checksum mismatch'):
        replay.checked_file(tmp_path, 'protocol.json', checks)


@pytest.mark.parametrize('change', ['empty', 'truncated', 'duplicate'])
def test_modified_caller_manifest_cannot_select_trust(tmp_path, change):
    root = ROOT/'research'
    original = json.loads((root/'translation_fr_training_20261116/model/download_manifest.json').read_text())
    records = [] if change == 'empty' else (original[:-1] if change == 'truncated' else original + original[:1])
    manifest = tmp_path/'manifest.json'; manifest.write_text(json.dumps(records))
    with pytest.raises(ValueError, match='does not match'):
        replay.verify_inputs(root, root/'neural_continuation_20261116/french', tmp_path, manifest)
