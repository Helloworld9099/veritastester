"""Acquire only pinned public research training text; never test data or models."""
import hashlib, json, os, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parent
COMMIT = '12a34149267dd9ae98e60e0ec2e415707fad5618'
BASE = 'https://raw.githubusercontent.com/WMT-Chat-task/chat-task-2024-data/' + COMMIT + '/'
FILES = {'README.md': 'wmt24_README.md', 'LICENSE.md': 'wmt24_LICENSE.md', 'train/en-fr.csv': 'wmt24_train_en-fr.csv'}
EXPECTED_SHA256 = {
    'wmt24_README.md': 'cd508bb1f76d4f1a3e9b2a93dc5943c1c397f7e9e82e84491ed39afac38bdc56',
    'wmt24_LICENSE.md': '98679f478515c0f5c50e720006d12b4c3800d488e8dccca4e59a657efbb86c6e',
    'wmt24_train_en-fr.csv': 'eff96c25a2b5b5fc436ab85d573d4b2186fb8354a31763445c02e1cc80cc76ad'}
LIMIT = 8 * 1024 * 1024

def main():
    os.umask(0o077)
    raw = ROOT / 'raw'; raw.mkdir(exist_ok=True)
    result = {'repository': 'https://github.com/WMT-Chat-task/chat-task-2024-data', 'revision': COMMIT,
        'license': 'CC-BY-NC-4.0; publisher restricts use to research', 'commercial_use_allowed': False,
        'official_valid_test_acquired': False, 'max_file_bytes': LIMIT, 'files': []}
    prior_path = raw / 'download_manifest.json'
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else None
    for remote, local in FILES.items():
        path = raw / local; url = BASE + remote
        if not path.exists():
            request = urllib.request.Request(url, headers={'User-Agent': 'veritastester-context-preparation'})
            with urllib.request.urlopen(request, timeout=45) as response:
                data = response.read(LIMIT + 1)
            if len(data) > LIMIT: raise ValueError('Source exceeds acquisition budget')
            path.write_bytes(data)
        if path.stat().st_size > LIMIT: raise ValueError('Existing source exceeds byte budget')
        data = path.read_bytes(); sha = hashlib.sha256(data).hexdigest()
        if sha != EXPECTED_SHA256[local]: raise ValueError('Pinned source content mismatch: ' + local)
        if prior:
            expected = next(x for x in prior['files'] if x['file'] == local)
            if sha != expected['sha256']: raise ValueError('Existing source hash changed: ' + local)
        result['files'].append({'file': local, 'url': url, 'bytes': len(data), 'sha256': sha})
    prior_path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
