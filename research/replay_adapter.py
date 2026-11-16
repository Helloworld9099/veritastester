"""Explicit offline research replay. No provider integration, activation or downloads."""
import argparse, hashlib, json, os
from pathlib import Path

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

# Trust is selected by the checked-in experiment identity, never by a CLI manifest.
BASES = {
    'es': ('Helsinki-NLP/opus-mt-es-en', 'c96e2c5399ebfae4fc43d9669556b9afa74bb69d',
           'translation_training_20261116/model', False),
    'fr': ('Helsinki-NLP/opus-mt-fr-en', 'c4aed37b318c763fd177aa449b44e3b783cc6c02',
           'translation_fr_training_20261116/model', True),
}
EXPERIMENTS = {
    'translation_training_20261116': 'es',
    'translation_fr_training_20261116': 'fr',
    'neural_continuation_20261116/spanish': 'es',
    'neural_continuation_20261116/french': 'fr',
}
INERT_EXTRAS = {'download_manifest.json', 'repository_info.json'}


def checked_file(root, relative, checks):
    path = root / relative
    if path.is_symlink() or path.resolve() != path.absolute():
        raise ValueError('Symlinked research file is not permitted: ' + relative)
    entry = checks[relative]
    if not path.is_file() or path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
        raise ValueError('Research checksum mismatch: ' + relative)
    return path


def verify_base(model_dir, manifest):
    records = json.loads(manifest.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError('Base manifest must be a nonempty list')
    names = set()
    for row in records:
        if not isinstance(row, dict):
            raise ValueError('Invalid base manifest record')
        name = row.get('file')
        if not isinstance(name, str) or Path(name).name != name or name in {'.', '..'}:
            raise ValueError('Invalid base filename in manifest')
        if name in names:
            raise ValueError('Duplicate base filename in manifest')
        names.add(name)
    if model_dir.is_symlink() or model_dir.absolute() != model_dir.resolve():
        raise ValueError('Symlinked base directory is not permitted')
    if not model_dir.is_dir():
        raise ValueError('Missing base directory')
    # Reject unpinned loader sidecars and alternative weights, not just known ones.
    extras = {p.name for p in model_dir.iterdir()} - names - INERT_EXTRAS
    if extras:
        raise ValueError('Unexpected base files: ' + ', '.join(sorted(extras)))
    for row in records:
        name = row['file']
        p = model_dir / name
        if p.is_symlink() or p.resolve().parent != model_dir.resolve():
            raise ValueError('Symlinked base file is not permitted: ' + name)
        if not p.is_file() or p.stat().st_size != row['size_bytes'] or sha256(p) != row['sha256']:
            raise ValueError('Missing or mismatched pinned base file: ' + name)


def verify_inputs(root, experiment_dir, model_dir, supplied_manifest):
    root = root.resolve()
    experiment = experiment_dir.resolve().relative_to(root).as_posix()
    if experiment_dir.is_symlink() or experiment_dir.absolute() != experiment_dir.resolve():
        raise ValueError('Symlinked experiment directory is not permitted')
    if experiment not in EXPERIMENTS:
        raise ValueError('Unknown research experiment')
    model_id, revision, base_relative, safetensors = BASES[EXPERIMENTS[experiment]]
    checks = json.loads((root / 'CHECKSUMS.json').read_text())['files']
    adapter = checked_file(root, experiment + '/adapter_selected.safetensors', checks)
    protocol_path = checked_file(root, experiment + '/protocol.json', checks)
    trusted_manifest = checked_file(root, base_relative + '/download_manifest.json', checks)
    info_path = checked_file(root, base_relative + '/repository_info.json', checks)
    protocol = json.loads(protocol_path.read_text())
    info = json.loads(info_path.read_text())
    for identity in (protocol['model'], info):
        if identity.get('model') != model_id or identity.get('revision') != revision:
            raise ValueError('Pinned base model identity mismatch')
    if sha256(supplied_manifest) != sha256(trusted_manifest):
        raise ValueError('Base manifest does not match the selected adapter pinned manifest')
    targets = protocol.get('target_modules', protocol.get('lora', {}).get('target_modules'))
    expected_targets = {
        f'model.{side}.layers.{layer}.{attention}.{projection}'
        for side in ('encoder', 'decoder') for layer in range(6)
        for attention in (('self_attn',) if side == 'encoder' else ('self_attn', 'encoder_attn'))
        for projection in ('q_proj', 'v_proj')
    }
    if not isinstance(targets, list) or len(targets) != 36 or set(targets) != expected_targets:
        raise ValueError('Unexpected LoRA architecture')
    verify_base(model_dir, trusted_manifest)
    return adapter, protocol, safetensors

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-dir', type=Path, required=True)
    p.add_argument('--base-model', type=Path, required=True)
    p.add_argument('--base-manifest', type=Path, required=True,
                   help='Byte-for-byte copy of the selected experiment pinned download_manifest.json')
    p.add_argument('--text')
    p.add_argument('--baseline', action='store_true')
    p.add_argument('--verify-only', action='store_true')
    a = p.parse_args()
    if not a.verify_only and (not a.text or not a.text.strip()):
        p.error('--text is required for inference')
    root = Path(__file__).resolve().parent
    try:
        adapter, protocol, use_safetensors = verify_inputs(
            root, a.experiment_dir, a.base_model, a.base_manifest)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        p.error(str(exc))
    if a.verify_only:
        print('Pinned base files and selected adapter verified; nothing loaded or activated.')
        return
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    import torch
    from transformers import MarianTokenizer, MarianMTModel
    from safetensors.torch import load_file
    torch.set_num_threads(1)
    class LoRA(torch.nn.Module):
        def __init__(self, base):
            super().__init__(); self.base = base
            self.lora_A = torch.nn.Parameter(torch.zeros(8, base.in_features))
            self.lora_B = torch.nn.Parameter(torch.zeros(base.out_features, 8))
        def forward(self, x):
            y = self.base(x)
            return y if a.baseline else y + ((x @ self.lora_A.T) @ self.lora_B.T) * 2
    model = MarianMTModel.from_pretrained(str(a.base_model), local_files_only=True, weights_only=True, use_safetensors=use_safetensors)
    tokenizer = MarianTokenizer.from_pretrained(str(a.base_model), local_files_only=True)
    targets = protocol.get('target_modules', protocol.get('lora', {}).get('target_modules'))
    if not targets or len(targets) != 36:
        raise ValueError('Unexpected LoRA architecture')
    for name in targets:
        parent, attr = name.rsplit('.', 1)
        child = model.get_submodule(name)
        if not isinstance(child, torch.nn.Linear) or not name.endswith(('q_proj', 'v_proj')):
            raise ValueError('Unexpected adapter target')
        setattr(model.get_submodule(parent), attr, LoRA(child))
    weights = load_file(str(adapter)); state = model.state_dict()
    if set(weights) != {k for k in state if '.lora_' in k}:
        raise ValueError('Adapter key mismatch')
    with torch.no_grad():
        for name, tensor in weights.items():
            if tensor.shape != state[name].shape or not torch.isfinite(tensor).all():
                raise ValueError('Invalid adapter tensor')
            state[name].copy_(tensor)
        model.eval()
        inputs = tokenizer(a.text, return_tensors='pt', truncation=True, max_length=128)
        output = model.generate(**inputs, num_beams=4, max_new_tokens=96)
    print(json.dumps({'output': tokenizer.decode(output[0], skip_special_tokens=True),
                      'baseline': a.baseline,
                      'warning': 'Research-only context-free replay; no app activation or readiness guarantee.'}))

if __name__ == '__main__':
    main()
