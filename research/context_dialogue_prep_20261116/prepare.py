"""Bounded offline context preparation. No training, provider calls, or invented labels."""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, os, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get('VERITASTESTER_DATA_WORKSPACE', str(ROOT.parent))).resolve()
DATA = WORKSPACE / 'veritastester-data-prep'
SEED = 'context-dialogue-v1:20261116'
MAX_ROWS = {'train': 2048, 'dev': 256}
MAX_PER_GROUP = 16
DEV_GROUPS = 24
MAX_HISTORY = 6
MAX_TEXT_CHARS = 768

def norm(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())

def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def rows(path):
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + '\n')

def write_jsonl(path, values):
    with path.open('w', encoding='utf-8') as f:
        for row in values: f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')

def protected_text():
    protected = set()
    paths = []
    for name in ['tatoeba.fr-en.dev.jsonl', 'tatoeba.fr-en.test.jsonl']:
        paths.append(DATA / 'private' / name)
    for pair in ('fr-en', 'en-fr'):
        paths += [DATA / 'private' / f'diabla.{pair}.inputs.test.jsonl',
                  DATA / 'private' / f'diabla.{pair}.references.test.jsonl']
    for path in paths:
        for r in rows(path):
            texts = [r.get('utterance', ''), r.get('reference_translation', ''), *r.get('previous', [])]
            protected.update(norm(t) for t in texts if t.strip())
    return protected, paths

def holdout_registry():
    groups = {r['group_id'] for r in rows(DATA / 'private/diabla.groups.test.jsonl')}
    smoke = set()
    for pair in ('fr-en', 'en-fr'):
        smoke |= {r['group_id'] for r in rows(DATA / 'private' / f'diabla.{pair}.smoke.inputs.test.jsonl')}
    guard_path = WORKSPACE / 'translation_fr_training_20261116/diabla_guard.jsonl'
    guard = {r['group_id'] for r in rows(guard_path)}
    assert smoke <= groups and guard <= groups and not smoke & guard
    continued_path = WORKSPACE / 'translation_continued_20261116/french/guard.jsonl'
    continued = {r['group'] for r in rows(continued_path)}
    assert continued <= groups and not continued & (smoke | guard)
    fresh = groups - smoke - guard - continued
    return {'dataset': 'DiaBLa', 'train_allowed': False,
        'group_count': len(groups), 'smoke_groups': sorted(smoke),
        'consumed_french_pilot_guard_groups': sorted(guard),
        'consumed_continued_french_guard_groups': sorted(continued), 'unscored_candidate_groups': sorted(fresh),
        'unscored_group_count': len(fresh),
        'status': 'ALL144 dialogues consumed or reserved:40 smoke+64 original pilot+40 continued pilot. None is a fresh independent test; retain as regression only.',
        'direction_policy': 'A dialogue is one group across BOTH directions.',
        'all_dialogues_remain_evaluation_only': True}

def input_snapshot():
    handoff = json.loads((DATA / 'HANDOFF_MANIFEST.json').read_text())
    snapshot = {}
    for item in handoff['files']:
        path = DATA / item['path']
        actual = sha(path)
        if actual != item['sha256']: raise ValueError('Original handoff changed: ' + str(path))
        snapshot[str(path.relative_to(WORKSPACE))] = actual
    paths = [DATA / 'HANDOFF_MANIFEST.json',
        WORKSPACE / 'translation_training_20261116/NEURAL_TRAINING_REPORT.md',
        WORKSPACE / 'translation_fr_training_20261116/data_manifest.json',
        WORKSPACE / 'translation_fr_training_20261116/diabla_guard.jsonl',
        WORKSPACE / 'translation_continued_20261116/french/guard.jsonl',
        WORKSPACE / 'translation_continued_20261116/french/data_manifest.json',
        WORKSPACE / 'translation_continued_20261116/REPORT.md',
        WORKSPACE / 'translation_independent_revalidation_20261116/registration.json']
    for path in paths: snapshot[str(path.relative_to(WORKSPACE))] = sha(path)
    source = json.loads((ROOT / 'raw/download_manifest.json').read_text())
    for item in source['files']:
        path = ROOT / 'raw' / item['file']
        if sha(path) != item['sha256']: raise ValueError('Pinned new source changed')
        snapshot[str(path.relative_to(WORKSPACE))] = item['sha256']
    return snapshot

def load_source():
    path = ROOT / 'raw/wmt24_train_en-fr.csv'
    docs = collections.defaultdict(list)
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        expected = {'source_language', 'target_language', 'source', 'reference', 'doc_id', 'sender', 'client_id'}
        if set(reader.fieldnames or []) != expected: raise ValueError('Unexpected WMT schema')
        for index, r in enumerate(reader, 1):
            if (r['source_language'], r['target_language']) not in {('en', 'fr'), ('fr', 'en')}:
                raise ValueError('Unexpected direction')
            r['source_row'] = index; r['turn_index'] = len(docs[r['doc_id']])
            docs[r['doc_id']].append(r)
    if sum(map(len, docs.values())) > 20000: raise ValueError('Raw row budget exceeded')
    return docs

def make_example(r, doc_id, split, history, source_sha):
    cid = 'wmt24-en-fr:row:' + str(r['source_row'])
    return {'schema': 'context_dialogue_input_v1', 'case_id': cid,
        'group_id': 'wmt24-dialogue:' + digest(doc_id)[:24], 'split': split, 'original_split': 'train',
        'source_row': r['source_row'], 'turn_index': r['turn_index'], 'dataset': 'WMT24-Chat',
        'language': 'fr', 'target_language': 'en', 'utterance': r['source'],
        'previous': list(history[-MAX_HISTORY:]), 'situation': '',
        'history_policy': 'Earlier original source utterances in both languages; no reference translations. Access to both original channels assumed.',
        'synthetic': False, 'train_allowed': split == 'train', 'research_only': True,
        'commercial_use_allowed': False, 'license': 'CC-BY-NC-4.0',
        'attribution': 'Mohammed et al. (2024), Findings of WMT24 Chat Translation Shared Task; Unbabel; see retained README and LICENSE.',
        'source_revision': '12a34149267dd9ae98e60e0ec2e415707fad5618', 'source_sha256': source_sha}

def request_payload(row, arm='all_history', selected_indices=()):
    """Allow-list construction: never copy reference/labels/rights fields to a model."""
    if arm not in ('no_context', 'all_history', 'selected_history', 'selected_history_plus_scene'):
        raise ValueError('Unknown evaluation arm')
    if any(not isinstance(i, int) or i < 0 or i >= len(row['previous']) for i in selected_indices):
        raise ValueError('Invalid selected context index')
    history = [] if arm == 'no_context' else row['previous']
    if arm.startswith('selected_history'): history = [history[i] for i in selected_indices]
    return {'language': row['language'], 'target_language': row['target_language'],
        'utterance': row['utterance'],
        'previous': [{'text': h['text'], 'language': h['language'], 'speaker': h['speaker']} for h in history],
        'situation': row['situation'] if arm == 'selected_history_plus_scene' else ''}

def prepare(output):
    os.umask(0o077)
    output = output.resolve()
    if output.parent != ROOT or output.name in ('raw',): raise ValueError('Output must be a new direct child of this package')
    if output.exists(): raise ValueError('Refusing to overwrite an existing artifact directory')
    before = input_snapshot()
    external, protected_paths = protected_text()
    registry = holdout_registry()
    docs = load_source(); ordered = sorted(docs, key=lambda d: digest(SEED + ':' + d))
    dev_ids = set(ordered[:DEV_GROUPS])
    # Preserve all original-source/reference strings in reserved dev dialogues,
    # even if a row is later excluded or omitted by the bounded sample.
    dev_text = {norm(r[k]) for d in dev_ids for r in docs[d] for k in ('source', 'reference') if r[k].strip()}
    result = {}; rejected = collections.Counter()
    source_sha = sha(ROOT / 'raw/wmt24_train_en-fr.csv')
    for split in ('dev', 'train'):
        blocked = external | (dev_text if split == 'train' else set())
        candidates = []
        for doc_id in ordered:
            if (doc_id in dev_ids) != (split == 'dev'): continue
            history = []
            for r in docs[doc_id]:
                source, reference = r['source'], r['reference']
                valid_text = source.strip() and reference.strip() and max(len(source), len(reference)) <= MAX_TEXT_CHARS
                if r['source_language'] == 'fr':
                    reason = ('empty_or_long' if not valid_text else
                              'protected_exact_text' if norm(source) in blocked or norm(reference) in blocked else
                              'no_safe_history' if not history else '')
                    if reason: rejected[split + ':' + reason] += 1
                    else:
                        inp = make_example(r, doc_id, split, history, source_sha)
                        ref = {'case_id': inp['case_id'], 'group_id': inp['group_id'], 'split': split,
                            'reference_translation': reference, 'reference_type': 'source-published-reference',
                            'label_kind': 'translation_not_memory_utility_or_exact_slang_sense',
                            'license': inp['license'], 'attribution': inp['attribution'], 'source_sha256': source_sha}
                        candidates.append((inp, ref))
                # Current source is appended ONLY AFTER producing the current input.
                # Preserve original row indices even when intervening boilerplate is removed.
                if source.strip() and len(source) <= MAX_TEXT_CHARS and norm(source) not in blocked:
                    history.append({'source_row': r['source_row'], 'turn_index': r['turn_index'],
                        'text': source, 'language': r['source_language'], 'speaker': r['sender']})
        count = collections.Counter(); selected = []
        for inp, ref in sorted(candidates, key=lambda pair: digest(SEED + ':' + pair[0]['case_id'])):
            if count[inp['group_id']] >= MAX_PER_GROUP: continue
            selected.append((inp, ref)); count[inp['group_id']] += 1
            if len(selected) >= MAX_ROWS[split]: break
        result[split] = selected
    if len(result['train']) < 128 or len(result['dev']) < 32:
        raise ValueError('Too little data after strict exclusions; do not weaken exclusions silently')
    output.mkdir(mode=0o700)
    for split, selected in result.items():
        write_jsonl(output / (split + '.inputs.jsonl'), (x[0] for x in selected))
        write_jsonl(output / (split + '.references.jsonl'), (x[1] for x in selected))
    write_json(output / 'holdout_registry.json', registry)
    write_json(output / 'original_inputs_snapshot.json', before)
    # Real train-only review candidates; no inferred utility labels or scenes.
    train = [x[0] for x in result['train']]; queue = []
    for i, current in enumerate(train[:32]):
        alternate = next(x for x in train[i+1:] + train[:i] if x['group_id'] != current['group_id'])
        for origin, candidate, candidate_group in (
            ('earlier_same_dialogue_unjudged', current['previous'][-1]['text'], current['group_id']),
            ('other_training_dialogue_unjudged', alternate['utterance'], alternate['group_id'])):
            queue.append({'case_id': current['case_id'] + ':' + origin, 'group_id': current['group_id'],
                'candidate_group_id': candidate_group, 'split': 'train', 'language': 'fr',
                'utterance': current['utterance'], 'previous': current['previous'], 'candidate_memory': candidate,
                'candidate_origin': origin, 'situation': '', 'reference_translation': '',
                'human_judgement': None, 'label_status': 'UNLABELLED_NOT_TRAINING_DATA',
                'license': 'CC-BY-NC-4.0', 'research_only': True, 'attribution': current['attribution'],
                'source_case_id': current['case_id'], 'source_sha256': source_sha})
    write_jsonl(output / 'memory_review_queue.UNLABELLED.jsonl', queue)
    plan = {'version': 1, 'status': 'PREPARATION_ONLY_NO_TRAINING_OR_MODEL_EVALUATION',
        'primary_language': 'fr-en', 'seed': SEED, 'max_training_rows': MAX_ROWS['train'],
        'max_development_rows': MAX_ROWS['dev'], 'history_turn_cap': MAX_HISTORY,
        'training': {'allowed_inputs': 'train.inputs.jsonl + joined train.references.jsonl for target loss only',
            'selection_only': 'dev.inputs.jsonl + dev.references.jsonl',
            'forbidden': ['DiaBLa in training/glossary/tuning', 'test references in model requests',
                'future turns', 'gold translations in history', 'invented utility/sense labels'],
            'recipe': 'Next experiment only after a separate explicit run plan. This package starts no run.',
            'memory_supervision_ready': False, 'commercial_use_allowed': False},
        'evaluation': {'fresh_dialogue_groups': registry['unscored_group_count'],
            'holdout_status': 'No fresh DiaBLa groups. Existing DiaBLa is regression-only. Official WMT24 valid/test are unacquired future candidates, requiring split/group/text checks BEFORE any next fitting.',
            'arms': ['no_context', 'all_history', 'selected_history', 'selected_history_plus_scene'],
            'controls': 'Same cases/base checkpoint/decoding/current-utterance token budget; history cap6, selector cap2; charge selector time. Freeze configuration and case membership before any holdout inference.',
            'primary': 'Bilingual blinded paired meaning-preservation ratings with context; harmful changes adjudicated.',
            'secondary': ['corpus chrF', 'corpus BLEU', 'paired bootstrap by dialogue for mean sentence-chrF difference',
                'human context-sensitive accuracy: antecedents, gender/formality, ellipsis, lexical consistency',
                'helpful rejection and harmful selection only on independently labelled memories',
                'median/p95 latency, context tokens, truncation counts, abstention and failure rates'],
            'bootstrap': {'resample_unit': 'whole dialogue, keeping both directions together', 'replicates': 2000,
                'seed': 20261116, 'interval': '2.5th/97.5th percentiles',
                'warning': 'Mean sentence-chrF intervals do not estimate corpus chrF deltas.'},
            'freeze_log': 'Record checkpoint/tokenizer/selector hashes, chosen thresholds, decoding, exact case IDs, source/reference hashes, context policy, all arms and predeclared acceptance rule before accessing gold.',
            'acceptance': 'Pre-register a human meaningful-gain target and noninferiority margins with reviewers; no promotion until independent context-sensitive benefit and harm/latency limits pass. No fresh local DiaBLa set remains; acquire a new independent conversational test before any claim of dialogue improvement.',
            'stress_cases': ['misleading older memory', 'empty history', 'wrong scene', 'session boundary reset',
                'code switching', 'negation', 'source-text prompt injection'],
            'human_review': 'Two fluent reviewers blinded to model/arm; independent adjudication; no candidate outputs when authoring references. No claim of utility from adjacent/different-dialogue membership.'}}
    write_json(output / 'evaluation_plan.json', plan)
    audit = {'status': 'prepared', 'raw_rows': sum(map(len, docs.values())), 'raw_dialogues': len(docs),
        'raw_french_to_english_rows': sum(r['source_language'] == 'fr' for rs in docs.values() for r in rs),
        'reserved_dev_dialogues': DEV_GROUPS, 'excluded_strings_external': len(external),
        'rejected': dict(rejected), 'splits': {s: {'rows': len(rs), 'groups': len({r['group_id'] for r, _ in rs}),
            'history_items': sum(len(r['previous']) for r, _ in rs)} for s, rs in result.items()},
        'memory_review_unlabelled_rows': len(queue), 'untouched_dialogue_candidate_groups': registry['unscored_group_count'],
        'sampling': 'SHA256 case ranking, max16 rows per dialogue, max2048 train/max256 dev. No scores used.',
        'history_filter': 'Exact protected strings removed from previous original-source turns; gaps retain original indices. Never fills gaps with gold/MT guesses.',
        'limitations': ['conversation-disjoint; speaker identity unavailable', 'client/domain/template identity can recur',
            'exact normalization cannot exclude paraphrases or base pretraining exposure',
            'filtered data differs from official WMT distributions', 'bilingual source-channel history differs from DiaBLa speaker-visible MT history',
            'WMT24 README reports15027 train turns; actual pinned CSV parsed to15026 records; use measured count']}
    write_json(output / 'data_audit.json', audit)
    after = input_snapshot()
    if after != before: raise ValueError('Original inputs changed during preparation')
    manifest = {'source_inputs': before, 'artifacts': {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)}
        for p in sorted(output.iterdir()) if p.is_file()}, 'original_sources_unchanged': True}
    write_json(output / 'manifest.json', manifest)
    print(json.dumps(audit, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, default=ROOT / 'artifacts')
    prepare(parser.parse_args().output)
