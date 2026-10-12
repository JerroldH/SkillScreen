"""Explicit optional GPU stages. --dry-run never imports ML packages or downloads anything."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent / 'original'
PROTOCOLS = {'msd': 'm_structural_disjoint', 'sd': 'source_disjoint'}


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def code_hashes(work):
    return {p.name: hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for p in sorted((work / 'scripts').glob('*.py'))}


def checked_config(args):
    family = 'skillscreen' if args.model == 'untuned' else args.model
    manifest = load(PACKAGE / 'configs' / family / 'runs.json')
    selected = [r for r in manifest['runs'] if (r['protocol'], r['seed']) == (args.protocol, args.seed)]
    if len(selected) != 1 or args.model == 'untuned' and args.seed != 42:
        raise ValueError('Choose a seed listed in this model/protocol manifest. Untuned uses seed 42 only.')
    config = load(PACKAGE / selected[0]['config'])
    if args.model == 'untuned':
        config['run_id'] = 'untuned-' + args.protocol
    return config


def paths(args, config):
    work = args.output_root.resolve() / args.model / args.protocol / f'seed-{args.seed}'
    base = work / 'outputs/base'
    if config['base_model'].get('subfolder'):
        base = base / config['base_model']['subfolder']
    return work, base, work / 'outputs' / config['run_id']


def child(work, script, argv):
    env = dict(os.environ, SKILLSCREEN_FULL_GPU='1', SKILLSCREEN_DEVICE='cuda')
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    subprocess.run([sys.executable, str(work / 'scripts' / script), *map(str, argv)],
                   cwd=work, env=env, check=True)


def activate(work):
    sys.path.insert(0, str(work / 'scripts'))
    os.environ['SKILLSCREEN_FULL_GPU'] = '1'
    os.environ['SKILLSCREEN_DEVICE'] = 'cuda'


def validate_data(source, config):
    summary = load(source / 'data/summary.json')
    if summary['dataset'] != config['dataset']:
        raise ValueError('Restored data and model configuration have different dataset revisions/protocols.')
    for partition in ('train', 'validation', 'test'):
        relative = Path(summary['files'][partition]['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Prepared data paths must stay inside their workspace.')
        if digest(source / relative) != summary['files'][partition]['sha256']:
            raise ValueError(f'Restored {partition} file failed its checksum.')
    return summary


def prepare(args, config, work, base, run_dir):
    source = (args.data_dir or PACKAGE / 'outputs/data' / args.protocol).resolve()
    summary = validate_data(source, config)
    # A new workspace prevents accidental changes to a previous run or another protocol.
    work.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source / 'data', work / 'data')
    shutil.copytree(SOURCE, work / 'scripts', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    kind = 'encoder' if args.model == 'modernbert' else 'direct'
    save(work / f'configs/{kind}.json', config)
    (work / 'outputs').mkdir(exist_ok=True)
    shutil.copyfile(PACKAGE / 'configs/skillscreen/large-input-profile.json', work / 'outputs/large-input-profile.json')
    if args.model != 'modernbert':
        shutil.copyfile(PACKAGE / f'configs/skillscreen/{args.protocol}-train-token-lengths.csv', work / 'outputs/train_token_lengths.csv')
    if args.base_dir:
        # Explicit local model reuse avoids a network request. Verify all supplied historical file hashes below.
        shutil.copytree(args.base_dir.resolve(), base)
    else:
        from huggingface_hub import snapshot_download
        model = config['base_model']
        prefix = (model.get('subfolder', '') + '/').lstrip('/')
        patterns = [prefix + name for name in ('*.json', '*.safetensors', '*.jinja', '*.txt', '*.model')]
        snapshot_download(repo_id=model['repo_id'], revision=model['revision'],
                          allow_patterns=patterns, local_dir=work / 'outputs/base')
    for name, checksum in config['base_model'].get('files_sha256', {}).items():
        if digest(base / name) != checksum:
            raise ValueError(f'Pinned model file differs: {name}')
    required = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
    if not all((base / name).is_file() for name in required) or not list(base.glob('*.safetensors')):
        raise ValueError('The pinned local model is missing config, tokenizer, or safetensors weights.')
    # Local OpenJev reuse must be byte-equivalent to the pinned snapshot, not merely similarly named.
    if args.base_dir and args.model != 'modernbert':
        raise ValueError('--base-dir is only supported for ModernBERT, whose complete historical base hashes are supplied. For OpenJev use the revision-pinned download.')
    save(work / 'preparation.json', {'model': args.model, 'protocol': args.protocol, 'seed': args.seed,
         'config_sha256': digest(work / f'configs/{kind}.json'), 'base_model': config['base_model'],
         'base_files': {str(p.relative_to(base)): digest(p) for p in sorted(base.rglob('*')) if p.is_file() and '.cache' not in p.parts},
         'runner_sha256': digest(Path(__file__)), 'scripts_sha256': code_hashes(work),
         'test_used_for_selection': False})
    activate(work)
    if args.model == 'modernbert':
        from transformers import AutoTokenizer
        from encoder_common import build_cache, check_tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
        check_tokenizer(tokenizer)
        build_cache(work, tokenizer, config)
    elif args.model == 'skillscreen':
        child(work, 'cache_direct_inputs.py', ['--base-dir', base, '--max-length', '8192', '--output', work / 'outputs/direct-inputs-8k'])
    else:
        # This native baseline has no training or validation-selected operating point.
        frozen = common_freeze(config, work)
        frozen.update(adapter=inspect_files(base, work), calibration={'slope': 1.0, 'intercept': 0.0, 'method': 'unused_for_untuned'},
                      operating_point={'threshold': 0.0, 'calibrated_probability_threshold': 0.5},
                      baselines={'Untuned OpenJev-2B': {'threshold': 0.5}},
                      policy='Native entailment score >= 0.5. No calibration or threshold fitting for this baseline.')
        save(work / 'frozen.json', frozen)
    print(f'Prepared {work}')


def verify_preparation(config, work):
    prepared = load(work / 'preparation.json')
    config_name = 'encoder' if config.get('kind') == 'encoder' else 'direct'
    if digest(work / f'configs/{config_name}.json') != prepared['config_sha256']:
        raise ValueError('Prepared run configuration changed.')
    if code_hashes(work) != prepared['scripts_sha256'] or digest(Path(__file__)) != prepared['runner_sha256']:
        raise ValueError('Prepared execution code changed.')
    base = work / 'outputs/base'
    if config['base_model'].get('subfolder'):
        base /= config['base_model']['subfolder']
    for relative, checksum in prepared['base_files'].items():
        if digest(base / relative) != checksum:
            raise ValueError(f'Prepared base file changed: {relative}')
    validate_data(work, config)


def train(args, config, work, base, run_dir):
    if args.model == 'untuned':
        raise ValueError('Untuned OpenJev has no train stage.')
    verify_preparation(config, work)
    script = 'train_encoder.py' if args.model == 'modernbert' else 'train_direct.py'
    argv = ['--base-dir', base, '--output', run_dir, '--stage', 'train', '--code-commit', 'repro-' + digest(Path(__file__))]
    child(work, script, argv)


def inspect_files(directory, work):
    return {'directory': directory.relative_to(work).as_posix(), 'files': {
        p.name: {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(directory.iterdir()) if p.is_file()}}


def common_freeze(config, work):
    summary = load(work / 'data/summary.json')
    return {'run_id': config['run_id'], 'seed': config['seed'], 'base_model': config['base_model'],
            'dataset': config['dataset'], 'input': config.get('input') or {
                'scope': 'SKILL.md only', 'max_length': 8192, 'hypothesis': config['hypothesis'],
                'template': 'Premise: {skill_text}\nHypothesis: {hypothesis}',
                'truncation': 'Premise prefix plus complete fixed hypothesis'},
            'test_file_sha256': summary['files']['test']['sha256'],
            'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
            'evaluation_code_commit': 'repro-' + digest(Path(__file__)),
            'evaluation_scripts_sha256': code_hashes(work)}


def calibrate(args, config, work, base, run_dir):
    if args.model == 'untuned':
        raise ValueError('Untuned uses the native 0.5 rule; no calibration stage.')
    verify_preparation(config, work)
    if (work / 'frozen.json').exists():
        raise ValueError('Refusing to overwrite a previous freeze.')
    activate(work)
    import numpy as np
    from calibrate_direct import fit_sigmoid, load_scores, probabilities, read_partition
    from word_svm import recall_gate
    if args.model == 'modernbert':
        from encoder_common import log_odds
    else:
        from calibrate_direct import log_odds
    run = load(run_dir / 'run.json')
    summary = load(work / 'data/summary.json')
    total = math.ceil(summary['prepared_counts']['train']['total'] / 16)
    if run['status'] != 'completed' or run['completed_updates'] != total:
        raise ValueError('The full one-epoch run must complete before freezing.')
    if run['seen_training_records'] != summary['prepared_counts']['train']['total']:
        raise ValueError('Training did not consume the complete training partition.')
    events = [json.loads(line) for line in (run_dir / 'training.jsonl').read_text().splitlines() if line.strip()]
    if [e['update'] for e in events] != list(range(1, total + 1)):
        raise ValueError('Training log is incomplete or contains duplicate updates.')
    if not all(math.isfinite(e[k]) for e in events for k in ('loss', 'grad_norm')):
        raise ValueError('Training contained non-finite losses or gradients.')
    best = min(run['validation'], key=lambda m: (m['validation_recall_gates'][1]['fpr'], -m['average_precision'], m['loss']))
    if run['selected_update'] != best['update']:
        raise ValueError('Selected checkpoint differs from the predefined validation selection rule.')
    scores_path = run_dir / f"validation-step-{best['update']}.csv"
    if digest(scores_path) != best['scores_sha256']:
        raise ValueError('Selected validation predictions changed.')
    expected, validation_sha = read_partition('validation')
    rows = load_scores(scores_path, expected)
    labels = np.array([r['label'] for r in expected])
    values = log_odds(rows)
    calibration = fit_sigmoid(values, labels)
    gate = recall_gate(labels, values, 0.99)
    if args.model == 'modernbert' and gate['threshold'] != best['validation_recall_gates'][1]['threshold']:
        raise ValueError('Encoder log-odds gate differs from the validation-selected gate.')
    selected = run_dir / run['selected_checkpoint' if args.model == 'modernbert' else 'selected_adapter']
    if not selected.is_dir() or not list(selected.glob('*.safetensors')):
        raise ValueError('Selected checkpoint export is missing.')
    frozen = common_freeze(config, work)
    frozen.update(selected_update=best['update'], training_code_commit=run['code_commit'],
                  cache_manifest_sha256=run['cache_manifest_sha256'], validation_file_sha256=validation_sha,
                  validation_scores_file=scores_path.relative_to(work).as_posix(),
                  validation_scores_sha256=digest(scores_path), calibration=calibration,
                  operating_point={**gate, 'decision_variable': 'two_way_log_odds' if args.model == 'modernbert' else 'raw_log_odds',
                      'equivalent_raw_probability_threshold': float(probabilities(gate['threshold'])),
                      'calibrated_probability_threshold': float(probabilities(calibration['slope'] * gate['threshold'] + calibration['intercept']))},
                  policy='Fit calibration and threshold on validation only. Recheck checkpoint hashes and replay validation logits before reading held-out test.')
    frozen['checkpoint' if args.model == 'modernbert' else 'adapter'] = inspect_files(selected, work)
    if args.model == 'modernbert':
        frozen.update(variant=config['variant'], model_name=config['model_name'], report_model=config['report_model'])
    save(work / 'frozen.json', frozen)
    print(json.dumps({'selected_update': best['update'], 'validation_gate': gate, 'frozen_sha256': digest(work / 'frozen.json')}, indent=2))


def evaluate(args, config, work, base, run_dir):
    verify_preparation(config, work)
    frozen = work / 'frozen.json'
    target = work / 'outputs/test'
    argv = ['--frozen', frozen, '--frozen-sha256', digest(frozen), '--output', target]
    if args.model == 'modernbert':
        child(work, 'score_encoder.py', ['score', *argv])
    else:
        child(work, 'score_frozen_direct.py', ['--base-dir', base, '--model', 'untuned' if args.model == 'untuned' else 'direct', *argv])
    print(f'Predictions: {target / "scores.csv"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('skillscreen', 'modernbert', 'untuned'), required=True)
    parser.add_argument('--protocol', choices=PROTOCOLS, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stage', choices=('prepare', 'train', 'calibrate', 'evaluate'), required=True)
    parser.add_argument('--data-dir', type=Path, help='Restored protocol workspace, e.g. outputs/data/msd')
    parser.add_argument('--output-root', type=Path, default=PACKAGE / 'outputs/gpu')
    parser.add_argument('--base-dir', type=Path, help='Optional ModernBERT local snapshot; every pinned base hash is checked')
    parser.add_argument('--dry-run', action='store_true', help='Print the selected recipe without imports, file writes, GPU use, or network')
    args = parser.parse_args()
    try:
        config = checked_config(args)
    except ValueError as error:
        parser.error(str(error))
    if args.model == 'untuned' and args.stage in ('train', 'calibrate'):
        parser.error('Untuned supports only prepare and evaluate.')
    if args.base_dir and args.model != 'modernbert':
        parser.error('--base-dir is supported only for ModernBERT; OpenJev uses the pinned Hub revision.')
    work, base, run_dir = paths(args, config)
    if args.dry_run:
        print(json.dumps({'model': args.model, 'protocol': args.protocol, 'seed': args.seed, 'stage': args.stage,
            'workspace': str(work), 'run_id': config['run_id'], 'base_model': config['base_model'],
            'data_workspace': str(args.data_dir or PACKAGE / 'outputs/data' / args.protocol),
            'effects_if_executed': {'prepare': 'Create separate workspace; download pinned base; tokenize train/validation only (untuned skips cache).',
                'train': 'Run exactly one training epoch on the selected seed. No test evaluation.',
                'calibrate': 'Check completed training and selected validation scores; freeze calibration and 99%-validation-recall threshold.',
                'evaluate': 'Verify frozen model/code/cache; replay validation logits; score all held-out test rows. Untuned uses native 0.5.'}[args.stage]}, indent=2))
        return
    if sys.platform == 'win32':
        parser.error('GPU reruns use Linux/WSL2 with CUDA; the original scripts use POSIX resource accounting. --dry-run works everywhere.')
    {'prepare': prepare, 'train': train, 'calibrate': calibrate, 'evaluate': evaluate}[args.stage](args, config, work, base, run_dir)


if __name__ == '__main__':
    main()
