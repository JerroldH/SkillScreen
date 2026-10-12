"""Optional API baseline reruns. Plan is offline; score/submit/collect require explicit paid permission."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

from common import ROOT, CONFIG, read_json, rows, save, tokenizer, write_scores


def now():
    return datetime.now(timezone.utc).isoformat()


def select(method, protocol):
    return next(r for r in read_json(CONFIG / 'api.json')['models']
                if r['method'] == method and r['protocol'] == protocol)


def visible(tok, text):
    from input_and_typesafe import premise
    cfg = read_json(CONFIG / 'input.json')
    state, original, used = premise(tok, text, cfg['hypothesis'], cfg['max_length'])
    if not state.startswith('Premise: '):
        raise ValueError('Visible prefix lost premise marker')
    return state[len('Premise: '):], state, {'original_tokens': original, 'used_tokens': used,
        'truncated': original > used, 'state_sha256': hashlib.sha256(state.encode()).hexdigest()}


def prepare(args, config):
    records = rows(args.data_dir, args.protocol, 'test')
    if args.limit < 0:
        raise ValueError('Limit must be nonnegative; 0 explicitly selects all records')
    if args.limit:
        records = records[:args.limit]
    tok = tokenizer(args.tokenizer_dir)
    if 'fewshot' in args.model:
        import fewshot_prompt
        fewshot_prompt.configure(read_json(CONFIG / ('fewshot-' + args.protocol + '.json')),
                                 rows(args.data_dir, args.protocol, 'train'), tok)
    return records, tok


def safe_record(row, outcome):
    return {'benchmark_id': row['benchmark_id'], 'label': row['label'],
        'score': outcome.score, 'verdict': outcome.verdict,
        'unavailable_reason': outcome.unavailable_reason, 'model': outcome.returned_model,
        'input_tokens': outcome.input_tokens, 'output_tokens': outcome.output_tokens,
        'reasoning_tokens': outcome.reasoning_tokens}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan','tokenizer','validate-inputs','score','submit','collect'])
    parser.add_argument('--protocol', choices=['msd','sd'], default='msd')
    parser.add_argument('--model', choices=['typesafe','gpt-5.5','gpt-5.5-fewshot','gpt-5.5-reasoning',
        'grok-4.3','claude-haiku-4-5','gpt-5.4-mini'], default='gpt-5.5')
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--tokenizer-dir', type=Path, default=Path('outputs/tokenizer'))
    parser.add_argument('--output', type=Path, default=Path('outputs/api-rerun'))
    parser.add_argument('--limit', type=int, default=1, help='First N records for an explicit new run; 0 = complete test')
    parser.add_argument('--batch-state', type=Path)
    parser.add_argument('--allow-paid', action='store_true', help='Explicitly permit external paid API requests')
    args = parser.parse_args()
    if args.command == 'tokenizer':
        from huggingface_hub import hf_hub_download
        cfg = read_json(CONFIG / 'input.json')['base_model']
        args.tokenizer_dir.mkdir(parents=True, exist_ok=False)
        for name in ('tokenizer.json','tokenizer_config.json'):
            downloaded = hf_hub_download(repo_id=cfg['repo_id'], revision=cfg['revision'],
                filename=cfg['subfolder'] + '/' + name)
            shutil.copyfile(downloaded, args.tokenizer_dir / name)
        tokenizer(args.tokenizer_dir)
        print('Pinned tokenizer downloaded and encoding verified. No model/API request.')
        return
    config = read_json(CONFIG / 'typesafe.json')[args.protocol] if args.model == 'typesafe' else select(args.model,args.protocol)
    if args.command == 'plan':
        print(json.dumps({'command':'offline-plan','model':args.model,'protocol':args.protocol,
            'records':read_json(ROOT/'data/splits'/args.protocol/'summary.json')['prepared_counts']['test']['total'],
            'config':config,'paid_requests':False}, indent=2))
        return
    if not args.data_dir:
        parser.error('--data-dir is required outside plan/tokenizer')
    if args.command in ('score','submit','collect') and not args.allow_paid:
        parser.error('This is an independent paid rerun entry point. Add --allow-paid to authorize requests.')
    records, tok = prepare(args, config)
    if args.command == 'validate-inputs':
        total = 0
        for row in records:
            _, _, info = visible(tok, row['skill_text'])
            total += info['used_tokens']
        if not args.limit:
            expected = {'msd':3236492,'sd':2203448}[args.protocol]
            if total != expected:
                raise ValueError(f'Visible-token total changed: {total} != {expected}')
        print(json.dumps({'records':len(records),'used_tokens':total,'fewshot_verified':'fewshot' in args.model,
                          'paid_requests':False}))
        return
    if args.model == 'typesafe':
        if args.command != 'score':
            parser.error('TypeSafe uses score only')
        from input_and_typesafe import request_score
        key = os.environ.get('TYPESAFE_API_KEY','').strip()
        if not key:
            parser.error('Set TYPESAFE_API_KEY in the environment; do not put it in configs')
        args.output.mkdir(parents=True, exist_ok=False)
        def type_scores():
            for row in records:
                _, state, _ = visible(tok, row['skill_text'])
                response = request_score(key, config['model'], state, read_json(CONFIG/'input.json')['hypothesis'])
                yield {'benchmark_id':row['benchmark_id'],'label':row['label'], **response}
        write_scores(args.output/'scores.csv', type_scores())
        save(args.output/'run.json', {'model':args.model,'protocol':args.protocol,'records':len(records),
                                    'completed_at_utc':now(),'input_scope':'SKILL.md visible prefix'})
        return
    from llm_judge_providers import make_judge, Outcome, BATCH_FALLBACK_REASONS
    key = os.environ.get(config['provider']['key_env'],'').strip()
    if not key:
        parser.error('Set ' + config['provider']['key_env'] + ' in the environment')
    judge = make_judge(config['entry'],config['provider'],config['provider_name'],key)
    if args.command == 'submit':
        if config['entry']['mode']['test'] != 'batch':
            parser.error('Historical mode is synchronous; use score for this model')
        args.output.mkdir(parents=True, exist_ok=False)
        items = [(r['benchmark_id'], visible(tok,r['skill_text'])[0]) for r in records]
        batch_id, _ = judge.batch_submit(items)
        save(args.output/'batch.json', {'model':args.model,'protocol':args.protocol,'ids':[r['benchmark_id'] for r in records],
            'batch_id':batch_id,'submitted_at_utc':now(),'config_sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()})
        print('Batch submitted. Use collect --batch-state <output>/batch.json with the same model, protocol and limit.')
        return
    if args.command == 'collect':
        if not args.batch_state:
            parser.error('--batch-state is required')
        if (args.output/'scores.csv').exists() or (args.output/'run.json').exists():
            raise ValueError('Collection output already contains scores; use a fresh output directory')
        state = read_json(args.batch_state)
        if (state['model'] != args.model or state['protocol'] != args.protocol or
                state['ids'] != [r['benchmark_id'] for r in records] or
                state['config_sha256'] != hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()):
            raise ValueError('Batch state does not match current fixed inputs/configuration')
        status = judge.batch_status(state['batch_id'])
        if not status['finished']:
            print(json.dumps(status)); return
        args.output.mkdir(parents=True, exist_ok=True)
        outcomes = judge.batch_results(state['batch_id'])
        def batch_scores():
            for row in records:
                outcome = outcomes.get(row['benchmark_id']) or Outcome(unavailable_reason='batch_errored',mode='batch')
                if outcome.unavailable_reason in BATCH_FALLBACK_REASONS:
                    outcome = judge.judge(visible(tok,row['skill_text'])[0])
                yield safe_record(row,outcome)
        results = batch_scores()
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        results = (safe_record(row,judge.judge(visible(tok,row['skill_text'])[0])) for row in records)
    write_scores(args.output/'scores.csv', results)
    save(args.output/'run.json', {'model':args.model,'protocol':args.protocol,'records':len(records),
        'completed_at_utc':now(),'transport':'batch+original-sync-fallback' if args.command=='collect' else 'sync',
        'historical_transport':config['entry']['mode']['test'],
        'historical_prompt_sha256':config['prompt_sha256'],'raw_responses_saved':False})


if __name__ == '__main__':
    main()
