"""Recover the exact frozen few-shot prompt from local prepared training text."""
import hashlib
import json

from llm_judge_prompt import SCHEMA, SYSTEM, USER_TEMPLATE, build_messages as zero_messages

PROMPT_VERSION = '1+fewshot'
INTRO = ("Below are labelled examples from the benchmark's training partition. Each shows a skill text, possibly cut at "
         "a length limit, followed by the correct answer. Use them to calibrate what this benchmark counts as malicious.")
_system = None
_hash = None


def configure(manifest, training_rows, tokenizer):
    global _system, _hash
    by_id = {r['benchmark_id']: r for r in training_rows}
    examples = []
    for reference in manifest['examples']:
        row = by_id[reference['benchmark_id']]
        if row['label'] != reference['label']:
            raise ValueError('Few-shot label differs from frozen training record')
        ids = tokenizer.encode(row['skill_text'], add_special_tokens=False)
        limit = manifest['max_tokens']
        excerpt = row['skill_text'] if len(ids) <= limit else tokenizer.decode(ids[:limit]) + '\n[truncated]'
        if hashlib.sha256(excerpt.encode()).hexdigest() != reference['excerpt_sha256']:
            raise ValueError('Few-shot excerpt differs from historical prompt')
        if len(ids) != reference['original_tokens']:
            raise ValueError('Few-shot tokenizer length differs')
        examples.append({**reference, 'text': excerpt})
    lines = [INTRO, '']
    for index, example in enumerate(examples, 1):
        answer = {'malicious': example['label'] == 1, 'probability': example['probability']}
        lines += [f'### Example {index}', '<skill_md>', example['text'], '</skill_md>',
                  'Answer: ' + json.dumps(answer, separators=(', ', ': ')), '']
    _system = SYSTEM + '\n\n' + '\n'.join(lines).rstrip()
    # Hash the actual recovered prompt content. The historical hash also includes
    # its now-omitted raw-text JSON file hash, retained separately in api.json.
    _hash = hashlib.sha256(json.dumps({'system': _system, 'user_template': USER_TEMPLATE,
        'schema': SCHEMA, 'version': PROMPT_VERSION}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return _hash


def build_messages(markdown):
    if _system is None:
        raise ValueError('Reconstruct training examples before using few-shot prompt')
    result = zero_messages(markdown)
    result['system'] = _system
    return result


def prompt_sha256():
    if _hash is None:
        raise ValueError('Few-shot prompt not initialized')
    return _hash
