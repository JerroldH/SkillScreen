"""Inference with the existing released SkillScreen checkpoints and input recipes."""
import json
from pathlib import Path

from .package_inputs import encode_package


ROOT = Path(__file__).resolve().parents[1]
HYPOTHESIS = 'This AI agent skill is malicious.'


def encode_input(tokenizer, row, mode, max_length):
    if mode not in ('package', 'markdown'):
        raise ValueError('Choose package or markdown input mode.')
    if tokenizer.num_special_tokens_to_add(False) != 0:
        raise ValueError('The released model requires a tokenizer without added special tokens.')
    if mode == 'package':
        return encode_package(tokenizer, row, HYPOTHESIS, max_length)
    suffix = '\nHypothesis: ' + HYPOTHESIS
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    if type(max_length) is not int or max_length <= len(suffix_ids) + 8:
        raise ValueError('Input budget is too small for the premise and fixed hypothesis.')
    ids = tokenizer.encode('Premise: ' + row['skill_text'] + suffix, add_special_tokens=False)
    original = len(ids)
    if original > max_length:
        ids = ids[:max_length - len(suffix_ids)] + suffix_ids
    return ids, original, {'original_tokens': original, 'used_tokens': len(ids),
                           'truncated': original > max_length, 'files': ['SKILL.md']}


def score_model(row, mode):
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ValueError('Model inference needs requirements/inference.txt. Package inspection and dependency checks need only Python.') from exc
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('The released inference recipe requires a CUDA GPU with BF16 support. Use --inspect or --checks dependencies on this machine.')
    spec = json.loads((ROOT / 'configs/community-models.json').read_text(encoding='utf-8'))[mode]
    options = {'revision': spec['revision'], 'trust_remote_code': False}
    tokenizer = AutoTokenizer.from_pretrained(spec['repo_id'], **options)
    ids, _, coverage = encode_input(tokenizer, row, mode, spec['max_length'])
    model = AutoModelForSequenceClassification.from_pretrained(
        spec['repo_id'], dtype=torch.bfloat16, attn_implementation='sdpa', **options)
    if model.config.label2id != {'contradiction': 0, 'entailment': 1, 'neutral': 2}:
        raise ValueError('Unexpected model label mapping.')
    tokenizer.pad_token_id = model.config.pad_token_id
    model.config.use_cache = model.config.text_config.use_cache = False
    model = model.cuda().eval()
    inputs = torch.tensor([ids], dtype=torch.long, device='cuda')
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs)).logits[0].float().double()
    if not torch.isfinite(logits).all():
        raise ValueError('Model returned nonfinite logits.')
    score = logits.softmax(-1)[1].item()
    return {'status': 'complete', 'repo_id': spec['repo_id'], 'revision': spec['revision'],
            'training_scope': spec['training_scope'], 'dataset': spec['dataset'],
            'dataset_revision': spec['dataset_revision'], 'protocol': spec['protocol'],
            'threshold_source': spec['threshold_source'],
            'input_mode': mode, 'score': score, 'score_kind': 'native entailment softmax',
            'threshold': 0.5, 'review': score >= 0.5, 'coverage': coverage}
