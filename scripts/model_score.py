"""Local inference with versioned SkillScreen deployment models."""
import json
import math
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


def load_release(mode, model_dir=None):
    """Resolve one local release or the pinned Hub release for this input mode."""
    if model_dir is None and (ROOT / 'deployment_settings.json').is_file():
        model_dir = ROOT
    if model_dir is not None:
        source = str(Path(model_dir).resolve())
        settings = json.loads((Path(source) / 'deployment_settings.json').read_text(encoding='utf-8'))
        options = {'local_files_only': True, 'trust_remote_code': False}
        revision = settings['release']
    else:
        from huggingface_hub import hf_hub_download
        spec = json.loads((ROOT / 'configs/community-models.json').read_text(encoding='utf-8'))[mode]
        source, revision = spec['repo_id'], spec['revision']
        path = hf_hub_download(source, 'deployment_settings.json', revision=revision)
        settings = json.loads(Path(path).read_text(encoding='utf-8'))
        if settings['repo_id'] != source:
            raise ValueError('Model identity differs from the pinned release.')
        options = {'revision': revision, 'trust_remote_code': False}
    if settings['input_mode'] != {'package': 'package32k', 'markdown': 'markdown8k'}[mode]:
        raise ValueError('This model uses a different input mode.')
    if settings['hypothesis'] != HYPOTHESIS:
        raise ValueError('Unexpected hypothesis in deployment settings.')
    return source, options, settings, revision


def select_threshold(settings, operating_point='auto'):
    policy = settings['default_policy'] if operating_point == 'auto' else operating_point
    key = {'native': 'default', 'high-recall': 'high_recall', 'balanced': 'balanced'}.get(policy)
    if key is None:
        raise ValueError('Unknown operating point.')
    gate = settings['thresholds'][key]
    raw, native = gate['raw_threshold'], gate['native_threshold']
    if not all(math.isfinite(x) for x in (raw, native)) or not 0 < native < 1:
        raise ValueError('Invalid saved threshold.')
    if not math.isclose(raw, math.log(native) - math.log1p(-native), abs_tol=1e-9):
        raise ValueError('Raw and native thresholds disagree.')
    return policy, gate


def score_model(row, mode, model_dir=None, operating_point='auto'):
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ValueError('Model inference needs requirements/inference.txt. Package inspection and dependency checks need only Python.') from exc
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('The released inference recipe requires a CUDA GPU with BF16 support. Use --inspect or --checks dependencies on this machine.')
    source, options, settings, revision = load_release(mode, model_dir)
    policy, gate = select_threshold(settings, operating_point)
    torch.set_num_threads(8)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(source, **options)
    ids, _, coverage = encode_input(tokenizer, row, mode, settings['max_length'])
    model = AutoModelForSequenceClassification.from_pretrained(
        source, dtype=torch.bfloat16, attn_implementation='sdpa', **options)
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
    raw = (logits[1] - torch.logsumexp(logits[[0, 2]], dim=0)).item()
    return {'status': 'complete', 'repo_id': settings['repo_id'], 'revision': revision,
            'release': settings['release'], 'training_scope': settings['training_scope'],
            'dataset': settings['dataset']['repo_id'], 'dataset_revision': settings['dataset']['revision'],
            'protocol': 'three_fold_oof_then_all_data_refit', 'operating_point': policy,
            'threshold_source': 'native 0.5 reference' if policy == 'native' else 'frozen three-fold OOF threshold',
            'input_mode': mode, 'score': score, 'score_kind': 'native entailment softmax',
            'raw_log_odds': raw, 'threshold': gate['native_threshold'],
            'raw_threshold': gate['raw_threshold'], 'review': raw >= gate['raw_threshold'],
            'coverage': coverage}
