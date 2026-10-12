"""Strict prepared-data checks shared by optional rerun commands."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/baselines'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(data_dir, protocol, partition):
    reference = read_json(ROOT / 'data/splits' / protocol / 'summary.json')
    path = Path(data_dir) / reference['files'][partition]['path']
    if digest(path) != reference['files'][partition]['sha256']:
        raise ValueError('Prepared ' + protocol + '/' + partition + ' checksum differs from frozen experiment')
    with path.open(encoding='utf-8') as handle:
        records = [json.loads(line) for line in handle]
    if len(records) != reference['prepared_counts'][partition]['total']:
        raise ValueError('Prepared record count differs')
    if len({r['benchmark_id'] for r in records}) != len(records):
        raise ValueError('Duplicate prepared IDs')
    return records


def tokenizer(path):
    from transformers import AutoTokenizer
    from input_and_typesafe import tokenizer_length_fingerprint
    cfg = read_json(CONFIG / 'input.json')
    tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    if digest(Path(path) / 'tokenizer.json') != cfg['tokenizer_files_sha256']['tokenizer.json']:
        raise ValueError('Tokenizer vocabulary/encoding bytes differ from the historical tokenizer')
    expected = next(iter(read_json(CONFIG / 'large-input-profile.json').values()))['tokenizer_sha256']
    if tokenizer_length_fingerprint(tok) != expected or tok.num_special_tokens_to_add(False) != 0:
        raise ValueError('Tokenizer encoding behavior differs from the historical tokenizer')
    tok.padding_side = 'right'
    return tok


def write_scores(path, records):
    fields = ['benchmark_id','label','score','verdict','unavailable_reason','model','input_tokens','output_tokens','reasoning_tokens']
    with Path(path).open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in records:
            writer.writerow(row)
            handle.flush()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
