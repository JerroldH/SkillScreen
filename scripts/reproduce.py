"""Recompute Table 1 from the supplied predictions, offline, using Python only."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('recall', 'fpr', 'auroc', 'fpr99')


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def checked_file(record):
    path = ROOT / record['path']
    if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
        raise ValueError(f'File checksum differs: {record["path"]}')
    return path


def scores(rows, field):
    result = []
    for row in rows:
        value = float(row[field]) if row[field] != '' else None
        if value is not None and not math.isfinite(value):
            raise ValueError(f'Nonfinite {field} for {row["benchmark_id"]}')
        result.append(value)
    return result


def decisions(values, rule):
    out = []
    for value in values:
        if value is None:
            if rule['unavailable_policy'] != 'review':
                raise ValueError('Missing prediction under error policy')
            out.append(True)
        else:
            out.append(value > rule['threshold'] if rule['comparator'] == '>' else value >= rule['threshold'])
    return out


def confusion(labels, flags):
    return {key: sum(y == label and flag == decision for y, flag in zip(labels, flags))
            for key, label, decision in [('tp',1,True), ('fp',0,True), ('tn',0,False), ('fn',1,False)]}


def auroc(labels, values):
    """Equivalent to sklearn ROC AUC: ties receive their mean rank."""
    pairs = sorted((v, y) for v, y in zip(values, labels) if v is not None)
    positive = sum(y for _, y in pairs)
    negative = len(pairs) - positive
    if not positive or not negative:
        raise ValueError('AUROC requires both classes among finite scores')
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        rank_sum += ((i + 1 + j) / 2) * sum(y for _, y in pairs[i:j])
        i = j
    return (rank_sum - positive * (positive + 1) / 2) / (positive * negative)


def fpr99(labels, values):
    """Original review_metrics.matched_fpr rule, without numpy or bootstrap."""
    v = [math.inf if x is None else x for x in values]
    positive = sorted((x for x, y in zip(v, labels) if y == 1), reverse=True)
    negative = [x for x, y in zip(v, labels) if y == 0]
    k = max(1, min(math.ceil(0.99 * len(positive) - 1e-12), len(positive)))
    threshold = positive[k - 1]
    return {'fpr': sum(x >= threshold for x in negative) / len(negative),
            'recall': sum(x >= threshold for x in positive) / len(positive),
            'threshold': threshold if math.isfinite(threshold) else None}


def evaluate(run, rule, cohort):
    rows = read_csv(checked_file(run['prediction']))
    by_id = {r['benchmark_id']: r for r in rows}
    if len(rows) != len(by_id) or set(by_id) != set(cohort):
        raise ValueError(f'Duplicate, missing or extra IDs: {run["id"]}')
    rows = [by_id[key] for key in sorted(cohort)]
    labels = [int(r['label']) for r in rows]
    if labels != [cohort[key] for key in sorted(cohort)]:
        raise ValueError(f'Labels differ from the test cohort: {run["id"]}')
    values = scores(rows, rule['field'])
    cm = confusion(labels, decisions(values, rule))
    matched = fpr99(labels, values)
    result = {'id':run['id'], 'seed':run['seed'], 'records':len(rows), 'unavailable_scores':values.count(None),
              'confusion':cm, 'recall':cm['tp']/(cm['tp']+cm['fn']), 'fpr':cm['fp']/(cm['fp']+cm['tn']),
              'auroc':auroc(labels, values), 'fpr99':matched['fpr'],
              'test_fpr99_recall':matched['recall'], 'test_fpr99_threshold':matched['threshold']}
    gate = run.get('archived_operating_point')
    if gate and gate.get('rule'):
        observed = confusion(labels, decisions(scores(rows, gate['rule']['field']), gate['rule']))
        if gate.get('expected_confusion') and observed != gate['expected_confusion']:
            raise ValueError(f'Archived operating point does not replay: {run["id"]}')
        result['archived_operating_point'] = {'kind':gate['kind'], 'confusion':observed, 'included_in_table1':False}
    return result


def summarize(members):
    return {metric: {'mean':statistics.mean(r[metric] for r in members),
                      'std':statistics.stdev(r[metric] for r in members) if len(members)>1 else None}
            for metric in METRICS}


def display(metric, summary):
    value = summary['mean'] * (1 if metric == 'auroc' else 100)
    text = f'{value:.3f}' if metric == 'auroc' else f'{value:.1f}'
    if metric in ('recall', 'fpr') and summary['std'] is not None:
        text += f' +/- {summary["std"] * 100:.1f}'
    return text


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def write_tex(path, table, methods, summaries):
    lines = [r'% Requires booktabs. Rates are percentages.',
             r'\begin{table*}[t]', r'\centering\small', r'\setlength{\tabcolsep}{4pt}',
             r'\begin{tabular}{lrrrrrrrr}', r'\toprule',
             r'& \multicolumn{4}{c}{MSD} & \multicolumn{4}{c}{SD} \\',
             r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}',
             r'Method & Recall & FPR & AUROC & FPR$_{99}$ & Recall & FPR & AUROC & FPR$_{99}$ \\', r'\midrule']
    group = None
    for row, method in zip(table, methods):
        if group != method['group']:
            if group is not None:
                lines.append(r'\midrule')
            group = method['group']
            lines.append(r'\multicolumn{9}{l}{\emph{' + group + r'}} \\')
        cells = [row['method']]
        for split in ('msd', 'sd'):
            for metric in METRICS:
                cell = row[f'{split}_{metric}']
                if ' +/- ' in cell:
                    mean, std = cell.split(' +/- ')
                    cell = f'${mean}_{{\\pm {std}}}$'
                if metric in ('auroc','fpr99'):
                    values = sorted({v[split][metric]['mean'] for v in summaries.values()}, reverse=metric=='auroc')
                    rank = values.index(summaries[method['id']][split][metric]['mean'])
                    if rank < 2:
                        cell = ('\\textbf{' if rank == 0 else '\\underline{') + cell + '}'
                cells.append(cell)
        lines.append(' & '.join(cells) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}',
                  r'\caption{Table 1 recomputed from the supplied predictions. FPR$_{99}$ uses test-matched 99\% recall. Subscripts give standard deviations across training seeds.}', r'\end{table*}'])
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metric', choices=('all','fpr99'), default='all', help='Limit the console display to test FPR@99')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/table1')
    args = parser.parse_args()
    output = args.output.resolve()
    # Avoid accidentally overwriting inputs with --output.
    if output == ROOT or any(output == ROOT/p or (ROOT/p) in output.parents for p in ('data','configs','expected','scripts','requirements')):
        parser.error('Use a separate output directory, such as outputs/table1')
    manifest = json.loads((ROOT/'configs/runs.json').read_text(encoding='utf-8'))
    cohorts = {}
    for split, info in manifest['protocols'].items():
        rows = read_csv(checked_file(info['test_cohort']))
        cohort = {r['benchmark_id']:int(r['label']) for r in rows}
        if (len(cohort) != len(rows) or len(rows) != info['records']
                or sum(cohort.values()) != info['malicious'] or any(y not in (0,1) for y in cohort.values())):
            raise ValueError(f'Invalid test cohort: {split}')
        cohorts[split] = cohort
    runs, summaries, table = [], {}, []
    for method in manifest['methods']:
        summary = {}
        for split in ('msd','sd'):
            members = [evaluate(run, method['fixed_rule'], cohorts[split]) for run in method['protocols'][split]]
            runs.extend(members)
            summary[split] = summarize(members)
        summaries[method['id']] = summary
        table.append({'method':method['name'], **{f'{split}_{metric}':display(metric, summary[split][metric])
                                                  for split in ('msd','sd') for metric in METRICS}})
    expected = read_csv(ROOT/'expected/table1.csv')
    differences = []
    if len(expected) != len(table):
        differences.append({'error':'Reference row count differs'})
    for actual, reference in zip(table, expected):
        for key, value in actual.items():
            if reference.get(key) != value:
                differences.append({'method':actual['method'], 'column':key, 'actual':value, 'expected':reference.get(key)})
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'table1.csv', table)
    write_tex(output/'table1.tex', table, manifest['methods'], summaries)
    report = {'passed':not differences, 'prediction_files':len(runs), 'test_records':{k:len(v) for k,v in cohorts.items()},
              'reference_display_matches':not differences, 'differences':differences,
              'rules':manifest['metrics'], 'summaries':summaries, 'runs':runs}
    (output/'verification.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n', encoding='utf-8')
    for row in table:
        values = [row['method'], row['msd_fpr99'], row['sd_fpr99']] if args.metric=='fpr99' else list(row.values())
        print(' | '.join(values))
    print(f'\n{"PASS" if not differences else "FAIL"}: {len(runs)} runs, {len(table)*8} displayed cells. Outputs: {output}')
    if differences:
        print(json.dumps(differences, indent=2))
        return 1
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
