"""Optional fixed-version SkillSpector static scan, with network disabled in its worker."""
import argparse
import importlib.metadata
import multiprocessing
from pathlib import Path
import tempfile

from common import CONFIG, read_json, rows, save, write_scores
from scanner_worker import scanner_worker, unavailable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan','score'])
    parser.add_argument('--protocol', choices=['msd','sd'], required=True)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--output', type=Path, default=Path('outputs/static-rerun'))
    args = parser.parse_args()
    config = read_json(CONFIG/'skillspector.json')
    if args.command == 'plan':
        print(config)
        return
    if not args.data_dir:
        parser.error('--data-dir is required')
    if importlib.metadata.version('skillspector') != config['version']:
        raise ValueError('Install exactly the pinned SkillSpector version')
    records = rows(args.data_dir,args.protocol,'test')
    args.output.mkdir(parents=True,exist_ok=False)
    context = multiprocessing.get_context('spawn')
    results = []
    with tempfile.TemporaryDirectory(prefix='static-scan-') as temporary:
        skill = Path(temporary)/'SKILL.md'
        def start():
            parent,child = context.Pipe()
            process = context.Process(target=scanner_worker,args=(child,str(skill)),daemon=True)
            process.start(); child.close()
            if not parent.poll(60) or parent.recv() != {'ready':True}:
                process.terminate(); process.join(); parent.close()
                raise RuntimeError('Offline scanner did not start')
            return process,parent
        worker,connection = start()
        try:
            for row in records:
                skill.write_bytes(row['skill_text'].encode('utf-8'))
                connection.send(True)
                if connection.poll(90):
                    report = connection.recv()
                    if report.get('scanner_error'):
                        report = unavailable(report['scanner_error'])
                else:
                    worker.terminate(); worker.join(); connection.close()
                    report = unavailable({'type':'timeout','message':'Static scan exceeded 90 seconds.'})
                    worker,connection = start()
                risk = report['risk_assessment']
                results.append({'benchmark_id':row['benchmark_id'],'label':row['label'],'score':risk['score'],
                    'verdict':risk['severity'],'unavailable_reason':report.get('scanner_error',{}).get('type',''),
                    'model':'skillspector-2.12.0'})
        finally:
            if worker.is_alive():
                connection.send(False); worker.join(10)
            if worker.is_alive():
                worker.terminate(); worker.join()
            connection.close()
    write_scores(args.output/'scores.csv',results)
    save(args.output/'run.json',{'config':config,'protocol':args.protocol,'records':len(results),
        'timeout_seconds':90,'rule':{'threshold':50,'comparator':'>','unavailable_policy':'review'},
        'input_scope':'one neutral-named SKILL.md, full text, no companion files',
        'raw_findings_saved':False})


if __name__ == '__main__':
    main()
