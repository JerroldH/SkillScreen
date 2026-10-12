"""Optional Word-SVM fit/evaluation using the original fixed estimator recipe."""
import argparse
from pathlib import Path

from common import CONFIG, digest, read_json, rows, save, write_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan','fit','evaluate'])
    parser.add_argument('--protocol', choices=['msd','sd'], required=True)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--output', type=Path, default=Path('outputs/word-svm-rerun'))
    args = parser.parse_args()
    config = read_json(CONFIG/'word-svm.json')
    if args.command == 'plan':
        print(config)
        return
    if not args.data_dir:
        parser.error('--data-dir is required')
    import joblib
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC
    if args.command == 'fit':
        args.output.mkdir(parents=True, exist_ok=False)
        train = rows(args.data_dir,args.protocol,'train')
        validation = rows(args.data_dir,args.protocol,'validation')
        tfidf = dict(config['tfidf'])
        tfidf['dtype'] = np.float32
        tfidf['ngram_range'] = tuple(tfidf['ngram_range'])
        estimator = make_pipeline(TfidfVectorizer(**tfidf), LinearSVC(**config['svm']))
        estimator.fit([r['skill_text'] for r in train],[r['label'] for r in train])
        scores = estimator.decision_function([r['skill_text'] for r in validation])
        if not np.isfinite(scores).all():
            raise ValueError('Nonfinite validation scores')
        threshold = float(np.sort(scores[np.array([r['label'] for r in validation])==1])[::-1]
                          [int(np.ceil(.99*sum(r['label'] for r in validation)))-1])
        joblib.dump(estimator,args.output/'model.joblib',compress=3)
        reloaded = joblib.load(args.output/'model.joblib')
        replay = reloaded.decision_function([r['skill_text'] for r in validation])
        if not np.allclose(scores,replay,atol=1e-12,rtol=1e-12):
            raise ValueError('Saved SVM failed validation reload')
        write_scores(args.output/'validation_scores.csv',
                     ({'benchmark_id':r['benchmark_id'],'label':r['label'],'score':float(s)} for r,s in zip(validation,scores)))
        save(args.output/'frozen.json',{'protocol':args.protocol,'config':config,
            'model_sha256':digest(args.output/'model.joblib'),
            'validation_frozen_threshold':threshold,'target_validation_recall':.99,'comparator':'>=',
            'table_threshold':0., 'table_comparator':'>=', 'test_used':False})
        return
    frozen = read_json(args.output/'frozen.json')
    if frozen['protocol'] != args.protocol or frozen['config'] != config:
        raise ValueError('Saved estimator protocol/config mismatch')
    if digest(args.output/'model.joblib') != frozen['model_sha256']:
        raise ValueError('Saved estimator hash changed')
    test = rows(args.data_dir,args.protocol,'test')
    estimator = joblib.load(args.output/'model.joblib')
    scores = estimator.decision_function([r['skill_text'] for r in test])
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite test scores')
    write_scores(args.output/'scores.csv',
                 ({'benchmark_id':r['benchmark_id'],'label':r['label'],'score':float(s)} for r,s in zip(test,scores)))
    print('Saved test scores. Native scores are margins, not probabilities.')


if __name__ == '__main__':
    main()
