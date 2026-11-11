"""Inspect or screen a local skill package. Python 3.12+. Sample files are read only."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

from scripts.skill_package import read_package, write_json


def scan(path, checks='model', mode=None, inspect_only=False, model_dir=None, operating_point='auto'):
    row, coverage = read_package(path)
    bundled = Path(model_dir) if model_dir else Path(__file__).resolve().parent
    settings_file = bundled / 'deployment_settings.json'
    if mode is None and settings_file.is_file():
        saved_mode = json.loads(settings_file.read_text(encoding='utf-8'))['input_mode']
        mode = {'package32k': 'package', 'markdown8k': 'markdown'}[saved_mode]
    mode = mode or ('markdown' if coverage['input_kind'] == 'markdown_file' else 'package')
    report = {'created_at': datetime.now(timezone.utc).isoformat(),
              'package': coverage, 'model': {'status': 'not_run'},
              'dependencies': {'status': 'not_run'}}
    if inspect_only:
        return report
    if checks in ('dependencies', 'all'):
        from scripts.check_dependencies import check_dependencies
        report['dependencies'] = check_dependencies(row)
    if checks in ('model', 'all'):
        from scripts.model_score import score_model
        try:
            report['model'] = score_model(row, mode, model_dir=model_dir, operating_point=operating_point)
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            report['model'] = {'status': 'unavailable', 'reason': str(exc)}
    return report


def print_summary(report):
    coverage = report['package']
    print(f"Read {coverage['text_files']} text files; skipped {len(coverage['skipped'])} entries.")
    model = report['model']
    if model['status'] == 'complete':
        decision = 'review' if model['review'] else 'below the configured review threshold'
        print(f"Model: {model['score']:.6f} / threshold {model['threshold']:.6f} ({decision}).")
        print(f"Release: {model['repo_id']} @ {model['revision']}; operating point: {model['operating_point']}.")
        print(f"Model input: {model['input_mode']}; {model['coverage']['used_tokens']} tokens; truncated={model['coverage']['truncated']}.")
    else:
        print('Model: ' + model['status'].replace('_', ' ') + '.')
        if model.get('reason'):
            print(model['reason'])
    deps = report['dependencies']
    print('Dependencies: ' + deps['status'].replace('_', ' ') + '.')
    if deps['status'] != 'not_run':
        print(f"Checked {len(deps['checked'])} exact pins; {len(deps['findings'])} advisory matches; {len(deps['unresolved'])} unresolved entries; {len(deps['errors'])} lookup errors.")
        for finding in deps['findings']:
            print(f"  {finding['package']}=={finding['version']}: {finding['advisory_id']} at {finding['path']}:{finding['line']}")
        print('Only declared exact PyPI pins are checked. A match is not a finding of malicious intent.')
    if model['status'] == deps['status'] == 'not_run':
        print('Inspection only. No security checks were run.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path, help='Skill directory containing SKILL.md, or a single SKILL.md')
    parser.add_argument('--inspect', action='store_true', help='Offline inventory only; no models or network')
    parser.add_argument('--checks', choices=('model', 'dependencies', 'all'), default='model')
    parser.add_argument('--mode', choices=('package', 'markdown'), help='Defaults to package for a directory, markdown for a file')
    parser.add_argument('--model-dir', type=Path, help='Use a downloaded deployment release instead of the Hub')
    parser.add_argument('--operating-point', choices=('auto', 'native', 'high-recall', 'balanced'), default='auto',
                        help='Saved operating point; auto uses the release default, native uses 0.5')
    parser.add_argument('--output', type=Path, help='Create a JSON report without raw file contents')
    args = parser.parse_args(argv)
    if args.output and (args.output.exists() or args.output.resolve().is_relative_to(args.path.resolve() if args.path.is_dir() else args.path.parent.resolve())):
        parser.error('Choose a new output file outside the input package.')
    report = scan(args.path, args.checks, args.mode, args.inspect, args.model_dir, args.operating_point)
    if args.output:
        write_json(args.output, report)
    print_summary(report)
    return 1 if any(report[key]['status'] == 'unavailable' or report[key].get('errors') for key in ('model', 'dependencies')) else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
