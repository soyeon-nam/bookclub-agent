"""Audit requirement/measurement links without claiming product quality."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

SPEC_PATH = Path(__file__).with_name('specification.json')


def load_spec(path=SPEC_PATH):
    spec = json.loads(Path(path).read_text(encoding='utf-8'))
    validate_spec(spec)
    return spec


def validate_spec(spec):
    if spec.get('schema_version') != 1:
        raise ValueError('Unsupported measurement contract version')
    ids = {}
    for group in ('requirements', 'failure_modes', 'metrics', 'constraints', 'non_adopted'):
        rows = spec.get(group)
        if not isinstance(rows, list):
            raise ValueError('Missing contract group: ' + group)
        names = [row['id'] for row in rows]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate identifiers in ' + group)
        ids[group] = set(names)
    if ids['requirements'] & ids['non_adopted']:
        raise ValueError('Inactive requirement cannot be scored')
    for row in spec['requirements'] + spec['metrics']:
        if set(row.get('failure_modes', [])) - ids['failure_modes']:
            raise ValueError('Unknown failure mode: ' + row['id'])
    for metric in spec['metrics']:
        if set(metric['requirements']) - ids['requirements']:
            raise ValueError('Unknown or inactive requirement: ' + metric['id'])
        if metric['status'] not in ('proxy', 'descriptive', 'unvalidated', 'unimplemented'):
            raise ValueError('Unsubstantiated measurement status: ' + metric['id'])
        if not metric.get('limitation'):
            raise ValueError('Measurement limitation is required')
        if metric['kind'] == 'planned' and (metric['status'] != 'unimplemented' or metric['availability'] != 'not_available'):
            raise ValueError('Planned measure cannot be advertised as available')
    if any(not any(row['id'] in m['requirements'] for m in spec['metrics']) for row in spec['requirements']):
        raise ValueError('Requirement missing a measured or planned instrument')
    return spec


def build_coverage(spec=None):
    spec = load_spec() if spec is None else validate_spec(spec)
    rows = []
    for requirement in spec['requirements']:
        linked = [m for m in spec['metrics'] if requirement['id'] in m['requirements']]
        rows.append({'requirement': requirement['id'], 'failure_modes': requirement['failure_modes'],
                     'available': [m['id'] for m in linked if m['status'] != 'unimplemented'],
                     'planned': [m['id'] for m in linked if m['status'] == 'unimplemented'],
                     'quality_verdict': None})
    return {'schema_version': 1, 'requirements': rows,
            'without_current_instrument': [row['requirement'] for row in rows if not row['available']],
            'metric_status': {m['id']: m['status'] for m in spec['metrics']},
            'note': 'Instrument availability is not validity, requirement satisfaction, or product quality.'}


def automatic_contracts(keys):
    spec = load_spec()
    records = {m['id'].split('.', 1)[1]: m for m in spec['metrics'] if m['id'].startswith('automatic.')}
    if set(records) != set(keys):
        raise ValueError('Computed metrics and measurement contracts differ')
    return {'specification_sha256': hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest(),
            'population': deepcopy(spec['population']),
            'metrics': {key: deepcopy(records[key]) for key in sorted(keys)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = build_coverage()
    result['specification_sha256'] = hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest()
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        if args.output.resolve() == SPEC_PATH.resolve():
            parser.error('Output must not overwrite the measurement contract')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
