"""Render released aggregate metrics; contains no model or correction algorithm."""
import argparse
import csv
import json
from pathlib import Path


def render(config_path):
    config_path = Path(config_path).resolve()
    cfg = json.loads(config_path.read_text())
    path = (config_path.parent / cfg['results_file']).resolve()
    with path.open(newline='') as f:
        rows = [r for r in csv.DictReader(f)
                if r['dataset'] == cfg['dataset'] and r['model'] == cfg['model']]
    by_arm = {r['arm']: r for r in rows}
    if len(by_arm) != len(rows) or set(by_arm) != set(cfg['arms']):
        raise ValueError('Expected one row per configured arm')
    cols = cfg['columns']; digits = int(cfg['percentage_decimals'])
    lines = [f"# {cfg['dataset']} / {cfg['model']}", '',
             '| Arm | ' + ' | '.join(cols) + ' |',
             '|---|' + '---:|' * len(cols)]
    for arm in cfg['arms']:
        r = by_arm[arm]; values = []
        for k in cols:
            v = r[k]
            values.append('—' if not v else (f'{int(v):,}' if k == 'G' else f'{float(v):.{digits}f}'))
        lines.append('| ' + arm + ' | ' + ' | '.join(values) + ' |')
    lines += ['', 'G is a point count; NIR, IoU and mAP are percentages.',
              'These are aggregate candidate results, not a runnable method release.']
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', help='Optional Markdown output file')
    args = p.parse_args()
    text = render(args.config)
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text, end='')
