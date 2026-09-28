"""Одна команда для точного офлайн-воспроизведения выбранного решения на CPU."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from solution import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--artifacts-dir', default='artifacts')
    parser.add_argument('--output', default='answer_reproduced.csv')
    args = parser.parse_args()
    report = json.loads((Path(args.artifacts_dir) / 'final_report.json').read_text())
    env = os.environ.copy()
    env.update(
        HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',
        CUDA_VISIBLE_DEVICES='',
        OPENBLAS_NUM_THREADS='6',
        OMP_NUM_THREADS='6',
    )
    engine = report.get('engine', 'solution')
    command = [
        sys.executable,
        str(Path(__file__).with_name(engine + '.py')),
        'predict',
        '--full',
        '--data-dir',
        args.data_dir,
        '--artifacts-dir',
        args.artifacts_dir,
        '--output',
        args.output,
    ]
    if engine == 'solution':
        if report['dense']:
            command.append('--dense')
        if report.get('metadata'):
            command.append('--metadata')
    else:
        command += ['--sources', report['sources'], '--split', 'benchmark']
        if engine == 'advanced':
            command += ['--loss', report['loss']]
    subprocess.run(command, env=env, check=True)
    result = validate(args.data_dir, args.output)
    if result['sha256'] != report['answer_sha256']:
        raise RuntimeError(
            'Формат корректен, но контрольная сумма отличается от зафиксированного ответа.'
        )
    print('Результат точно воспроизведён:', args.output)


if __name__ == '__main__':
    main()
