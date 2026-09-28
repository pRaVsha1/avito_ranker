"""Фиксация второй итерации после выбора по dev, отчёт и финальный CSV."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import shutil
import numpy as np
import pandas as pd
from solution import validate, read_pickle, dump_json, blend_scores, top_indices
from advanced import fusion_score
from ensemble import Components, select


def checksum(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def errors(out, sources, engine, loss):
    saved = read_pickle(out / f'advanced_{sources}_dev.pkl')
    names = saved['feature_names']
    corpus = pd.read_parquet(out / 'corpus.parquet')
    queries = pd.read_parquet(out / 'queries_dev.parquet').set_index('query_id')
    if engine == 'ensemble':
        config = json.loads((out / f'ensemble_{sources}.json').read_text())
        component = Components(out, sources, names)
        choose = lambda x: select(component.score(x), config['selection'])
    else:
        from catboost import CatBoost

        config = json.loads((out / f'advanced_{sources}_{loss}.json').read_text())
        model = CatBoost().load_model(str(out / f'advanced_{sources}_{loss}.cbm'))
        choose = lambda x: top_indices(
            blend_scores(
                model.predict(
                    x, prediction_type='RawFormulaVal', ntree_end=config['trees']
                ),
                fusion_score(x, names, **config['fusion']),
                config['blend_weight'],
            ),
            50,
        )
    result = []
    per_query = []
    by_id = corpus.set_index('item_id')
    for row in saved['records']:
        q = queries.loc[row['query_id']]
        selected = corpus.iloc[row['indices'][choose(row['x'])]]
        candidates = set(corpus.iloc[row['indices']].item_id)
        relevant = set(q.relevant)
        picked = set(selected.item_id)
        per_query.append(
            dict(
                query_id=row['query_id'],
                recall50=len(picked & relevant) / len(relevant),
                candidate_recall=len(candidates & relevant) / len(relevant),
            )
        )
        for item in sorted(relevant - picked):
            truth = by_id.loc[item]
            result.append(
                dict(
                    query=q.search_query,
                    filters=q.search_infm_params_text,
                    error_type='selection' if item in candidates else 'retrieval',
                    relevant_title=truth.item_title_raw,
                    same_location=truth.item_location_id == q.search_location_id,
                    predicted_titles=' | '.join(selected.item_title_raw.head(3)),
                )
            )
    pd.DataFrame(result).to_csv(out / 'dev_errors.csv', index=False)
    pd.DataFrame(per_query).to_csv(out / 'dev_per_query_v2.csv', index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine', choices=['advanced', 'ensemble'], default='ensemble')
    p.add_argument('--sources', choices=['title', 'base', 'domain'], required=True)
    p.add_argument('--loss', choices=['pair', 'logloss'], default='logloss')
    p.add_argument(
        '--existing-answer', help='Уже проверенный CSV для той же конфигурации.'
    )
    p.add_argument(
        '--existing-config', help='Конфигурация, которой получен existing-answer.'
    )
    p.add_argument('--data-dir', default='data')
    p.add_argument('--artifacts-dir', default='artifacts')
    p.add_argument('--output', default='answer.csv')
    a = p.parse_args()
    out = Path(a.artifacts_dir)
    config_path = out / (
        f'ensemble_{a.sources}.json'
        if a.engine == 'ensemble'
        else f'advanced_{a.sources}_{a.loss}.json'
    )
    config = json.loads(config_path.read_text())
    dev_score = (
        config['selection']['recall50']
        if a.engine == 'ensemble'
        else config['recall50']
    )
    command = [
        sys.executable,
        a.engine + '.py',
        'predict',
        '--sources',
        a.sources,
        '--artifacts-dir',
        str(out),
    ]
    if a.engine == 'advanced':
        command += ['--loss', a.loss]
    # Тот же test, что в первой итерации, оценивается только после нового выбора
    # по dev. Это повторная контрольная оценка, не новая слепая выборка.
    subprocess.run(
        command + ['--split', 'test', '--output', str(out / 'test_v2.json')], check=True
    )
    started = time.monotonic()
    if a.existing_answer:
        if not a.existing_config:
            raise ValueError('Требуется конфигурация ранее проверенного CSV.')
        previous = json.loads(Path(a.existing_config).read_text())
        for key in ['feature_names', 'selection']:
            assert previous[key] == config[key], f'Конфигурация отличается: {key}'
        shutil.copyfile(a.existing_answer, a.output)
        elapsed = None
    else:
        subprocess.run(
            command + ['--split', 'benchmark', '--full', '--output', a.output],
            check=True,
        )
        elapsed = time.monotonic() - started
    checked = validate(a.data_dir, a.output)
    test = json.loads((out / 'test_v2.json').read_text())
    errors(out, a.sources, a.engine, a.loss)
    source_names = {
        'title': 'E5-small + заголовки',
        'base': 'E5-small + E5-base + география',
        'domain': 'E5-small + E5-base + доменная модель + география',
    }
    models = {'e5-small': '614241f622f53c4eeff9890bdc4f31cfecc418b3'}
    if a.sources in ['base', 'domain']:
        models['e5-base'] = 'd128750597153bb5987e10b1c3493a34e5a4502a'
    if a.sources == 'domain':
        models['e5-domain'] = 'local-finetune'
    report = dict(
        selected_name=source_names[a.sources]
        + (' + ансамбль' if a.engine == 'ensemble' else ' + ' + a.loss),
        engine=a.engine,
        sources=a.sources,
        loss=a.loss,
        kind='v2',
        dense=True,
        metadata=True,
        dev_recall50=dev_score,
        dev_candidate_recall=config['candidate_recall'],
        test_recall50=test['recall50'],
        test_candidate_recall=test['candidate_recall'],
        answer_sha256=checked['sha256'],
        answer_rows=checked['rows'],
        cpu_prediction_seconds_server=(
            round(elapsed, 2) if elapsed is not None else None
        ),
        reused_verified_prediction=bool(a.existing_answer),
        seed=42,
        config=config,
        code_sha256=checksum('solution.py'),
        source_code_sha256={
            f: checksum(f)
            for f in [
                'solution.py',
                'advanced.py',
                'ensemble.py',
                'category_prior.py',
                'geo_prior.py',
            ]
        },
        data_sha256={
            p.name: checksum(p) for p in sorted(Path(a.data_dir).glob('*.parquet'))
        },
        models={
            name: dict(
                revision=revision,
                weights_sha256=checksum(out / name / 'model.safetensors'),
            )
            for name, revision in models.items()
        },
        output_order='Selected IDs sorted lexicographically.',
        evaluation_note='Dev selects all variants. Test is reused only for reporting after freezing each iteration, not for tuning. Observed positives; supplemented corpus.',
    )
    journal_path = Path('submissions.json')
    if journal_path.exists():
        attempts = json.loads(journal_path.read_text())['attempts']
        exact = [r for r in attempts if r['sha256'] == checked['sha256']]
        report['platform_recall50_by_user'] = exact[-1]['recall50'] if exact else None
        report['best_confirmed_platform_recall50'] = max(
            r['recall50'] for r in attempts
        )
    dump_json(out / 'final_report.json', report)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ['config', 'source_code_sha256', 'data_sha256', 'models']
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
