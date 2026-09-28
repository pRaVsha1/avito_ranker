"""Исполняемый notebook второй итерации; значения берутся из сохранённых отчётов."""

from pathlib import Path
import json
import nbformat as nbf


def main():
    root = Path(__file__).resolve().parent
    report = json.loads((root / 'artifacts/final_report.json').read_text())
    cells = []

    def md(s):
        cells.append(nbf.v4.new_markdown_cell(s))

    def code(s):
        cells.append(nbf.v4.new_code_cell(s))

    md(
        f"""# Поиск объявлений Авито: воспроизводимое решение

**Recall@50 платформы: {report["platform_recall50_by_user"]:.6f} · 2 452 запроса · локальное предсказание без API**

Выбран вариант **{report['selected_name']}**. Dev Recall@50 — **{report['dev_recall50']:.4f}**, контрольная test — **{report['test_recall50']:.4f}**. Оценки платформы приведены отдельно в конце.

Этот notebook заново рассчитывает `answer_reproduced.csv` на CPU из сохранённых индексов, эмбеддингов и моделей, а затем проверяет точное совпадение SHA-256. Готовые ответы не подставляются по `query_id`. Исходный код всех алгоритмов приложен рядом с notebook.

Локальная метрика использует наблюдавшиеся выборы пользователей, а не полную разметку релевантности."""
    )
    md(
        """## Метод и границы проверки

1. Разбиение по нормализованным текстам запроса: 80/10/10, seed=42. Все варианты одного текста остаются в одной части.
2. Для ранжировщика равномерно выбраны 8 000 текстов, для dev и test — по 1 200. На текст взят один случайный контекст локации и фильтров.
3. Локальный корпус — все объявления benchmark плюс отсутствующие позитивы экспериментальных запросов. На dev/test позитивы не вставляются в кандидаты поиска.
4. История, прогноз подкатегории и географические статистики обучены только на train-части, дополнительно исключая тексты обучения ранжировщика. Это предотвращает использование собственной метки как признака.
5. Все варианты и параметры выбираются по dev. Test повторно оценивается только после фиксации второй итерации. Это тот же контрольный набор, что в первой итерации, а не новая слепая проверка. Его метки не используются для настройки.

Отсутствие взаимодействия не доказывает нерелевантность. Конкуренты — слабые отрицательные примеры; известные позитивы из обучающего запроса исключены из них."""
    )
    md(
        """## Окружение

Перед первым запуском установите зависимости из `requirements-inference.txt` и выполните `python download_artifacts.py` из корня репозитория. Он загрузит проверенный комплект из GitHub Release и проверит SHA-256. Далее notebook работает без сети. Для стандартного запуска достаточно Python 3.12. GPU и PyTorch не нужны: представления текстов уже рассчитаны. Для полной пересборки нужны `requirements.txt`, локальные веса E5 и две GPU; проверены GTX 1080 Ti и PyTorch 2.7.1 + CUDA 11.8.

Данные: [архив задания](https://disk.yandex.ru/d/sNhfo0YOjGtufg). SHA-256 исходного ZIP: `8dd3cba59201bae333c11db89c5111198fa10cc52a70c70bdc57bd6a248fb777`. Подготовительная загрузка отделена от офлайн-обучения и предсказания."""
    )
    code("""import os
os.environ.update(OMP_NUM_THREADS='6', OPENBLAS_NUM_THREADS='6',
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='')
from pathlib import Path
import hashlib, json, subprocess, sys, time
import numpy as np
import pandas as pd
from IPython.display import display
ROOT=Path.cwd()
DATA_DIR=ROOT/'data'
ARTIFACTS_DIR=ROOT/'artifacts'
OUTPUT=ROOT/'answer_reproduced.csv'
report=json.loads((ARTIFACTS_DIR/'final_report.json').read_text())
audit=json.loads((ARTIFACTS_DIR/'data_audit.json').read_text())
print('Выбранный вариант:', report['selected_name'])""")
    md(
        """## Данные и качество

Идентификаторы остаются строками; регистр и ведущие нули сохраняются. Нормализация Unicode NFKC, регистра, `ё` и пробелов используется для разбиения и лексических признаков. Нейросети получают исходный текст с собственным токенизатором. Пропуски числовых признаков сохраняются и обрабатываются CatBoost."""
    )
    code("""queries=pd.read_parquet(DATA_DIR/'benchmark_queries.parquet')
items=pd.read_parquet(DATA_DIR/'benchmark_items.parquet',columns=['item_id'])
assert len(queries)==2452 and queries.query_id.is_unique
assert len(items)==189212 and items.item_id.is_unique
assert queries.query_id.str.len().eq(16).all()
assert items.item_id.str.fullmatch('[0-9a-f]{16}').all()
quality=[]
for name in ['train','benchmark_queries','benchmark_items']:
    r=audit[name]
    quality.append({'Таблица':name,'Строк':r['rows'],'Полных дубликатов':r['duplicate_rows'],
                    'Полей с пропусками':sum(v>0 for v in r['missing'].values())})
display(pd.DataFrame(quality))
for name,expected in report['data_sha256'].items():
    with (DATA_DIR/name).open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==expected
print('Контрольные суммы трёх Parquet совпали.')""")
    md(
        """### Отсутствие пересечений между частями

Проверяются нормализованные тексты, а не внутренние идентификаторы. Один и тот же текст не должен попадать одновременно в обучение ранжировщика и отложенную оценку."""
    )
    code(
        """from solution import normalize
split=pd.read_parquet(ARTIFACTS_DIR/'split.parquet')
groups={name:pd.read_parquet(ARTIFACTS_DIR/f'queries_{name}.parquet') for name in ['train','dev','test']}
sets={name:set(frame.search_query.map(normalize)) for name,frame in groups.items()}
assert split.normalized_query.is_unique
assert not (sets['train']&sets['dev'] or sets['train']&sets['test'] or sets['dev']&sets['test'])
assert all(len(sets[k])==len(groups[k]) for k in groups)
history=pd.read_parquet(ARTIFACTS_DIR/'history_train.parquet',columns=['search_query'])
history_texts=set(history.search_query.map(normalize))
assert not history_texts&(sets['train']|sets['dev']|sets['test'])
display(pd.DataFrame({'Полное разбиение':split['split'].value_counts(),
                      'Экспериментальные запросы':{k:len(v) for k,v in groups.items()}}))"""
    )
    md(
        """## Поиск кандидатов

- **BM25:** заголовок с повышенным весом, параметры и начало описания; Snowball stemmer, `k1=1.2`, `b=0.75`. Редкие термины сохраняются (`min_df=1`). Отдельно индексируются заголовки и исторические запросы.
- **Символьный TF-IDF:** n-граммы 3–5 по заголовкам для опечаток и словоформ.
- **E5-small и E5-base:** 384 и 768 измерений. Префиксы `query:`/`passage:`, masked mean pooling, L2-нормализация, до 256 токенов. Используются запрос с фильтрами, текст запроса без фильтров и короткое представление объявления по заголовку и параметрам.
- **География:** исходные координаты и статистики разрешённой истории. Для части поисковых локаций центр оценивается по обучающим взаимодействиям. Дополнительный источник мягко уменьшает семантическую оценку с расстоянием. Глобальные кандидаты сохраняются, жёстких фильтров нет.

Каждый источник возвращает до 200 глобальных и до 200 локальных кандидатов. После объединения удаляются повторные `item_id`. Семантический поиск использует точное скалярное произведение, без приближённого индекса."""
    )
    md(
        """## Отбор 50 объявлений

Базовый RRF суммирует `60/(60+rank)` с весами, выбранными на dev. В первой итерации CatBoost PairLogit обучался с 50 конкурентами на запрос, во второй — с 250: 80% сложных по поисковой оценке и 20% случайных из найденных. Позитивы добавляются только при обучении.

Сравниваются PairLogit и Logloss, их смешивание с RRF и подтверждённой моделью первой итерации. Проверены также квоты рекомендаций разных моделей. Конкретный выбранный способ записан в конфигурации — правила едины для всех запросов.

Признаки: оценки и ранги источников, пересечения слов и фильтров, совпадение категории и локации, расстояние, цена, рейтинг, отзывы, популярность и контактные ограничения. MultinomialNB добавляет мягкий прогноз подкатегории. Географические признаки учитывают частоты переходов между локациями и различия между локальными и удалёнными услугами.

`query_id` не используется как признак. Равные оценки разрешаются по `item_id`. После выбора 50 ID сортируются перед записью: метрика зависит от множества, а канонический порядок стабилизирует SHA-256 между платформами."""
    )
    code("""if report.get('engine')=='ensemble':
    print('Выбранная стратегия:', report['config']['selection'])
else:
    config=report['config']
    print({k:config[k] for k in ['trees','blend_weight'] if k in config})
from advanced import fusion_score
import inspect
print(inspect.getsource(fusion_score))""")
    md(
        """### Доменное дообучение как отдельный эксперимент

Если доменный вариант присутствует в сравнении ниже, он получен из E5-small на разрешённой истории: тексты dev, test и обучения ранжировщика исключены. Используется симметричная контрастивная функция с несколькими позитивами; известные положительные пары не становятся внутрибатчевыми негативами. Словарные embeddings фиксируются, обучаются контекстные слои. Параметры и фактическое число шагов сохранены в `e5-domain/training_config.json`.

Сам факт проведения эксперимента не означает, что он выбран. Решение определяется dev-метрикой и конфигурацией в начале notebook."""
    )
    md(
        """### Численная воспроизводимость

Смесь с компонентом `base_pair_d8` показала dev 0.933333, но при проверке Linux/macOS отличалось одно объявление в одном запросе: различия FP32 около 10⁻⁷ изменили границу топ-200 кандидатов. Этот компонент исключён из окончательной смеси целиком. Сохранён проверенный вариант с dev 0.933194; никаких исключений для отдельных `query_id` нет. Повторное CPU-предсказание ниже проверяет именно окончательную конфигурацию."""
    )
    md(
        """## Сравнение на dev

Все варианты используют одни и те же 1 200 уникальных текстов. Полнота кандидатов — верхняя граница для последующего отбора. Множество вариантов и подбор параметров могут делать dev-оценку оптимистичной; контрольный test и оценка платформы показаны отдельно."""
    )
    code(
        """rows=[]
for kind,title,family in [('lexical','Лексический','Лексический'),('dense','E5-small','E5-small'),('dense_meta','E5-small + подкатегория','E5-small')]:
    path=ARTIFACTS_DIR/f'validation_{kind}.json'
    if not path.exists(): continue
    data=json.loads(path.read_text())
    rrf=max(data['grid'],key=lambda r:r['recall50'])
    if kind!='dense_meta':
        rows.append({'Вариант':title+' / RRF','Семейство':family,'Recall@50':rrf['recall50'],'Полнота кандидатов':rrf['candidate_recall']})
    rows.append({'Вариант':title+' / CatBoost + RRF','Семейство':family,
                 'Recall@50':data['selected']['dev_recall50'],'Полнота кандидатов':rrf['candidate_recall']})
for source,title in [('title','Заголовки'),('base','E5-base + география'),('domain','Доменное дообучение')]:
    for loss in ['pair','pair_d8','logloss','logloss_d8']:
        path=ARTIFACTS_DIR/f'advanced_{source}_{loss}.json'
        if not path.exists(): continue
        data=json.loads(path.read_text())
        rows.append({'Вариант':title+' / '+loss,'Семейство':title,'Recall@50':data['recall50'],'Полнота кандидатов':data['candidate_recall']})
    path=ARTIFACTS_DIR/f'ensemble_{source}.json'
    if path.exists():
        data=json.loads(path.read_text())
        rows.append({'Вариант':title+' / ансамбль','Семейство':title,'Recall@50':data['selection']['recall50'],'Полнота кандидатов':data['candidate_recall']})
comparison=pd.DataFrame(rows)
display(comparison.drop(columns='Семейство').style.format({'Recall@50':'{:.4f}','Полнота кандидатов':'{:.4f}'}))"""
    )
    code("""import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
plot=comparison.loc[comparison.groupby('Семейство',sort=False)['Recall@50'].idxmax()]
fig,ax=plt.subplots(figsize=(11,max(3,len(plot)*.7)))
bars=ax.barh(plot['Вариант'],plot['Recall@50'],height=.58,color='#3566a1')
ax.invert_yaxis();ax.set_xlim(0,1.04);ax.set_xlabel('Средняя Recall@50 по запросам')
ax.set_title('Лучший вариант каждого семейства · dev, 1 200 текстов',loc='left',pad=14)
ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
for bar,value in zip(bars,plot['Recall@50']):
    ax.text(value+.01,bar.get_y()+bar.get_height()/2,f'{value:.4f}',va='center')
fig.tight_layout();fig.savefig(ARTIFACTS_DIR/'validation_comparison.png',dpi=150,bbox_inches='tight')
plt.show()""")
    md(
        """### Контрольная оценка и ошибки

Следующее значение получено после фиксации выбранного варианта. Примеры ошибок относятся только к dev. Ошибка поиска означает отсутствие релевантного объявления среди кандидатов; ошибка отбора — его потерю при выборе 50."""
    )
    code(
        """display(pd.DataFrame([{'Выборка':'Контрольная test','Запросов':1200,'Recall@50':report['test_recall50'],
                        'Полнота кандидатов':report['test_candidate_recall']}]))
errors=pd.read_csv(ARTIFACTS_DIR/'dev_errors.csv')
display(errors.groupby(['error_type','same_location']).size().rename('Пропущенных позитивов').reset_index())
display(errors[['query','error_type','relevant_title','same_location']].head(8))"""
    )
    md(
        """## Повторное предсказание на CPU

Команда заново вычисляет кандидатов и оценки, затем проверяет формат и SHA-256. Сохранённый `answer.csv` не читается для получения предсказаний. Внешние API и GPU не используются. Расчёт может занять несколько минут."""
    )
    code("""started=time.monotonic()
completed=subprocess.run([sys.executable,str(ROOT/'reproduce.py'),'--data-dir',str(DATA_DIR),
    '--artifacts-dir',str(ARTIFACTS_DIR),'--output',str(OUTPUT)],cwd=ROOT,
    check=True,capture_output=True,text=True)
print('\\n'.join(completed.stdout.replace(str(ROOT)+'/','').splitlines()[-4:]))
print(f'CPU-предсказание: {time.monotonic()-started:.1f} с')""")
    md(
        """### Проверка всех идентификаторов и граничных случаев

Проверяются точное множество `query_id`, две колонки, по 50 уникальных существующих `item_id` и сохранение регистра. Тесты покрывают пустые тексты, неизвестную локацию, пропуски чисел, отсутствие лексических совпадений, равные оценки, географическое сглаживание и удаление повторов при квотах."""
    )
    code("""from solution import validate
validation=validate(DATA_DIR,OUTPUT)
assert validation['sha256']==report['answer_sha256']
answer=pd.read_csv(OUTPUT,dtype=str)
display(answer.assign(answer=answer.answer.str.slice(0,85)+' …').head(3))
tests=subprocess.run([sys.executable,'-m','unittest','-q','test_solution','test_advanced'],
                     cwd=ROOT,capture_output=True,text=True,check=True)
print(tests.stderr.strip())""")
    md(
        """## Полная пересборка и open-source компоненты

Команды предсказания приведены в `README.md`, полный цикл подготовки и обучения — в `docs/TRAINING.md`. Используйте новый каталог артефактов, чтобы старые кэши не подменяли пересчёт. Загрузка исходного архива и открытых весов выполняется отдельно. Стандартное воспроизведение этого notebook работает из зафиксированных артефактов.

Использованы NumPy, pandas, PyArrow, SciPy, scikit-learn, NLTK/Snowball, CatBoost, PyTorch, Transformers, Jupyter и Matplotlib. Модели [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) и [multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) распространяются под MIT; конкретные ревизии и контрольные суммы сохранены. Ранжирование основано на [CatBoost](https://catboost.ai/docs/en/concepts/loss-functions-ranking).

Ограничения: положительные метки неполны, описания обрезаются, локальный корпус дополнен позитивами, географические статистики отражают поведение пользователей. Dev ограничен 1 200 текстами и используется для выбора нескольких вариантов. Полное переобучение в другом окружении может отличаться численно; точное воспроизведение CSV проверено из приложенных артефактов."""
    )
    md(
        """## Подтверждённые оценки платформы

Журнал содержит только сообщённые пользователем отправки. Используется суммарная метрика, без восстановления скрытой разметки и без правил для отдельных `query_id`. Общий лимит — семь попыток."""
    )
    code("""journal=json.loads((ROOT/'submissions.json').read_text())
display(pd.DataFrame(journal['attempts'])[['file','recall50','submitted_by']])
print('Подтверждённых отправок:',len(journal['attempts']))
print('Отправок агентом:',journal['submitted_by_agent'])""")
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3',
            },
            'language_info': {'name': 'python', 'version': '3.12.6'},
        },
    )
    nbf.validate(notebook)
    nbf.write(notebook, root / 'solution.ipynb')
    print('Created notebook:', len(cells), 'cells')


if __name__ == '__main__':
    main()
