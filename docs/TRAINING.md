# Полная пересборка и обучение


Используйте новый пустой каталог `artifacts/`: этапы кэшируются. Загрузки архива и открытых весов отделены от вычислений. Установка PyTorch ниже соответствует проверенным GTX 1080 Ti; зависимости офлайн-предсказания существенно меньше.

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python download_data.py --data-dir data
python download_model.py --artifacts-dir artifacts
python encode_extra.py download --model base --artifacts-dir artifacts
python solution.py prepare --data-dir data --artifacts-dir artifacts
python audit_data.py
python solution.py index --artifacts-dir artifacts
python solution.py index --full --artifacts-dir artifacts
python solution.py prior --artifacts-dir artifacts
python solution.py prior --full --artifacts-dir artifacts
python geo_prior.py --data-dir data --artifacts-dir artifacts
python geo_prior.py --full --data-dir data --artifacts-dir artifacts

python solution.py encode --gpu 0 --shard 0 --shards 2 --batch-size 64 &
python solution.py encode --gpu 1 --shard 1 --shards 2 --batch-size 64 &
wait
python solution.py merge --shards 2
python solution.py train --dense --metadata --ranker

python encode_extra.py encode --model small --gpu 0 --shard 0 --batch-size 64 &
python encode_extra.py encode --model small --gpu 1 --shard 1 --batch-size 64 &
wait
python encode_extra.py merge --model small
python encode_extra.py encode --model base --gpu 0 --shard 0 --batch-size 32 &
python encode_extra.py encode --model base --gpu 1 --shard 1 --batch-size 32 &
wait
python encode_extra.py merge --model base
python advanced.py train --sources base --loss logloss
python advanced.py train --sources base --loss pair
python advanced.py train --sources base --loss logloss --depth 8 --tag d8
python advanced.py train --sources base --loss pair --depth 8 --tag d8
python ensemble.py tune --sources base
```

Доменное дообучение, входящее в итоговый вариант:

```bash
python finetune_e5.py --gpu 0 --steps 800 --batch-size 48 --max-seconds 86400
python encode_extra.py encode --model domain --gpu 0 --shard 0 --batch-size 64 &
python encode_extra.py encode --model domain --gpu 1 --shard 1 --batch-size 64 &
wait
python encode_extra.py merge --model domain
python advanced.py train --sources domain --loss logloss
python ensemble.py tune --sources domain --exclude-component base_pair_d8
```

Для воспроизведения сравнений дополнительно выполняются `solution.py train --ranker`, `solution.py train --dense --ranker`, `advanced.py train --sources title --loss pair`, `advanced.py train --sources title --loss logloss`, `ensemble.py tune --sources title`. Необходимость других PairLogit-моделей видна по таблице и приложенным конфигурациям.

После выбора по dev фиксируются контрольная оценка и CSV:

```bash
python finalize_v2.py --engine ensemble --sources domain --loss logloss --data-dir data --artifacts-dir artifacts --output answer.csv
python solution.py validate --data-dir data --output answer.csv
```

Численные результаты полного переобучения в другом окружении могут отличаться. Проверенный способ получить тот же CSV — офлайн-предсказание из приложенных зафиксированных артефактов. Запуск `finalize_v2.py` заново оценивает test только для отчёта; он не выбирает параметры.


Для получения того же отправленного CSV используйте зафиксированные артефакты и `reproduce.py`, как описано в [README](../README.md). Полная пересборка нужна для проверки алгоритма или новых экспериментов; её побитовое совпадение после нового обучения не гарантируется.
