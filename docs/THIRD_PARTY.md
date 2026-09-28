# Использованные открытые компоненты

Алгоритм и код решения находятся в этом репозитории. Внешние модели и библиотеки используются локально; сервисы генерации предсказаний не вызываются.

| Компонент | Роль | Источник |
|---|---|---|
| Multilingual E5-small / E5-base | Семантические представления запросов и объявлений | [Карточка small](https://huggingface.co/intfloat/multilingual-e5-small), [карточка base](https://huggingface.co/intfloat/multilingual-e5-base) |
| PyTorch, Transformers | Доменное дообучение и кодирование текстов | [PyTorch](https://pytorch.org/), [Transformers](https://github.com/huggingface/transformers) |
| CatBoost | Обучаемый отбор кандидатов | [CatBoost](https://github.com/catboost/catboost) |
| scikit-learn, SciPy, NLTK | Разреженный поиск, символьный TF-IDF, MultinomialNB, стемминг | [scikit-learn](https://scikit-learn.org/), [SciPy](https://scipy.org/), [NLTK](https://www.nltk.org/) |
| NumPy, pandas, PyArrow | Вычисления, таблицы и Parquet | [NumPy](https://numpy.org/), [pandas](https://pandas.pydata.org/), [Arrow](https://arrow.apache.org/docs/python/) |
| Jupyter, Matplotlib | Воспроизводимый notebook и график сравнения | [Jupyter](https://jupyter.org/), [Matplotlib](https://matplotlib.org/) |

Карточки E5 указывают MIT. Сохранена [лицензия проекта E5 / UniLM](licenses/E5-MIT.txt), источник — [microsoft/unilm/LICENSE](https://github.com/microsoft/unilm/blob/master/LICENSE). Исходные карточки моделей и лицензия также включены в архив артефактов. Доменная модель является локально дообученной E5-small; параметры модификации записаны в `artifacts/e5-domain/training_config.json`.

Ревизии моделей и SHA-256 весов находятся в `artifacts/final_report.json`. Версии используемых пакетов — в [environment.json](environment.json), зависимости запуска — в requirements-файлах корня. Форматирование Python выполнено Black с проверкой неизменности AST; это инструмент оформления, не зависимость предсказания.

Исходные данные получены из [архива задания](https://disk.yandex.ru/d/sNhfo0YOjGtufg). Их повторная упаковка служит воспроизводимости этого решения и не меняет условия исходного источника.
