# Классификатор тренировочных ограничений по анамнезу

Определяет по свободному тексту жалобы/анамнеза одну из 8 групп тренировочных ограничений,
чтобы фитнес-приложение или тренер могли адаптировать программу и снизить риск травм.

## Группы ограничений

Позвоночник · Суставы · Кардио+невро · Дыхательные · Эндокрин/метабол · ЖКТ/брюшные ·
Мочеполовые/таз · Кожа.

Группы получены переформулировкой кодов МКБ-10 под задачу (см. `src/mapping.py`).

## Данные

`data/*.jsonl` — train (4640), dev (839), test (~813). Поля: `idx`, `symptoms`, `code`.
Коды без отношения к тренировкам (D50/H65/Z00) исключаются. Главная метрика — **macro-F1**
(классы несбалансированы, дисбаланс ~7×).

## Модели и результаты (dev)

| Модель | accuracy | macro-F1 |
|---|---|---|
| tf-idf + LogReg | 0.855 | 0.841 |
| CatBoost на fastText | 0.839 | 0.828 |
| fastText + LogReg | 0.833 | 0.822 |
| RuBioRoBERTa (frozen) + MLP | 0.811 | 0.815 |
| RuBioRoBERTa (файнтюн, LoRA) | 0.851 | **0.853** |

## Запуск

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python -m src.baseline --split dev
python -m src.fasttext_clf --split dev
python -m src.catboost_clf --split dev
pytest
```

Трансформер (RuBioRoBERTa) считается на GPU — см. `notebooks/02_transformer_colab.ipynb`
для Google Colab. Тяжёлые зависимости — в `requirements-transformer.txt`.

## Структура

```
src/        модели, маппинг, загрузка данных, метрики
notebooks/  EDA и Colab-ноутбук трансформера
tests/      pytest
reports/    графики и метрики
data/       сплиты
```
