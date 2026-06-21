# House Prices — Apache Spark ML Pipeline

An end-to-end Apache Spark (PySpark MLlib) regression pipeline built for Kaggle's [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) competition. Covers data cleaning, categorical/ordinal encoding, feature selection, cross-validated model comparison across four regression algorithms, and competition submission.

## Results

| Model | Test RMSE | R² | CV RMSE | Train time |
|---|---|---|---|---|
| **Random Forest** (200 trees, depth 15) | **27,468** | **0.89** | 29,633 | — |
| Linear Regression | ~33,700 | 0.84 | — | 59s |
| GBT | ~33,800 | 0.84 | — | 664s |
| Decision Tree | 39,611 | — | — | — |

Random Forest was the strongest and most consistent performer and was selected for the final submission. Linear Regression matched GBT's accuracy at a fraction of the training time. Decision Tree underperformed as expected, lacking the variance reduction of ensemble methods.

**Attempt 2** re-trained Random Forest on a log-transformed target (matching Kaggle's log-RMSE scoring), added engineered features (total square footage, total bathrooms, house age, years since remodel, quality × area interaction), and expanded the hyperparameter grid — improving R² from 0.89 to 0.97.

## Pipeline overview

1. **Data inspection & cleaning** — null/`"NA"`-string auditing, type correction for fields incorrectly inferred as strings (`LotFrontage`, `MasVnrArea`), outlier review (outliers retained as genuine signal rather than noise).
2. **Feature encoding**
   - Categorical fields → `StringIndexer` + `OneHotEncoder`
   - Ordinal fields (e.g. quality/condition ratings) → manually mapped to preserve natural ordering, since one-hot encoding would discard it
3. **Feature selection** — Spark's `UnivariateFeatureSelector` (F-regression test), tuned to the top 50 features
4. **Model comparison** — Linear Regression, Decision Tree, Random Forest, and GBT, each tuned via `ParamGridBuilder` + `CrossValidator`
5. **Submission** — best model applied to the competition test set, predictions formatted and submitted via the Kaggle CLI

A second iteration (**Attempt 2**) builds on the above with a log-transformed target, engineered features, and a refined Random Forest grid.

## Performance notes

A few Spark-specific optimisations made a meaningful difference to training time:

- **Shuffle partitions** set to `2 × CPU cores` rather than Spark's default of 200, which is oversized for this dataset and adds unnecessary overhead.
- **Caching** the fully-preprocessed training DataFrame before cross-validation, since Spark's lazy evaluation would otherwise re-run the entire transformation chain (read, cast, encode, split) on every fold.
- **Parallelism** in `CrossValidator` to evaluate multiple folds concurrently rather than sequentially.

## Tech stack

- Apache Spark (PySpark) — `pyspark.ml` for the full pipeline (`StringIndexer`, `OneHotEncoder`, `VectorAssembler`, `StandardScaler`, `UnivariateFeatureSelector`, `CrossValidator`)
- Kaggle API for data download and submission

## Possible next steps

- Wider feature selection threshold testing
- Further feature engineering
- Broader model coverage, including a less time-constrained GBT hyperparameter search

## Running it

This notebook was developed in Google Colab. To run locally:

```bash
pip install pyspark kaggle
```

Add your Kaggle API credentials (`~/.kaggle/kaggle.json`) and run the notebook top to bottom — it handles data download via the Kaggle CLI automatically.

---
*Built as part of an MSc Data Science assignment on Apache Spark / processing big data.*
