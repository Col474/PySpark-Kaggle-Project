# PySpark-Kaggle-Project
A Project which uses PySpark to train an ML model to compete in the House Prices Kaggle competition. The competiion can be found here https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data.

This project successfully implemented an end-to-end machine learning pipeline in 
Apache PySpark to predict house prices on the Ames Housing dataset. Four MLlib 
regression models were trained, tuned with cross-validated grid search, and evaluated. 
Random Forest was consistently the best model across both submissions. The first 
submission achieved a Kaggle score of 0.14812 (rank 2,777). The second submission, 
incorporating five engineered features and a log-transform of the target variable, 
improved this to 0.14344 (rank 2,388) — a meaningful improvement achieved by a 
targeted methodological change rather than simply increasing model complexity.
The project demonstrated several important aspects of working with PySpark in a 
resource-constrained environment: managing Spark's lazy evaluation model through 
strategic caching, configuring the SparkSession for small-data workloads, parallelising 
cross-validation to reduce training time, and structuring a two-stage pipeline to 
separate preprocessing from model fitting. The final pipeline is generalisable and can be 
applied to new data by applying the same preprocessing and feature engineering 
functions followed by the fitted model.
