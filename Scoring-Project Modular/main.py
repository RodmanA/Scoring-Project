"""
main.py
This module is the main entry point for the program
"""
from data_cleaning import get_df_csv, adjust_french_decimal, split_data
from data_statistical_analysis import plot_ratio_distributions, statistical_tests, plot_lower_correlation, plot_top_corr
from regression_estimation import run_estimation_for, compute_auc_for_models, forecast_default
from model_evaluation import pearson_residuals, w_optimal_threshold_models
import pandas as pd
import warnings


"Step 1: Load the CSV and prepare the data"
df = get_df_csv('defaut2000.csv', '-99,99')
print(df.head())
df = adjust_french_decimal(df)
print(df.head())
df = df.apply(pd.to_numeric, errors='coerce')
print(df.head())


X_train, X_test, y_train, y_test = split_data(df, Yd="yd", rows="odd")

print("X_train:")
print(X_train.head())
print("y_train:")
print(y_train.head())

# Question 3 to 10 Answers here
"Step 2: Vizualisation and analysis of the data"
train_df = pd.concat([y_train, X_train], axis=1)
print(train_df)
plot_ratio_distributions(train_df, target="yd")

table1, table2, table3, table4 = statistical_tests(train_df, target="yd")

corr_matrix = plot_lower_correlation(train_df, target="yd")

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    plot_top_corr(train_df, target_col="yd", ranking_table=table4, rank_col="Rank_corr", top_n=5)

# Question 11 to 14 Answers here
"Step 3: Regression, modeling and estimation"
# Question 11
explanatory = ["tdta"]
models_Q11 = run_estimation_for(train_df, target="yd", explanatory_vars=explanatory, output="summary")

# Question 13
explanatory = ["tdta", "opita", "lta", "gempl"]
models_Q13 = run_estimation_for(train_df, target="yd", explanatory_vars=explanatory, output="tabulate")
compute_auc_for_models(models_Q13, y_train, X_train[explanatory])

# Question 14
forecast_df = forecast_default(models_Q13, X_test, y_test, explanatory)
print(forecast_df.head())

#Question 15 to 18
"Step 4: Model evaluation and finetuning"
# Question 15/16
pearson_residuals(models_Q13, y_test, X_test[explanatory])

# Question 17
print("\n Optimal threshold computation for all models")
optimal_threshold_table = w_optimal_threshold_models(models_Q13,X_test[explanatory],y_test,LGD=0.6,margin=0.1,
                                                     plot_roc=True)
