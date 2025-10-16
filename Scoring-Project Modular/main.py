"""
main.py
This module is the main entry point for the program
"""
# In[1]: Import and Function initialisation
"Import and Function initialisation"
import os
import sys
import pandas as pd
import warnings
import statsmodels.api as sm
import numpy as np

print(os.getcwd())
project_root = os.path.abspath(os.getcwd())
directory = os.path.join(project_root, "Scoring-Project Modular")
if directory not in sys.path:
    sys.path.insert(0, directory)
    print("Project root:", directory)
    print("sys.path[0]:", sys.path[0])

from data_cleaning import get_df_csv, adjust_french_decimal, split_data
from data_statistical_analysis import (plot_ratio_distributions, statistical_tests, plot_lower_correlation,
                                       plot_top_corr)
from regression_estimation import run_estimation_for, compute_auc_for_models, forecast_default
from model_evaluation import pearson_residuals, w_optimal_threshold_models

# In[2]: CSV loading and data prep
"CSV loading and data prep"

df = get_df_csv('defaut2000.csv', '-99,99')
print(df.head())
df = adjust_french_decimal(df)
print(df.head())
df = df.apply(pd.to_numeric, errors='coerce')
print(df.head())

# 1) Non-linear transformations

# Leverage curvature (financial distress accelerates with leverage)
df["tdta_sq"] = df["tdta"] ** 2
df["ltdta_sq"] = df["ltdta"] ** 2

# Log transformation to stabilize skew
for col in ["tdta", "ltdta", "ebita", "opita", "lsls", "lta", "cacl", "qacl", "nwcta", "fata", "mveltd"]:
    df[f"log1p_{col}"] = np.log1p(df[col].clip(lower=0))  # avoid log of negative

# Inverse transformation for liquidity and leverage
for col in ["tdta", "ltdta", "cacl", "qacl"]:
    df[f"inv_{col}"] = 1 / (1 + df[col].replace(0, np.nan))

# 2) Interaction features

# Leverage × profitability (amplified risk when profit is low and debt high)
df["tdta_x_ebita"] = df["tdta"] * df["ebita"]
df["ltdta_x_opita"] = df["ltdta"] * df["opita"]

# Leverage × liquidity (debt pressure vs buffer)
df["tdta_x_cacl"] = df["tdta"] * df["cacl"]
df["ltdta_x_qacl"] = df["ltdta"] * df["qacl"]

# Working capital × leverage (short-term liquidity under debt stress)
df["nwcta_x_tdta"] = df["nwcta"] * df["tdta"]

# Profitability × firm size
df["ebita_x_lta"] = df["ebita"] * df["lta"]

#  3) Composite indicators

# Liquidity gap (proxy for inventory intensity)
df["liq_gap"] = df["cacl"] - df["qacl"]

# Earnings efficiency: EBIT relative to asset base
df["ebit_to_lta"] = df["ebita"] / (df["lta"].replace(0, np.nan))

# Market leverage proxy
df["mveltd_to_tdta"] = df["mveltd"] / (df["tdta"].replace(0, np.nan))

# Altman-inspired solvency ratio (simplified version)
df["altman_like"] = (
        df["nwcta"] +
        df["reta"] +
        df["ebita"] +
        df["mveltd"] -
        df["tdta"]
)

# In[3]: Sample splitting
"Sample splitting"
X_train, X_test, y_train, y_test = split_data(df, Yd="yd", rows="odd")

print("X_train:")
print(X_train.head())
print("y_train:")
print(y_train.head())

# In[4]: Question 3 : Distribution Plot of the Training data
"Question 3 : Distribution Plot of the Training data"

train_df = pd.concat([y_train, X_train], axis=1)
print(train_df)
exclude_cols = ['tdta_sq',
                'ltdta_sq', 'log1p_tdta', 'log1p_ltdta', 'log1p_ebita', 'log1p_opita',
                'log1p_lsls', 'log1p_lta', 'log1p_cacl', 'log1p_qacl', 'log1p_nwcta',
                'log1p_fata', 'log1p_mveltd', 'inv_tdta', 'inv_ltdta', 'inv_cacl',
                'inv_qacl', 'tdta_x_ebita', 'ltdta_x_opita', 'tdta_x_cacl',
                'ltdta_x_qacl', 'nwcta_x_tdta', 'ebita_x_lta', 'liq_gap', 'ebit_to_lta',
                'mveltd_to_tdta', 'altman_like']
cols_to_plot = train_df.drop(columns=exclude_cols).columns
plot_ratio_distributions(train_df[cols_to_plot], target="yd")

# In[] : Question 3 : Normality Test and correlation analysis, Question 5/6 : t-stat/Correlation ranking,
# Question 8 : Bivariate correlation
table1, table2, table3, table4 = statistical_tests(train_df[cols_to_plot], target="yd")
corr_matrix = plot_lower_correlation(train_df[cols_to_plot], target="yd")

# In[] Question 10 : Bivariate clouds of point
"Question 10 : Bivariate clouds of point"
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    plot_top_corr(train_df, target_col="yd", ranking_table=table4, rank_col="Rank_corr", top_n=5)

# In[] Question 11 : Default regression table
"Question 11 : Default regression table"

explanatory = ["tdta"]
print("Default output :")
models_Q11_co = run_estimation_for(train_df, target="yd", explanatory_vars=explanatory, output="summary")
print("Tabulated output :")
models_Q11 = run_estimation_for(train_df, target="yd", explanatory_vars=explanatory, output="tabulate",
                                prob_var=explanatory)

# In[] Document C : Optimisation 1, variable combination testing
"Optimisation by k-fold cross validation for choosing variable"

import numpy as np
import itertools
from tqdm import tqdm
from model_evaluation import cross_val_auc

# PARAMETERS
target_col = "yd"
all_columns = [col for col in train_df.columns if col != target_col]
max_vars = 3
n_splits = 5  # k-fold CV
random_state = 42

# Variable combo search with CV
auc_records = []
total_combos = sum(1 for k in range(1, max_vars + 1) for _ in itertools.combinations(all_columns, k))
print(f"Total combinations to test: {total_combos}")

with tqdm(total=total_combos, desc="CV variable combo testing", unit="combo") as pbar:
    for k in range(1, max_vars + 1):
        for combo in itertools.combinations(all_columns, k):
            explanatory = list(combo)
            try:
                mean_auc, std_auc = cross_val_auc(train_df, explanatory_vars=explanatory, target_col=target_col,
                                                  random_state=random_state, model_type="Logit",
                                                  number_of_split=n_splits)
                auc_records.append({
                    "Variables": explanatory,
                    "Model": "Logit",
                    "AUC_mean": mean_auc,
                    "AUC_std": std_auc
                })
            except Exception as e:
                print(f"Skipped combo {combo} due to error: {e}")
            finally:
                pbar.update(1)

# Convert results to DataFrame
auc_df = pd.DataFrame(auc_records)
auc_df_sorted = auc_df.sort_values(by="AUC_mean", ascending=False)
top25_auc = auc_df_sorted.head(25)
print(top25_auc)

# %%
"Shapley modified score for choosing variable"
from collections import defaultdict
from math import factorial

# Preprocess variables into sorted tuples
auc_df_sorted['Variables'] = auc_df_sorted['Variables'].apply(lambda x: tuple(sorted(x)))

# Compute AUC mean and std grouped by combo
subset_auc_mean = auc_df_sorted.groupby('Variables')['AUC_mean'].mean().to_dict()
subset_auc_std = auc_df_sorted.groupby('Variables')['AUC_std'].mean().to_dict()

# Define stability-adjusted score
lambda_penalty = 0.5
subset_auc_adj = {}
for k in subset_auc_mean.keys():
    mu = subset_auc_mean.get(k, 0)
    sigma = subset_auc_std.get(k, 0)
    subset_auc_adj[k] = mu - lambda_penalty * sigma

# List of all variables
all_vars = sorted(set(v for combo in auc_df_sorted['Variables'] for v in combo))
n = len(all_vars)

shapley_values = defaultdict(float)

# Compute Shapley values on adjusted score
for var in all_vars:
    other_vars = [v for v in all_vars if v != var]
    for r in range(0, min(max_vars, len(other_vars)) + 1):
        for S in itertools.combinations(other_vars, r):
            S = tuple(sorted(S))
            S_with_var = tuple(sorted(S + (var,)))

            if S_with_var not in subset_auc_adj:
                continue

            v_S = subset_auc_adj.get(S, 0.0)
            v_S_with_var = subset_auc_adj.get(S_with_var, 0.0)

            s = len(S)
            weight = factorial(s) * factorial(n - s - 1) / factorial(n)

            shapley_values[var] += weight * (v_S_with_var - v_S)

# SConvert to DataFrame and sort
shapley_df = pd.DataFrame([
    {'variable': var, 'shapley_value': val}
    for var, val in shapley_values.items()
])

shapley_df['shapley_value'] = shapley_df['shapley_value'].fillna(0)
shapley_df = shapley_df.sort_values('shapley_value', ascending=False).reset_index(drop=True)
print(shapley_df.head(50))

top_var_list = shapley_df['variable'].head(8).tolist()
print(top_var_list)

# In[] Question 13 : Models estimation and ROC curve
"Question 13 : Models estimation and ROC curve"
explanatory_bench = ["tdta", "opita", "gempl"]
models_Q13_bench = run_estimation_for(train_df, target="yd", explanatory_vars=explanatory_bench, output="tabulate",
                                      prob_var=explanatory_bench)
compute_auc_for_models(models_Q13_bench["models"], y_train, X_train[explanatory_bench])
"Question 13 : Models estimation and ROC curve"
explanatory = ["tdta_sq", "log1p_opita", "gempl"]
models_Q13 = run_estimation_for(train_df, target="yd", explanatory_vars=explanatory, output="tabulate",
                                prob_var=explanatory)
compute_auc_for_models(models_Q13["models"], y_train, X_train[explanatory])

# In[] Question 14 : Models validation and forecasting
"Question 14 : Models validation and forecasting"
forecast_df_bench = forecast_default(models_Q13_bench["models"], X_test, y_test, explanatory_bench)
print(forecast_df_bench.head())

forecast_df = forecast_default(models_Q13["models"], X_test, y_test, explanatory)
print(forecast_df.head())

# In[] Question 15/16 : Standardized Pearson residuals
"Question 15/16 : Standardized Pearson residuals"
pearson_residuals(models_Q13_bench["models"], y_test, X_test[explanatory_bench], explanatory_bench)
pearson_residuals(models_Q13["models"], y_test, X_test[explanatory], explanatory)

# In[] Question 17 : Loss function and optimal treshold
"Question 17 : Loss function and optimal treshold"
print("\n Optimal threshold computation for all models")
optimal_threshold_table = w_optimal_threshold_models(models_Q13["models"], X_test[explanatory], y_test, LGD=0.6,
                                                     margin=0.1,
                                                     plot_roc=True)

# In[] Question 19 : Dummy Trap
from data_statistical_analysis import export_tables_as_png

"Question 19 : Dummy Trap"

df["ynd"] = 1 - df["yd"]  # Building dummy, 1 for non-defaulting firms
y = df["tdta"].astype(float)

#  Group means
mean_def = df.loc[df["yd"] == 1, "tdta"].mean()
mean_ndef = df.loc[df["yd"] == 0, "tdta"].mean()
overall = df["tdta"].mean()

print(f"\n Mean TDTA by group:")
print(f"   • Overall mean        = {overall:.4f}")
print(f"   • Defaulting firms    = {mean_def:.4f}")
print(f"   • Non-defaulting firms = {mean_ndef:.4f}")
print(f"   • Difference (Δ)       = {mean_def - mean_ndef:.4f}")


#  Helper function
def fit_and_print(X, title, constrained=False, constraint=None):
    X = sm.add_constant(X, has_constant="add")
    if constrained and constraint is not None:
        res = sm.OLS(y, X).fit_constrained(constraint)
    else:
        res = sm.OLS(y, X).fit()
    print(f"\n=== {title} ===")
    print(res.summary().tables[1])  # only coefficient table
    return res


# 1 TDTA ~ const + yd + ynd (Dummy trap)
res1 = fit_and_print(df[["yd", "ynd"]],
                     "1) TDTA on yd and ynd with common intercept (Dummy trap)")

# 2 TDTA ~ const + yd
res2 = fit_and_print(df[["yd"]],
                     "2) TDTA on yd with common intercept")

# 3 TDTA ~ const + ynd
res3 = fit_and_print(df[["ynd"]],
                     "3) TDTA on ynd with common intercept")

# 4 TDTA ~ const + (yd - ynd)
df["yd_minus_ynd"] = df["yd"] - df["ynd"]
res4 = fit_and_print(df[["yd_minus_ynd"]],
                     "4) TDTA on (yd - ynd) — Restricted model (β_yd + β_ynd = 0)")


# Extract parameters
def safe_get(res, variable):
    return float(res.params.get(variable, np.nan))


summary_data = pd.DataFrame({
    "Model": [
        "1) Dummy trap: yd + ynd + const",
        "2) yd with intercept",
        "3) ynd with intercept",
        "4) (yd - ynd), restricted"
    ],
    "Intercept": [
        safe_get(res1, "const"),
        safe_get(res2, "const"),
        safe_get(res3, "const"),
        safe_get(res4, "const")
    ],
    "Coef (yd / ynd / yd-ynd)": [
        f"yd={safe_get(res1, 'yd'):.4f}, ynd={safe_get(res1, 'ynd'):.4f}",
        f"{safe_get(res2, 'yd'):.4f}",
        f"{safe_get(res3, 'ynd'):.4f}",
        f"{safe_get(res4, 'yd_minus_ynd'):.4f}"
    ]
})

print("\nSummary of All Four Regressions:")
print(summary_data.to_markdown(index=False))

# Export summary table as PNG
export_tables_as_png({"Dummy_Trap_Summary": summary_data})

# Verification checks
print("\n--- Identities to Verify ---")
print(f"(2) Intercept ≈ mean(non-default): {safe_get(res2, 'const'):.4f} vs {mean_ndef:.4f}")
print(f"(2) Coef yd   ≈ mean(default) - mean(non-default): {safe_get(res2, 'yd'):.4f} vs {(mean_def - mean_ndef):.4f}")
print(f"(3) Intercept ≈ mean(default): {safe_get(res3, 'const'):.4f} vs {mean_def:.4f}")
print(f"(3) Coef ynd  ≈ mean(non-default) - mean(default): {safe_get(res3, 'ynd'):.4f} vs {(mean_ndef - mean_def):.4f}")
print(f"(4) Intercept ≈ overall mean: {safe_get(res4, 'const'):.4f} vs {overall:.4f}")
print(f"(4) Coef (yd - ynd) ≈ (mean_default - mean_nondefault)/2: "
      f"{safe_get(res4, 'yd_minus_ynd'):.4f} vs {0.5 * (mean_def - mean_ndef):.4f}")
print("Restriction enforced: β_yd + β_ynd = 0 (β_yd = -β_ynd = coef)")
