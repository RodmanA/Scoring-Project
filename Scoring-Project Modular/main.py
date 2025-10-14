"""
main.py
This module is the main entry point for the program
"""
# In[1]: Import and Function initialisation
"Import and Function initialisation"
import os
import sys
print(os.getcwd())
project_root = os.path.abspath(os.getcwd())
dir = os.path.join(project_root, "Scoring-Project Modular")
if dir not in sys.path:
    sys.path.insert(0, dir)
    print("Project root:", dir)
    print("sys.path[0]:", sys.path[0])
import itertools
from data_cleaning import get_df_csv, adjust_french_decimal, split_data
from data_statistical_analysis import plot_ratio_distributions, statistical_tests, plot_lower_correlation, plot_top_corr
from regression_estimation import run_estimation_for, compute_auc_for_models, forecast_default
from model_evaluation import pearson_residuals, w_optimal_threshold_models
from tqdm import tqdm
import pandas as pd
import warnings
import statsmodels.api as sm
import numpy as np
import matplotlib

# In[2]: CSV loading and data prep
"CSV loading and data prep"

df = get_df_csv('defaut2000.csv', '-99,99')
print(df.head())
df = adjust_french_decimal(df)
print(df.head())
df = df.apply(pd.to_numeric, errors='coerce')
print(df.head())

# New ratios / transforms

df["tdta_x_ebita"] = df["tdta"] * df["ebita"]  # Interaction: leverage × profitability  
df["tdta_sq"] = df["tdta"] ** 2            #  curvature in leverage
df["liq_gap"] = df["cacl"] - df["qacl"]    # liquidity gap: inventories proxy (current ratio - quick ratio)
df["ebit_to_lta"] = df["ebita"] / df["lta"]   # efficiency of assets in producing earnings
df["qacl_x_tdta"] = df["qacl"] * df["tdta"]   # Quick ratio × Leverage
df["cacl_x_tdta"] = df["cacl"] * df["tdta"]   # Current ratio × Leverage
df["ebita_x_lta"] = df["ebita"] * df["lta"]     # Profitability × Size
df["ebita_x_gempl"] = df["ebita"] * df["gempl"] # Profitability × Employment growth

df["qacl_x_tdta"] = df["qacl"] * df["tdta"]
df["ebita_x_fata"] = df["ebita"] * df["fata"]
df["gempl_x_ebita"] = df["gempl"] * df["ebita"]
df["tdta_x_liqgap"] = df["tdta"] * df["liq_gap"]
df["tdta_x_fata"]  = df["tdta"] * df["fata"]
df["tdta_x_lta"]   = df["tdta"] * df["lta"]



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
plot_ratio_distributions(train_df, target="yd")


# In[] : Question 3 : Normality Test and correlation analysis, Question 5/6 : t-stat/Correlation ranking,
# Question 8 : Bivariate correlation
table1, table2, table3, table4 = statistical_tests(train_df, target="yd")
corr_matrix = plot_lower_correlation(train_df, target="yd")


# In[] Question 10 : Bivariate clouds of point
"Question 10 : Bivariate clouds of point"
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    plot_top_corr(train_df, target_col="yd", ranking_table=table4, rank_col="Rank_corr", top_n=5)


# In[] Question 11 : Default regression table
"Question 11 : Default regression table"

explanatory = ["tdta"]
print("Default output :")
models_Q11 = run_estimation_for(train_df, target="yd", explanatory_vars=explanatory, output="summary")
print("Tabulated output :")
models_Q11 = run_estimation_for(train_df, target="yd", explanatory_vars=explanatory, output="tabulate",prob_var=explanatory)


# In[] Question 13 : Models estimation and ROC curve
"Question 13 : Models estimation and ROC curve"
explanatory = ["tdta", "opita", "lta", "gempl"]
models_Q13 = run_estimation_for(train_df, target="yd", explanatory_vars=explanatory, output="tabulate",prob_var=explanatory)
compute_auc_for_models(models_Q13["models"], y_train, X_train[explanatory])


# In[] Question 14 : Models validation and forecasting
"Question 14 : Models validation and forecasting"
forecast_df = forecast_default(models_Q13["models"], X_test, y_test, explanatory)
print(forecast_df.head())

# In[] Question 15/16 : Standardized Pearson residuals
"Question 15/16 : Standardized Pearson residuals"
pearson_residuals(models_Q13["models"], y_test, X_test[explanatory])


# In[] Question 17 : Loss function and optimal treshold
"Question 17 : Loss function and optimal treshold"
print("\n Optimal threshold computation for all models")
optimal_threshold_table = w_optimal_threshold_models(models_Q13["models"],X_test[explanatory],y_test,LGD=0.6,margin=0.1,
                                                     plot_roc=True)


# In[] Question 19 : Dummy Trap
"Question 19 : Dummy Trap"

df["ynd"] = 1 - df["yd"]  # Building dummy, 1 for non-defaulting firms
y = df["tdta"].astype(float)

#  Group means 
mean_def  = df.loc[df["yd"] == 1, "tdta"].mean()
mean_ndef = df.loc[df["yd"] == 0, "tdta"].mean()
overall   = df["tdta"].mean()

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
# --- Extract parameters ---
def safe_get(res, var): 
    return float(res.params.get(var, np.nan))

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
        f"yd={safe_get(res1,'yd'):.4f}, ynd={safe_get(res1,'ynd'):.4f}",
        f"{safe_get(res2,'yd'):.4f}",
        f"{safe_get(res3,'ynd'):.4f}",
        f"{safe_get(res4,'yd_minus_ynd'):.4f}"
    ]
})
print("\nSummary of All Four Regressions:")
print(summary_data.to_markdown(index=False))
# --- Verification checks ---
print("\n--- Identities to Verify ---")
print(f"(2) Intercept ≈ mean(non-default): {safe_get(res2,'const'):.4f} vs {mean_ndef:.4f}")
print(f"(2) Coef yd   ≈ mean(default) - mean(non-default): {safe_get(res2,'yd'):.4f} vs {(mean_def-mean_ndef):.4f}")
print(f"(3) Intercept ≈ mean(default): {safe_get(res3,'const'):.4f} vs {mean_def:.4f}")
print(f"(3) Coef ynd  ≈ mean(non-default) - mean(default): {safe_get(res3,'ynd'):.4f} vs {(mean_ndef-mean_def):.4f}")
print(f"(4) Intercept ≈ overall mean: {safe_get(res4,'const'):.4f} vs {overall:.4f}")
print(f"(4) Coef (yd - ynd) ≈ (mean_default - mean_nondefault)/2: "
      f"{safe_get(res4,'yd_minus_ynd'):.4f} vs {0.5*(mean_def-mean_ndef):.4f}")
print("Restriction enforced: β_yd + β_ynd = 0 (β_yd = -β_ynd = coef)")


# In[] Document C : Optimisation 1, variable combination testing

matplotlib.use('Agg')
"Document C : Optimisation 1, variable combination testing"
target_col= "yd"
all_columns = [col for col in train_df.columns if col!= target_col]
result=[]
auc_records=[]
max_vars = 3

total_combos = sum(
    1 for k in range(1, max_vars + 1)
    for _ in itertools.combinations(all_columns, k)
)
print(f"Total combinations to test: {total_combos}")

with tqdm(total=total_combos, desc="Variable combo testing", unit="combo") as pbar:
    for k in range(1, max_vars + 1):
        for combo in itertools.combinations(all_columns, k):
            explanatory = list(combo)
            try:
                models_dict = run_estimation_for(
                    train_df,
                    target=target_col,
                    explanatory_vars=explanatory,
                    output="tabulate",
                    prob_var=explanatory,
                    silent=True
                )

                forecast_df_rank = forecast_default(
                    models_dict["models"], X_test, y_test, explanatory, plot=False, silent=True
                )

                for idx, row in forecast_df_rank.iterrows():
                    auc_records.append({
                        "Variables": explanatory,
                        "Model": row["Model"],
                        "AUC": row["AUC"]
                    })

            except Exception as e:
                print(f"Skipped combo {combo} due to error: {e}")

            finally:
                pbar.update(1)
    

auc_df = pd.DataFrame(auc_records)
auc_df_sorted = auc_df.sort_values(by="AUC", ascending=False)
top10_auc = auc_df_sorted.head(10)
print(top10_auc)
matplotlib.use('TkAgg')

# %%
