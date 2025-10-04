"""
regression_estimation.py
This module contain all the function relative to estimation and regression analysis
Relevent for Question 11 to
"""
from data_statistical_analysis import significance_stars
import statsmodels.api as sm
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np


def run_estimation_for(df, target, explanatory_vars, output="summary"):
    """
    Run a Linear Probability Model, Logit, and Probit regression
    for a given dataframe and list of explanatory variables.
    Returns results_dict : dict as a dictionary of fitted model objects.
    """

    # Prepare data
    df_model = df[[target] + explanatory_vars].dropna().copy()
    y = df_model[target]
    X = sm.add_constant(df_model[explanatory_vars])

    # Estimate models
    lpm = sm.OLS(y, X).fit(cov_type="HC3")
    logit = sm.Logit(y, X).fit(disp=False)
    probit = sm.Probit(y, X).fit(disp=False)

    results_dict = {"LPM": lpm, "Logit": logit, "Probit": probit}

    # --- Classic output ---
    if output == "summary":
        print("\n Linear Probability Model (LPM):\n")
        print(lpm.summary())

        print("\n Logit Model:\n")
        print(logit.summary())

        print("\n Probit Model:\n")
        print(probit.summary())
        return results_dict

    # --- Tabulated output ---
    elif output == "tabulate":
        rows = []
        variables = X.columns

        # --- Main regression coefficients ---
        for var in variables:
            row = [var]
            for model_name, model in results_dict.items():
                params = model.params.get(var, np.nan)
                bse = model.bse.get(var, np.nan)
                pval = model.pvalues.get(var, np.nan)
                row.append(f"{params:.3f}{significance_stars(pval)}")
                row.append(f"({bse:.3f})")
            rows.append(row)

        # --- Compute AUCs ---
        auc_results = {}
        for name, model in results_dict.items():
            try:
                y_pred = model.predict(X)
                auc_results[name] = roc_auc_score(y, y_pred)
            except Exception:
                auc_results[name] = np.nan

        # --- Counts ---
        n_total = len(y)
        n_def = int((y == 1).sum())
        n_nondef = int((y == 0).sum())

        # --- Add performance rows to main table ---
        summary_rows = [
            ["# Obs (Total)", f"{n_total}", "", "", "", "", ""],
            ["# Default (yd=1)", f"{n_def}", "", "", "", "", ""],
            ["# Non-default (yd=0)", f"{n_nondef}", "", "", "", "", ""],
            ["AUC",
             f"{auc_results.get('LPM', np.nan):.3f}",
             "",
             f"{auc_results.get('Logit', np.nan):.3f}",
             "",
             f"{auc_results.get('Probit', np.nan):.3f}",
             ""]
        ]
        rows.extend(summary_rows)

        # --- Headers ---
        headers = ["Variable"]
        for model in ["LPM", "Logit", "Probit"]:
            headers += [f"{model} β", f"{model} (SE)"]

        print("\n Comparative Regression Table (LPM / Logit / Probit)\n")
        print(tabulate(rows, headers=headers, tablefmt="github"))

        return results_dict

    else:
        raise ValueError("Invalid output type. Use 'summary' or 'tabulate'.")


def compute_auc_for_models(results_dict, y, X):
    """
    Compute and plot ROC curves and AUC scores for each model.
    Works with statsmodels Logit, Probit, and OLS.
    """
    plt.figure(figsize=(7, 6))
    print("\n Model AUC Scores:")

    # Add constant if missing
    if "const" not in X.columns:
        X = sm.add_constant(X)

    for name, model in results_dict.items():
        try:
            y_pred = model.predict(X)
            auc = roc_auc_score(y, y_pred)
            fpr, tpr, _ = roc_curve(y, y_pred)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
            print(f"{name:8s} → AUC = {auc:.3f}")
        except Exception as e:
            print(f"{name}: Failed to compute AUC — {e}")

    plt.plot([0, 1], [0, 1], "k--", label="Random chance")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("ROC Curves for LPM, Logit, and Probit Models")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()