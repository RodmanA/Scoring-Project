"""
regression_estimation.py
This module contain all the function relative to estimation and regression analysis
Relevent for Question 11 to 14
"""
from data_statistical_analysis import significance_stars
import statsmodels.api as sm
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd
import numpy as np
from scipy.stats import norm


def run_estimation_for(df, target, explanatory_vars, output="summary", prob_var=None,silent =False):
    """
    Run LPM, Logit, and Probit models, and display:
    - Regression coefficients
    - AUC and counts
    - Probability interpretation (PD at 0.25 and 0.50) for chosen variable(s).
    """

    # Normalize prob_var to list
    if prob_var is None:
        prob_vars = []
    elif isinstance(prob_var, str):
        prob_vars = [prob_var]
    else:
        prob_vars = list(prob_var)

    # Prepare data
    df_model = df[[target] + explanatory_vars].dropna().copy()
    y = df_model[target]
    X = sm.add_constant(df_model[explanatory_vars])

    # Fit models
    lpm = sm.OLS(y, X).fit(cov_type="HC3")
    logit = sm.Logit(y, X).fit(disp=False)
    probit = sm.Probit(y, X).fit(disp=False)
    results_dict = {"LPM": lpm, "Logit": logit, "Probit": probit}

    # Helper function to compute probability
    def predict_prob(model, x_dict):
        x_vec = np.array([1] + [x_dict.get(var, 0) for var in explanatory_vars])
        linpred = np.dot(model.params.values, x_vec)
        if isinstance(model.model, sm.Logit):
            return 1 / (1 + np.exp(-linpred))
        elif isinstance(model.model, sm.Probit):
            return norm.cdf(linpred)
        else:  # LPM
            return linpred

    # Classic Python output
    if output == "summary":
        print("\n Linear Probability Model (LPM):\n")
        print(lpm.summary())
        print("\n Logit Model:\n")
        print(logit.summary())
        print("\n Probit Model:\n")
        print(probit.summary())
        return results_dict

    # Tabulated output
    elif output == "tabulate":
        rows = []
        variables = X.columns

        for var in variables:
            row = [var]
            # Coefficients
            for model_name, model in results_dict.items():
                params = model.params.get(var, np.nan)
                bse = model.bse.get(var, np.nan)
                pval = model.pvalues.get(var, np.nan)
                row.append(f"{params:.3f}{significance_stars(pval)}")
                row.append(f"({bse:.3f})")

            # PD calculation if this variable is in the list
            if var in prob_vars:
                for name, model in results_dict.items():
                    # scenario 0.25
                    d025 = {v: 0 for v in explanatory_vars}
                    d025[var] = 0.25
                    # scenario 0.50
                    d050 = {v: 0 for v in explanatory_vars}
                    d050[var] = 0.50
                    row.append(f"{predict_prob(model, d025):.3f}")
                    row.append(f"{predict_prob(model, d050):.3f}")
            else:
                for _ in results_dict:
                    row.append("")
                    row.append("")

            rows.append(row)

        # AUC & counts
        auc_results = {}
        for name, model in results_dict.items():
            try:
                y_pred = model.predict(X)
                auc_results[name] = roc_auc_score(y, y_pred)
            except Exception:
                auc_results[name] = np.nan

        n_total = len(y)
        n_def = int((y == 1).sum())
        n_nondef = int((y == 0).sum())

        rows.extend([
            ["# Obs (Total)", f"{n_total}", "", "", "", "", "",
             *["" for _ in range(len(results_dict) * 2)]],
            ["# Default (yd=1)", f"{n_def}", "", "", "", "", "",
             *["" for _ in range(len(results_dict) * 2)]],
            ["# Non-default (yd=0)", f"{n_nondef}", "", "", "", "", "",
             *["" for _ in range(len(results_dict) * 2)]],
            ["AUC",
             f"{auc_results.get('LPM', np.nan):.3f}", "",
             f"{auc_results.get('Logit', np.nan):.3f}", "",
             f"{auc_results.get('Probit', np.nan):.3f}", "",
             *["" for _ in range(len(results_dict) * 2)]]
        ])

        # Headers
        headers = ["Variable"]
        for model in ["LPM", "Logit", "Probit"]:
            headers += [f"{model} β", f"{model} (SE)"]

        # PD headers if prob_vars are given
        for model in ["LPM", "Logit", "Probit"]:
            headers += [f"{model} PD@0.25", f"{model} PD@0.50"]
        if silent == False :
            print("\n Comparative Regression Table (LPM / Logit / Probit)\n")
            print(tabulate(rows, headers=headers, tablefmt="github"))

        return {"models": results_dict, "auc": auc_results}

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


def forecast_default(models_dict, new_df, y_true, explanatory_vars, plot=True,silent=False):
    """
    Use trained models to forecast default probabilities on a new dataset,
    then evaluate their performance (AUC, confusion matrix, ROC curves).

    Returns results_df a Table summarizing AUC and accuracy for each model.
    """
    # Prepare data
    X_new = new_df[explanatory_vars].copy()
    X_new = sm.add_constant(X_new, has_constant="add")

    y_true = pd.Series(y_true).reset_index(drop=True)
    predictions = pd.DataFrame(index=new_df.index)

    auc_results = []
    plt.figure(figsize=(7, 6))
    if silent == False :
        print("\n Model Forecast Evaluation\n")

    for name, model in models_dict.items():
        try:
            # Predict probabilities
            y_pred = model.predict(X_new)
            predictions[f"P_hat_{name}"] = y_pred

            # Compute AUC
            auc = roc_auc_score(y_true, y_pred)

            # ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

            # confusion matrix (threshold 0.5)
            y_pred_class = (y_pred >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)

            auc_results.append([name, len(y_true), auc, accuracy, tp, fp, fn, tn])
            if silent == False:
                print(f" {name}: AUC = {auc:.3f}, Accuracy = {accuracy:.3f}")

        except Exception as e:
            if silent == False:
                print(f" {name} prediction failed: {e}")
            auc_results.append([name, len(y_true), np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

        # Plot ROC curves (optional)
        if plot:
            plt.plot([0, 1], [0, 1], "k--", label="Random chance")
            plt.xlabel("False Positive Rate (1 - Specificity)")
            plt.ylabel("True Positive Rate (Sensitivity)")
            plt.title("ROC Curves – Out-of-Sample Default Prediction")
            plt.legend(loc="lower right")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.show()

    # Tabulated summary
    headers = ["Model", "N Obs", "AUC", "Accuracy", "TP", "FP", "FN", "TN"]
    if silent == False:
        print("\n Model Evaluation Summary\n")
        print(tabulate(auc_results, headers=headers, tablefmt="github", floatfmt=".3f"))

    return pd.DataFrame(auc_results, columns=headers)


