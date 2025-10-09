"""
regression_estimation.py
This module contain all the function relative to estimation and regression analysis
Relevent for Question 11 to 17
"""
from data_statistical_analysis import significance_stars
import statsmodels.api as sm
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy import stats
import pandas as pd
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

        # Main regression coefficients
        for var in variables:
            row = [var]
            for model_name, model in results_dict.items():
                params = model.params.get(var, np.nan)
                bse = model.bse.get(var, np.nan)
                pval = model.pvalues.get(var, np.nan)
                row.append(f"{params:.3f}{significance_stars(pval)}")
                row.append(f"({bse:.3f})")
            rows.append(row)

        # Compute AUC
        auc_results = {}
        for name, model in results_dict.items():
            try:
                y_pred = model.predict(X)
                auc_results[name] = roc_auc_score(y, y_pred)
            except Exception:
                auc_results[name] = np.nan

        # Counts
        n_total = len(y)
        n_def = int((y == 1).sum())
        n_nondef = int((y == 0).sum())

        # Add performance rows to main tabl
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

        # Headers
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


def forecast_default(models_dict, new_df, y_true, explanatory_vars):
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

            print(f" {name}: AUC = {auc:.3f}, Accuracy = {accuracy:.3f}")

        except Exception as e:
            print(f" {name} prediction failed: {e}")
            auc_results.append([name, len(y_true), np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

    # Plot ROC curves
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
    print("\n Model Evaluation Summary\n")
    print(tabulate(auc_results, headers=headers, tablefmt="github", floatfmt=".3f"))

    return pd.DataFrame(auc_results, columns=headers)


def pearson_residuals(models_dict, y_true, X_new):
    """
    Compute and analyze standardized Pearson residuals for each model,
    highlighting default vs non-default observations.
    """
    # Add constant if needed
    if "const" not in X_new.columns:
        X_new = sm.add_constant(X_new, has_constant="add")

    y_true = np.asarray(y_true)
    summary_rows = []

    for name, model in models_dict.items():
        print(f"\n Pearson Residual Analysis — {name}")

        # Predicted probabilities
        y_pred = model.predict(X_new)
        p_hat = np.clip(y_pred, 1e-8, 1 - 1e-8)

        # Pearson residuals
        pearson = (y_true - p_hat) / np.sqrt(p_hat * (1 - p_hat))

        # Jarque–Bera test
        jb_stat, jb_pval = stats.jarque_bera(pearson)
        print(f" Jarque–Bera: stat = {jb_stat:.3f}, p-value = {jb_pval:.3f}")

        # Count large residuals (global)
        n_total = len(pearson)
        n_large2 = int((np.abs(pearson) > 2).sum())
        n_large3 = int((np.abs(pearson) > 3).sum())

        # Count separately for default / non-default
        mask_default = y_true == 1
        mask_nondefault = y_true == 0

        n_def = mask_default.sum()
        n_nondef = mask_nondefault.sum()

        n_large2_def = int((np.abs(pearson[mask_default]) > 2).sum())
        n_large2_nondef = int((np.abs(pearson[mask_nondefault]) > 2).sum())

        n_large3_def = int((np.abs(pearson[mask_default]) > 3).sum())
        n_large3_nondef = int((np.abs(pearson[mask_nondefault]) > 3).sum())

        print(f" |r| > 2 : {n_large2} ({n_large2 / n_total * 100:.1f}%)")
        print(f"    - Defaults (yd=1): {n_large2_def}/{n_def} ({n_large2_def/n_def*100:.1f}%)")
        print(f"    - Non-defaults   : {n_large2_nondef}/{n_nondef} ({n_large2_nondef/n_nondef*100:.1f}%)")
        print(f" |r| > 3 : {n_large3} ({n_large3 / n_total * 100:.1f}%)")
        print(f"    - Defaults (yd=1): {n_large3_def}/{n_def} ({n_large3_def/n_def*100:.1f}%)")
        print(f"    - Non-defaults   : {n_large3_nondef}/{n_nondef} ({n_large3_nondef/n_nondef*100:.1f}%)")

        # Visualization — overlaid histograms
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))

        ax[0].hist(pearson[mask_nondefault], bins=20, alpha=0.6,
                   edgecolor="black", label="Non-default (yd=0)")
        ax[0].hist(pearson[mask_default], bins=20, alpha=0.6,
                   edgecolor="black", label="Default (yd=1)")
        ax[0].axvline(0, color="black", linestyle="--")
        ax[0].axvline(2, color="red", linestyle="--")
        ax[0].axvline(-2, color="red", linestyle="--")
        ax[0].set_title(f"{name}: Pearson residuals by group")
        ax[0].legend()

        stats.probplot(pearson, dist="norm", plot=ax[1])
        ax[1].set_title(f"{name}: Q–Q plot")

        plt.tight_layout()
        plt.show()

        # Summary table row
        summary_rows.append([
            name, n_total, round(jb_stat, 3), round(jb_pval, 3),
            n_large2, f"{(n_large2 / n_total) * 100:.1f}%",
            n_large2_def, n_large2_nondef,
            n_large3, f"{(n_large3 / n_total) * 100:.1f}%",
            n_large3_def, n_large3_nondef
        ])

    # Print summary table
    headers = [
        "Model", "N Obs", "JB stat", "JB p-value",
        "|r|>2 (#)", "|r|>2 (%)", "|r|>2 (def)", "|r|>2 (non-def)",
        "|r|>3 (#)", "|r|>3 (%)", "|r|>3 (def)", "|r|>3 (non-def)"
    ]
    print("\n Pearson Residual Summary by Group\n")
    print(tabulate(summary_rows, headers=headers, tablefmt="github"))
