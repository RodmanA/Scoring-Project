"""
model_evaluation.py
This module contain all the function relative to evaluating the models
Relevent for Question 15 to 18
"""
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
import pandas as pd


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


def optimal_threshold(y_true, p_hat, LGD=0.6, margin=0.05, EAD=None,
                                grid=None, return_table=False):
    """
    Find the threshold s that minimizes expected loss on data.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_hat)
    n = len(y)
    if EAD is None: EAD = np.ones(n)
    EAD = np.asarray(EAD)

    # allow per-observation LGD/margin
    LGD_vec = np.broadcast_to(np.asarray(LGD), (n,))
    M_vec   = np.broadcast_to(np.asarray(margin), (n,))

    if grid is None:
        grid = np.linspace(0, 1, 1001)

    losses, tpr, fpr = [], [], []
    for s in grid:
        # decisions: lend if p < s; reject otherwise
        lend = p < s
        # Type I error: y=1 & lend -> lose LGD*EAD
        type1 = (y == 1) & lend
        # Type II error: y=0 & reject -> lose margin*EAD
        type2 = (y == 0) & (~lend)

        loss = (LGD_vec[type1] * EAD[type1]).sum() + (M_vec[type2] * EAD[type2]).sum()
        losses.append(loss)

        # Diagnostics (for plotting/ROC if you want)
        tp = ((y == 1) & (~lend)).sum()  # predicted default & is default
        fn = ((y == 1) & lend).sum()
        fp = ((y == 0) & (~lend)).sum()
        tn = ((y == 0) & lend).sum()
        tpr.append(tp / (tp + fn) if (tp + fn) else 0.0)
        fpr.append(fp / (fp + tn) if (fp + tn) else 0.0)

    idx = int(np.argmin(losses))
    s_emp = float(grid[idx])

    out = {
        "s_closed": float(margin / (LGD + margin)) if np.isscalar(LGD) and np.isscalar(margin) else None,
        "s_empirical": s_emp,
        "min_loss": float(losses[idx]),
        "grid": grid,
        "losses": np.array(losses),
        "tpr": np.array(tpr),
        "fpr": np.array(fpr),
    }

    if return_table:
        return out, pd.DataFrame({"threshold": grid, "expected_loss": losses,
                                  "TPR": tpr, "FPR": fpr})
    return out

def w_optimal_threshold_models(models_dict, X_new, y_true, LGD=0.6, margin=0.05, plot_roc=True):
    """
    Wrapper function to optimal_threshold()
    Compute optimal lending threshold s* for each model based on expected loss minimization.
    Additionally, draw the iso-loss tangent line through the optimal empirical threshold.
    """
    if "const" not in X_new.columns:
        X_new = sm.add_constant(X_new, has_constant="add")

    results = []

    if plot_roc:
        plt.figure(figsize=(8, 6))

    for name, model in models_dict.items():
        try:
            #  Predict PD 
            p_hat = model.predict(X_new)
            out = optimal_threshold(y_true, p_hat, LGD=LGD, margin=margin)

            #  Store summary 
            results.append([
                name,
                len(y_true),
                round(out["s_closed"], 4) if out["s_closed"] is not None else "N/A",
                round(out["s_empirical"], 4),
                round(out["min_loss"], 4)
            ])

            print(
                f"{name:8s} → s_closed={out['s_closed']:.4f} | s_empirical={out['s_empirical']:.4f} | min_loss={out['min_loss']:.4f}"
            )

            #  ROC curve plot 
            if plot_roc:
                plt.plot(out["fpr"], out["tpr"], label=f"{name}")

                # Find closest ROC point to s_empirical
                idx_emp = (np.abs(out["grid"] - out["s_empirical"])).argmin()
                fpr_star = out["fpr"][idx_emp]
                tpr_star = out["tpr"][idx_emp]

                # Mark the empirical threshold point
                plt.scatter(fpr_star, tpr_star, marker="o", s=80, color="red", edgecolors="black",
                            label=f"{name} s* empirical")

                # Mark the closed form threshold point if it exists
                if out["s_closed"] is not None:
                    idx_closed = (np.abs(out["grid"] - out["s_closed"])).argmin()
                    plt.scatter(out["fpr"][idx_closed], out["tpr"][idx_closed], marker="X", s=80, color="green",
                                edgecolors="black", label=f"{name} s* closed")

                #  Draw tangent iso-loss line 
                slope = margin / LGD
                intercept = tpr_star - slope * fpr_star

                fpr_line = np.linspace(0, 1, 200)
                tpr_line = intercept + slope * fpr_line
                tpr_line = np.clip(tpr_line, 0, 1)

                plt.plot(fpr_line, tpr_line, linestyle="--", color="red", alpha=0.6,
                         label=f"{name} iso-loss line")

        except Exception as e:
            print(f"{name}: Threshold computation failed — {e}")
            results.append([name, len(y_true), np.nan, np.nan, np.nan])

    #  Finalize ROC plot 
    if plot_roc:
        plt.plot([0, 1], [0, 1], "k--", label="Random chance")
        plt.xlabel("False Positive Rate (1 - Specificity)")
        plt.ylabel("True Positive Rate (Sensitivity)")
        plt.title("ROC Curve with Tangent Iso-Loss Line at s*")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

    #  Summary table 
    headers = ["Model", "N Obs", "s_closed", "s_empirical", "Min Expected Loss"]
    print("\n Optimal Threshold Summary\n")
    print(tabulate(results, headers=headers, tablefmt="github", floatfmt=".4f"))

    return pd.DataFrame(results, columns=headers)
