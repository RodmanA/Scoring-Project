"""
data_statistical_analysis.py
This module contain all the function relative to data statistical analysis
Relevent for Question 3, Question 6
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm,stats
from tabulate import tabulate


def significance_stars(p):
    """Return significance stars based on p-value."""
    if pd.isna(p):
        return ""
    elif p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.1:
        return "*"
    else:
        return ""

def plot_ratio_distributions(df, target="yd"):
    """
    Plot, for each numeric ratio column (excluding target):
      - Histogram of all data (combined groups)
      - KDE curve for non-default (yd=0)
      - KDE curve for default (yd=1)
      - Dashed normal distribution fits for both groups
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    ratio_cols = df.drop(columns=[target], errors="ignore").select_dtypes(include=[np.number]).columns.tolist()
    if not ratio_cols:
        raise ValueError("No numeric ratio columns found to plot.")

    for col in ratio_cols:
        data_all = df[col].dropna()
        data_0 = df.loc[df[target] == 0, col].dropna()
        data_1 = df.loc[df[target] == 1, col].dropna()

        if data_all.empty:
            continue

        plt.figure(figsize=(9, 5.5))

        # 1 Histogram for the combined distribution
        plt.hist(
            data_all,
            bins="auto",
            color="gray",
            alpha=0.35,
            density=True,
            label="All firms (histogram)"
        )

        # Prepare x-grid for continuous curves
        x_min, x_max = np.percentile(data_all, [0.5, 99.5])
        x_grid = np.linspace(x_min, x_max, 512)

        # 2 KDE curves for non-default and default
        if len(data_0) > 1:
            kde0 = gaussian_kde(data_0)
            plt.plot(x_grid, kde0(x_grid), color="blue", lw=2, label="Non-default (KDE)")
        if len(data_1) > 1:
            kde1 = gaussian_kde(data_1)
            plt.plot(x_grid, kde1(x_grid), color="red", lw=2, label="Default (KDE)")

        # 3 Dashed normal distribution fits
        if len(data_0) > 1:
            mu0, std0 = np.mean(data_0), np.std(data_0)
            plt.plot(
                x_grid, norm.pdf(x_grid, mu0, std0),
                color="blue", linestyle="--", lw=1.8,
                label=f"Non-default Normal Fit (μ={mu0:.2f}, σ={std0:.2f})"
            )
        if len(data_1) > 1:
            mu1, std1 = np.mean(data_1), np.std(data_1)
            plt.plot(
                x_grid, norm.pdf(x_grid, mu1, std1),
                color="red", linestyle="--", lw=1.8,
                label=f"Default Normal Fit (μ={mu1:.2f}, σ={std1:.2f})"
            )

        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

def statistical_tests(df, target="yd"):
    """
    Produce four summary tables:
    1) Skewness, Kurtosis, JB, and T-tests per group
    2) Kolmogorov–Smirnov tests
    3) Correlation with dependent variable
    4) Ranking by |t-stat| and |corr|
    """
    cols = [c for c in df.columns if c != target]
    table1, table2, table3 = [], [], []

    # - Compute all stats -
    for col in cols:
        data = df[[col, target]].dropna()

        # Split into groups
        defaulted = data.loc[data[target] == 1, col]
        non_defaulted = data.loc[data[target] == 0, col]
        N_def, N_nondef = len(defaulted), len(non_defaulted)

        if N_def == 0 or N_nondef == 0:
            continue

        #  Skewness & Kurtosis 
        skew_def, kurt_def = stats.skew(defaulted), stats.kurtosis(defaulted)
        skew_nondef, kurt_nondef = stats.skew(non_defaulted), stats.kurtosis(non_defaulted)

        #  Jarque–Bera 
        jb_def_stat, jb_def_pval = stats.jarque_bera(defaulted)
        jb_nondef_stat, jb_nondef_pval = stats.jarque_bera(non_defaulted)

        #  T-test 
        t_stat, t_pval = stats.ttest_ind(defaulted, non_defaulted, equal_var=False, nan_policy="omit")

        #  Kolmogorov–Smirnov 
        ks_stat, ks_pval = stats.ks_2samp(defaulted, non_defaulted)

        #  Pearson correlation 
        corr, corr_pval = stats.pearsonr(data[col], data[target])

        # - Table 1 -
        table1.append([
            col, N_nondef, N_def,
            round(skew_nondef, 3), round(kurt_nondef, 3),
            f"{round(jb_nondef_stat, 3)} ({round(jb_nondef_pval, 3)}{significance_stars(jb_nondef_pval)})",
            round(skew_def, 3), round(kurt_def, 3),
            f"{round(jb_def_stat, 3)} ({round(jb_def_pval, 3)}{significance_stars(jb_def_pval)})",
            f"{round(t_stat, 3)} ({round(t_pval, 3)}{significance_stars(t_pval)})"
        ])

        # - Table 2 -
        table2.append([
            col, N_nondef, N_def,
            f"{round(ks_stat, 3)} ({round(ks_pval, 3)}{significance_stars(ks_pval)})"
        ])

        # - Table 3 -
        table3.append([
            col, len(data),
            f"{round(corr, 3)} ({round(corr_pval, 3)}{significance_stars(corr_pval)})"
        ])

    # -- Print Tables 1–3 --
    headers1 = [
        "Variable", "N_nondef", "N_def",
        "Skew_0", "Kurt_0", "JB_0(stat,p)",
        "Skew_1", "Kurt_1", "JB_1(stat,p)",
        "T-test(stat,p)"
    ]
    headers2 = ["Variable", "N_nondef", "N_def", "KS-test(stat,p)"]
    headers3 = ["Variable", "N_total", "Correlation with yd (r,p)"]

    print("\n Question 3 : Normality Test and correlation analysis\n")
    print("\n Table 1: Normality and T-tests per Group\n")
    print(tabulate(table1, headers=headers1, tablefmt="github"))

    print("\n Table 2: Kolmogorov–Smirnov Tests\n")
    print(tabulate(table2, headers=headers2, tablefmt="github"))

    print("\n Table 3: Correlation with yd\n")
    print(tabulate(table3, headers=headers3, tablefmt="github"))

    # ===== TABLE 4: RANKING =====
    df1 = pd.DataFrame(table1, columns=headers1)
    df3 = pd.DataFrame(table3, columns=headers3)

    # Extract numeric values
    df1["t_stat"] = df1["T-test(stat,p)"].str.extract(r"([\-0-9\.]+)").astype(float)
    df3["corr"] = df3["Correlation with yd (r,p)"].str.extract(r"([\-0-9\.]+)").astype(float)

    df1["abs_t"] = df1["t_stat"].abs()
    df3["abs_corr"] = df3["corr"].abs()

    # Rank both
    df1["Rank_t"] = df1["abs_t"].rank(ascending=False, method="min").astype(int)
    df3["Rank_corr"] = df3["abs_corr"].rank(ascending=False, method="min").astype(int)

    # Merge rankings
    df_rank = pd.merge(
        df1[["Variable", "t_stat", "abs_t", "Rank_t"]],
        df3[["Variable", "corr", "abs_corr", "Rank_corr"]],
        on="Variable",
        how="inner"
    ).sort_values("Rank_t")

    # - Table 4 Output -
    print("\n  Question 5/6 : t-stat/Correlation ranking")
    print("\n Table 4: Ranking by |t-stat| and |Correlation|\n")
    print(tabulate(df_rank.round(4),
                   headers=df_rank.columns,
                   tablefmt="github"))

    return table1, table2, table3, df_rank

def plot_lower_correlation(df, target="yd", threshold=0.8, figsize=(10, 8), cmap="coolwarm"):
    """
    Plot a semi-lower triangular correlation matrix for all explanatory variables,
    and highlight variable pairs with |correlation| > threshold.
    """
    # Select explanatory variables only
    explanatory = df.drop(columns=[target], errors="ignore")

    # Compute correlation matrix
    corr = explanatory.corr(method="pearson")

    # Mask the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Plot lower triangular heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1,
                annot=True, fmt=".2f", linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Semi-Lower Correlation Heatmap of Explanatory Variables", fontsize=14, pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Find and print highly correlated pairs
    corr_unstacked = corr.abs().unstack()
    corr_unstacked = corr_unstacked[corr_unstacked < 1]  # remove self-correlations
    high_corr = corr_unstacked[corr_unstacked >= threshold].sort_values(ascending=False)

    if not high_corr.empty:
        print("\n Question 8 : Bivariate correlation\n")
        print(f"\n Highly correlated variable pairs (|r| ≥ {threshold}):\n")
        high_corr_pairs = pd.DataFrame(high_corr).reset_index()
        high_corr_pairs.columns = ["Variable_1", "Variable_2", "|r|"]
        print(high_corr_pairs.to_string(index=False))
    else:
        print(f"\n No variable pairs with |r| ≥ {threshold}.")

    return corr, high_corr_pairs if not high_corr.empty else pd.DataFrame(columns=["Variable_1", "Variable_2", "|r|"])


def plot_top_corr(df, target_col, ranking_table, rank_col="Rank_corr", top_n=5):
    """
    Plots a pairplot of the top N explanatory variables based on a ranking table.

    Parameters:
    --
    df : pd.DataFrame
        Dataframe containing explanatory variables and target column.
    target_col : str
        Name of the target column (e.g., 'yd').
    ranking_table : pd.DataFrame
        Dataframe with variable rankings (must have columns 'Variable' and rank_col).
    rank_col : str, default="Rank_t"
        Column to use for ranking (e.g., 'Rank_t' or 'Rank_corr').
    top_n : int, default=5
        Number of top variables to plot.
    """

    # Select top variables
    top_vars = ranking_table.sort_values(by=rank_col).head(top_n)["Variable"].tolist()

    # Subset dataframe
    plot_df = df[top_vars + [target_col]]
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna()
    # Create pairplot with regression lines, hue by target
    sns.pairplot(
        data=plot_df,
        vars=top_vars,
        hue=target_col,
        palette={0: "green", 1: "red"},
        kind="reg",
        plot_kws={'scatter_kws': {'alpha': 0.7}}
    )

    plt.show()