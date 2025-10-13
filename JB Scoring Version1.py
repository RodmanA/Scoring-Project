#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import Python libraries

# Limit futureWarning comments
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

# General libraries
import pandas as pd
import numpy as np

# Graphs libraries
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, rgb2hex
import seaborn as sns
from seaborn import pairplot
from seaborn import heatmap
import matplotlib.patches as mpatches

# Statistical libraries, norm is for normal distribution case1: stats.skew() case2: skew()
# Levene is test of equality of variance, sm is for OLS regression
from scipy import stats
import scipy.stats as stats
from scipy.stats import skew, kurtosis, norm, levene, ttest_ind
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit, Probit

# Machine Learning classification libraries (Scikit), Roc_auc_score is area under the Roc curve.
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, fbeta_score, f1_score, 
    roc_curve, auc, roc_auc_score, classification_report, confusion_matrix
)


# In[49]:


# Define variable trying to lolike label using SAS
variables = {
    'yd': 'Financial Difficulty', 'tdta': 'Debt/Assets', 'reta': 'Retained Earnings',
    'opita': 'Income/Assets', 'ebita': 'Pre-Tax Earnings/Assets', 'lsls': 'Log Sales',
    'lta': 'Log Assets', 'gempl': 'Employment Growth', 'invsls': 'Inventory/Sales',
    'nwcta': 'Net Working Capital/Assets', 'cacl': 'Current Assets/Liabilities', 
    'qacl': 'Quick Assets/Liabilities', 'fata': 'Fixed Assets/Total Assets', 
    'ltdta': 'Long-Term Debt/Total Assets', 'mveltd': 'Market Value Equity/Long-Term Debt'
}
# Load the dataset
df = pd.read_csv('defaut2000.csv', sep=';')
#Replace the commas (French CSV file) by dots
df= df.replace(",",".", regex=True)
#Variables as number and not string
df = df.apply(pd.to_numeric, errors='coerce')
# Alternative code: df = df.applymap(lambda x: float(str(x).replace(',', '.')) if isinstance(x, str) else x)
# Check output for first N observations
print(df.head(5))


# In[50]:


# Numbers of rows (observations) and columns (variables)
df.shape


# In[51]:


# List of variables and their type
df.dtypes


# In[52]:


# Univariate descriptive statistics for first eight variables in columns
df.describe().iloc[:,:8]


# In[53]:


# Replace the minima -99.99 placeholders by np.nan "not a number" correspond to . in other softwares
# Replace the minima -99.99 by np.nan "not a number" correspond to . in other softwares
df[['fata', "ltdta"]]= df[['fata', 'ltdta']].replace(-99.99, np.nan)
# Check the new minima for the two variables 
df[['fata', 'ltdta']].describe()
# Check the NaN observation for the two variables
print(df[['fata', 'ltdta']].head(20))


# In[54]:


df[['fata', 'ltdta']].describe()


# In[55]:


# Sort the dataset by Financial Difficulty and retained earnings/assets and reset index
df_sorted = df.sort_values(by=['yd', 'reta']).reset_index(drop=True)
# Check if it has been sorted correctly on the first N observations
print(df_sorted[['yd','reta', 'tdta']].head(5))


# In[56]:


# Check min and max and new number of observations below 181
df_sorted[['fata', 'ltdta']].describe()


# In[57]:


# Separate target (y) versus features (X) for train-test samples
# Separate dependent variable (y) versus explanatory variables (X) for estimation vs forecast samples
y_sorted = df_sorted['yd'].copy()
print(y_sorted.head(4))
X_sorted = df_sorted.drop(columns=['yd']).copy()
print(X_sorted.head(4))


# In[58]:


# Split into odd rows for training and even rows for validation
# Alternative code, create a dummy for odd rows: df_sorted['dumEV'] = (df_sorted.index % 2 != 0).astype(int)
# df_estimation= df_sorted[df_sorted['dumEV']==0]
X_train = X_sorted.iloc[1::2].copy()
X_test = X_sorted.iloc[::2].copy()
y_train = y_sorted.iloc[1::2].copy()
y_test = y_sorted.iloc[::2].copy()
print(X_sorted[['reta', 'tdta']].head(6))
print(X_train[['reta', 'tdta']].head(3))
print(X_test[['reta', 'tdta']].head(3))


# In[59]:


y_X_train = pd.concat([y_train, X_train], axis=1)
print(y_X_train[['yd', 'tdta']].head(3))


# In[60]:


# Separate train set into defaulting and non-defaulting groups
X_train_safe = X_train[y_train == 0]
X_train_default = X_train[y_train == 1]
print(X_train_safe[['reta', 'tdta']].head(3))
print(X_train_default[['reta', 'tdta']].head(3))


# In[61]:


X_train_default[['reta', 'tdta']].describe()


# In[62]:


X_train_safe[['reta', 'tdta']].describe()


# In[63]:


"""
STEP 2A: Bivariate relations on the train (estimation) sample
1) difference of means of univariate distributions by group yd for each variable TDTA or RETA
start of the idea of linear discriminant analysis, instead of uninformative bivariate of TDTA on TDTA
2) linear probability model: yd = a + b TDTA + e
3) analysis of variance: TDTA = a + b yd + e
4) highly correlated accounting regressors for the linear probability model TDTA and RETA
All these relations in graphs are summarized by a single number:
The simple correlation coefficient and its t statistics f(r, n_obs)
This t-statistics will be the same for t-test difference of means of normals with equal standard errors
and for the regression with homoskedastic disturbances for the standard error 
Missing: global histogram of yd instead of kde by yd, 
Missing: title above, and more numbers on the horizontal axis
Missing: confidence interval for the regression line
Missing: a thin grid for bivariate graphs
"""

# New dataframe
new_df = y_X_train.copy()
new_df["target"] = y_train
vars = ["yd", "tdta", "reta"]

# Check for zero variance in the variables
for var in vars:
    if new_df[var].var() == 0:
        print(f"Warning: Variable {var} has zero variance. KDE may fail.")

# Set up the PairGrid
g = sns.PairGrid(new_df, vars=vars, hue="target", palette={0: "green", 1: "red"})

# Map KDE to diagonal, split by hue, with filled areas
g.map_diag(sns.kdeplot, hue_order=[0, 1], common_norm=False, warn_singular=False, fill=True)

# Custom function for scatter plot and single regression line
def plot_scatter_reg(x, y, **kwargs):
    # Extract data from kwargs
    data = new_df
    # Scatter plot with hue
    sns.scatterplot(x=data[x.name], y=data[y.name], hue=data["target"], palette={0: "green", 1: "red"})
    # Single regression line (ignoring hue)
    sns.regplot(x=data[x.name], y=data[y.name], color="blue", scatter=False, ci=None, line_kws={"linewidth": 2})
# Map the custom function to off-diagonal
g.map_offdiag(plot_scatter_reg)
# Add legend
g.add_legend()
# Show plot
plt.show()


# In[64]:


# Define the column to plot (replace 'tdta' with any of the 14 numeric columns)
col = 'tdta'  # Example column name; can be any of your 14 numeric columns

# Assume y_X_train is your DataFrame with 'yd' as the first column and 14 numeric variables

# Calculate shared x-axis limits for the selected column
x_min = y_X_train[col].dropna().min()
x_max = y_X_train[col].dropna().max()

# Group by 'yd' to mimic SAS BY/CLASS
groups = y_X_train.groupby('yd')

# Calculate shared y-axis limit for density (to ensure same height for KDE/histogram)
max_density = 0
group_stats = {}
x = np.linspace(x_min, x_max, 100)
for yd, group in groups:
    data = group[col].dropna()
    mean = data.mean()
    std = data.std()
    group_stats[yd] = {'mean': mean, 'std': std}
    # Temporary KDE to get max density
    kde = sns.kdeplot(data)
    max_density = max(max_density, kde.get_lines()[0].get_data()[1].max())
    plt.close()  # Close temporary plot
y_max = max_density * 1.1  # Add 10% buffer

# Create a figure with 4 subplots: 2 univariate (one per yd group), 2 boxplots
fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True, gridspec_kw={'height_ratios': [1, 1, 0.5, 0.5]})

# Labels for yd groups (assuming yd is 0 or 1; adjust if different)
yd_labels = {0: 'Non-Default (yd=0)', 1: 'Default (yd=1)'}
colors = {0: 'green', 1: 'red'}
dark_colors = {0: 'darkgreen', 1: 'darkred'}

# Plot univariate distributions and boxplots for each yd group
for i, (yd, group) in enumerate(groups):
    data = group[col].dropna()
    label = yd_labels.get(yd, f'yd={yd}')
    color = colors.get(yd, 'blue')  # Default color if yd value is unexpected
    dark_color = dark_colors.get(yd, 'darkblue')
    
    # Univariate plot (histogram + KDE + normal) on axes[i]
    sns.histplot(data, label=f'{label} Hist', color=color, alpha=0.4, stat='density', ax=axes[i])
    sns.kdeplot(data, label=f'{label} KDE', color=color, linewidth=2, ax=axes[i])
    mean = group_stats[yd]['mean']
    std = group_stats[yd]['std']
    axes[i].plot(x, norm.pdf(x, mean, std), color=dark_color, linestyle='--', label=f'{label} Normal')
    axes[i].set_title(f'Distribution of {col} for {label}')
    axes[i].set_ylabel('Density')
    axes[i].set_ylim(0, y_max)  # Shared y-axis limit for consistent height
    axes[i].legend()

# Boxplots for each yd group (separate subplots)
for i, (yd, group) in enumerate(groups, start=2):  # Start at axes[2] for boxplots
    data = group[col].dropna()
    label = yd_labels.get(yd, f'yd={yd}')
    color = colors.get(yd, 'blue')
    sns.boxplot(x=data, orient='h', color=color, ax=axes[i],
                showmeans=True, meanprops={'marker': 'D', 'markersize': 8, 'markeredgecolor': 'black'})
    axes[i].set_yticks([])
    axes[i].set_title(f'Boxplot for {label}', pad=10)
    axes[i].set_ylabel('')

# Set x-label for the bottom plot
axes[3].set_xlabel(col)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()


# In[65]:


# Select the first column, assuming it is 'tdta'
col = 'tdta'

plt.figure(figsize=(10, 4))

# Calculate mean and standard deviation for each group
mean_safe = X_train_safe[col].dropna().mean()
std_safe = X_train_safe[col].dropna().std()
mean_default = X_train_default[col].dropna().mean()
std_default = X_train_default[col].dropna().std()

# Plot histograms for both groups
sns.histplot(X_train_safe[col].dropna(), label='Non-Default Hist', color='green', alpha=0.4, stat='density')
sns.histplot(X_train_default[col].dropna(), label='Default Hist', color='red', alpha=0.4, stat='density')

# Plot KDE for both groups with matching colors
sns.kdeplot(X_train_safe[col].dropna(), label='Non-Default KDE', color='green', linewidth=2)
sns.kdeplot(X_train_default[col].dropna(), label='Default KDE', color='red', linewidth=2)

# Generate points for normal distribution curves
x = np.linspace(
    min(X_train_safe[col].dropna().min(), X_train_default[col].dropna().min()),
    max(X_train_safe[col].dropna().max(), X_train_default[col].dropna().max()),
    100
)
plt.plot(x, norm.pdf(x, mean_safe, std_safe), color='darkgreen', linestyle='--', label='Non-Default Normal')
plt.plot(x, norm.pdf(x, mean_default, std_default), color='darkred', linestyle='--', label='Default Normal')

# Customize the plot
plt.title(f'Distribution of {col} for Default and Non-Default Groups')
plt.xlabel(col)
plt.ylabel('Density')
plt.legend()
plt.show()


# In[66]:


for col in X_train_default.columns:
    plt.figure(figsize=(10, 4))
    
    # Calculate mean and standard deviation for each group
    mean_safe = X_train_safe[col].dropna().mean()
    std_safe = X_train_safe[col].dropna().std()
    mean_default = X_train_default[col].dropna().mean()
    std_default = X_train_default[col].dropna().std()
    
    # Plot histograms for both groups
    sns.histplot(X_train_safe[col].dropna(), label='Non-Default Hist', color='green', alpha=0.4, stat='density')
    sns.histplot(X_train_default[col].dropna(), label='Default Hist', color='red', alpha=0.4, stat='density')
    
    # Plot KDE for both groups with matching colors
    sns.kdeplot(X_train_safe[col].dropna(), label='Non-Default KDE', color='green', linewidth=2)
    sns.kdeplot(X_train_default[col].dropna(), label='Default KDE', color='red', linewidth=2)
    
    # Generate points for normal distribution curves
    x = np.linspace(
        min(X_train_safe[col].dropna().min(), X_train_default[col].dropna().min()),
        max(X_train_safe[col].dropna().max(), X_train_default[col].dropna().max()),
        100
    )
    plt.plot(x, norm.pdf(x, mean_safe, std_safe), color='green', linestyle='--', label='Non-Default Normal')
    plt.plot(x, norm.pdf(x, mean_default, std_default), color='red', linestyle='--', label='Default Normal')
    
    # Customize the plot
    plt.title(f'Distribution of {col} for Default and Non-Default Groups')
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.legend()
    plt.show()


# In[67]:


# Pre-tests of normality Jarque-Bera X2 statistic: Joint H0: Skewness=0 AND Kurtosis-3=0

# Get numeric columns, excluding 'yd'
numeric_cols = y_X_train.select_dtypes(include='number').columns.drop('yd')

# List to store statistics
stats_list = []

# Group by 'yd' (dummy variable for group)
for group_name, group_data in y_X_train.groupby('yd'):
    group_label = 'NonDefault' if group_name == 0 else 'Default'
    for col in numeric_cols:
        data = group_data[col].dropna()
        n = len(data)
        if n > 0:  # Skip if no data after dropping NaNs
            skew = stats.skew(data)
            kurt = stats.kurtosis(data)
            jb_stat, jb_p = stats.jarque_bera(data)
            # Add star if p-value < 0.05
            p_display = f"{jb_p:.3f}*" if jb_p < 0.05 else f"{jb_p:.3f}"
            stats_list.append({
                'Group': group_label,
                'Variable': col,
                'Obs.': n,
                'Skewness': f"{skew:.3f}",
                'Kurtosis-3': f"{kurt:.3f}",
                'JB Stat.': f"{jb_stat:.3f}",
                'P-value': p_display
            })

# Create DataFrame from results
stats_df = pd.DataFrame(stats_list)

# Print the results
print('Jarque-Bera X2 statistic H0: Skewness=0 and Kurtosis-3=0')
print(stats_df.to_string(index=False))


# In[68]:


# Get the list of variable names from X_train
variables = X_train.columns.tolist()

# List to hold results
results = []

for var in variables:
    # Extract groups, dropping any NaNs
    group0 = X_train[y_X_train['yd'] == 0][var].dropna()
    group1 = X_train[y_X_train['yd'] == 1][var].dropna()
    
    n0 = len(group0)
    n1 = len(group1)
    
    if n0 < 2 or n1 < 2:
        sd0 = np.nan
        sd1 = np.nan
        stat = np.nan
        p = np.nan
        p_display = np.nan
    else:
        sd0 = np.std(group0, ddof=1)  # Sample standard deviation
        sd1 = np.std(group1, ddof=1)
        stat, p = levene(group0, group1)  # Levene's test for equal variances
        p_display = f"{round(p, 3)}*" if p < 0.05 else round(p, 3)
    
    # Round to 3 decimal places
    results.append({
        'Variable': var,
        'n0': n0,
        'sd0': round(sd0, 3) if not np.isnan(sd0) else np.nan,
        'n1': n1,
        'sd1': round(sd1, 3) if not np.isnan(sd1) else np.nan,
        'statistic': round(stat, 3) if not np.isnan(stat) else np.nan,
        'p_value': p_display
    })

# Create DataFrame for results
df_results = pd.DataFrame(results)

# Display the table
print(df_results.to_string(index=False))


# In[69]:


# t-test with equal versus unequal variance
# Get the list of variable names from X_train
variables = X_train.columns.tolist()
# List to hold results
results = []

for var in variables:
    # Extract groups, dropping any NaNs
    group0 = X_train[y_X_train['yd'] == 0][var].dropna()
    group1 = X_train[y_X_train['yd'] == 1][var].dropna()
    
    n0 = len(group0)
    n1 = len(group1)
    
    if n0 < 2 or n1 < 2:
        mean0 = np.nan
        mean_diff = np.nan
        t_stat_equal = np.nan
        p_equal = np.nan
        t_stat_unequal = np.nan
        p_unequal = np.nan
        p_equal_display = np.nan
        p_unequal_display = np.nan
    else:
        # Calculate means
        mean0 = np.mean(group0)
        mean1 = np.mean(group1)
        mean_diff = mean1 - mean0
        
        # T-test assuming equal variances
        t_stat_equal, p_equal = ttest_ind(group0, group1, equal_var=True)
        p_equal_display = f"{p_equal:.3f}*" if p_equal < 0.05 else f"{p_equal:.3f}"
        
        # T-test assuming unequal variances (Welch's t-test)
        t_stat_unequal, p_unequal = ttest_ind(group0, group1, equal_var=False)
        p_unequal_display = f"{p_unequal:.3f}*" if p_unequal < 0.05 else f"{p_unequal:.3f}"
    
    # Round to 3 decimal places
    results.append({
        'Variable': var,
        'n0': n0,
        'm0': round(mean0, 3) if not np.isnan(mean0) else np.nan,
        'n1': n1,
        'm1-m0': round(mean_diff, 3) if not np.isnan(mean_diff) else np.nan,
        't_stat': round(t_stat_equal, 3) if not np.isnan(t_stat_equal) else np.nan,
        'p_value': p_equal_display,
        't_stat_dif': round(t_stat_unequal, 3) if not np.isnan(t_stat_unequal) else np.nan,
        'p_value_dif': p_unequal_display
    })

# Create DataFrame for results
df_results = pd.DataFrame(results)

# Display the table
print('T-test for equality of means (yd=0 vs yd=1)')
print(df_results.to_string(index=False))


# In[70]:


# Extract the variables
y = y_X_train['yd']
X = y_X_train['tdta']

# Add a constant to the independent variable
X = sm.add_constant(X)

# Fit the linear probability model
model = sm.OLS(y, X).fit()

# Print the summary of the model
print(model.summary())


# In[71]:


# Create a scatter plot with regression line
plt.figure(figsize=(12, 7))

# Get the predicted values for each observation
y_pred = model.predict(X)

# Separate the data points by yd value
yd_0_indices = y_X_train['yd'] == 0
yd_1_indices = y_X_train['yd'] == 1

# Plot points where yd=0 in green
plt.scatter(y_X_train.loc[yd_0_indices, 'tdta'], 
            y_X_train.loc[yd_0_indices, 'yd'], 
            color='green', 
            alpha=0.7, 
            label='Actual yd=0')

# Plot points where yd=1 in red
plt.scatter(y_X_train.loc[yd_1_indices, 'tdta'], 
            y_X_train.loc[yd_1_indices, 'yd'], 
            color='red', 
            alpha=0.7, 
            label='Actual yd=1')

# Plot the forecasted values with lighter versions of the same colors
plt.scatter(y_X_train.loc[yd_0_indices, 'tdta'], 
            y_pred[yd_0_indices], 
            color='green', 
            marker='x', 
            label='Forecast for yd=0')

plt.scatter(y_X_train.loc[yd_1_indices, 'tdta'], 
            y_pred[yd_1_indices], 
            color='red', 
            marker='x', 
            label='Forecast for yd=1')

# Generate points for the regression line
x_range = np.linspace(min(y_X_train['tdta']), max(y_X_train['tdta']), 100)
x_range_with_const = sm.add_constant(x_range)
y_pred_line = model.predict(x_range_with_const)

# Plot the regression line in blue
plt.plot(x_range, y_pred_line, 'b-', linewidth=2, label='Regression line')

# Add labels and title
plt.xlabel('tdta', fontsize=12)
plt.ylabel('yd (Binary)', fontsize=12)
plt.title('Linear Probability Model using the train sample', fontsize=14)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

# Add horizontal lines at 0 and 1 to show the binary nature
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

# Show the plot
plt.tight_layout()
plt.show()


# In[72]:


# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Get predicted values
y_pred = model.predict(X)
residuals = y - y_pred

# Separate data by yd value
yd_0_indices = y_X_train['yd'] == 0
yd_1_indices = y_X_train['yd'] == 1

# SUBPLOT 1: Scatter plot with regression line
ax1.scatter(y_X_train.loc[yd_0_indices, 'tdta'], 
           y_X_train.loc[yd_0_indices, 'yd'], 
           color='green', 
           alpha=0.7, 
           label='Actual yd=0')
ax1.scatter(y_X_train.loc[yd_1_indices, 'tdta'], 
           y_X_train.loc[yd_1_indices, 'yd'], 
           color='red', 
           alpha=0.7, 
           label='Actual yd=1')
ax1.scatter(y_X_train.loc[yd_0_indices, 'tdta'], 
           y_pred[yd_0_indices], 
           color='lightgreen', 
           marker='x', 
           label='Forecast for yd=0')
ax1.scatter(y_X_train.loc[yd_1_indices, 'tdta'], 
           y_pred[yd_1_indices], 
           color='lightcoral', 
           marker='x', 
           label='Forecast for yd=1')
x_range = np.linspace(min(y_X_train['tdta']), max(y_X_train['tdta']), 100)
x_range_with_const = sm.add_constant(x_range)
y_pred_line = model.predict(x_range_with_const)
ax1.plot(x_range, y_pred_line, color='lightblue', linewidth=2.5, label='Regression line')
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('tdta', fontsize=12)
ax1.set_ylabel('yd (Binary)', fontsize=12)
ax1.set_title('Linear Probability Model', fontsize=14)
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# Calculate correlation coefficient
corr_coef = np.corrcoef(y_X_train['tdta'], y_X_train['yd'])[0, 1]

# Add coefficient information to the plot (bottom right)
coef_text = f"ŷ = {model.params[0]:.3f} + {model.params[1]:.3f}·tdta\n"
coef_text += f"R² = {model.rsquared:.3f}, Corr. = {corr_coef:.3f}"
ax1.text(0.95, 0.05, coef_text, transform=ax1.transAxes, 
         verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

# SUBPLOT 2: Residual distribution with histograms and KDE
# Calculate overall standard error of residuals
residual_std = np.std(residuals)

# Create separate residual series for each group
residuals_yd0 = residuals[yd_0_indices]
residuals_yd1 = residuals[yd_1_indices]

# Calculate means for each group
mean_yd0 = np.mean(residuals_yd0)
mean_yd1 = np.mean(residuals_yd1)

# Plot histograms with KDE
sns.histplot(residuals_yd0, kde=True, color='green', alpha=0.5, 
             label=f'yd=0: n={len(residuals_yd0)}, mean={mean_yd0:.2f}', ax=ax2)
sns.histplot(residuals_yd1, kde=True, color='red', alpha=0.5, 
             label=f'yd=1: n={len(residuals_yd1)}, mean={mean_yd1:.2f}', ax=ax2)

# Generate points for normal distribution with mean=0 and sd=residual_std
x = np.linspace(-3*residual_std, 3*residual_std, 1000)
normal_pdf = stats.norm.pdf(x, loc=0, scale=residual_std)
ax2.plot(x, normal_pdf * len(residuals) * (ax2.get_xlim()[1] - ax2.get_xlim()[0]) / 10, 
         'b--', linewidth=2, label=f'Normal(0, {residual_std:.3f})')

# Add labels and title
ax2.set_xlabel('Residuals', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Distribution of Residuals by Group', fontsize=14)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# Add vertical line at x=0
ax2.axvline(x=0, color='black', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


# In[73]:


# Create figure
fig, ax = plt.subplots(figsize=(9, 8))

# Get predicted values and residuals
y_pred = model.predict(X)
residuals = y - y_pred

# Separate data by yd value
yd_0_indices = y_X_train['yd'] == 0
yd_1_indices = y_X_train['yd'] == 1

# Calculate means for each group
mean_yd0 = np.mean(residuals[yd_0_indices])
mean_yd1 = np.mean(residuals[yd_1_indices])

# Plot residuals vs forecasted values
ax.scatter(y_pred[yd_0_indices], residuals[yd_0_indices], 
           color='green', alpha=0.7, label=f'yd=0: n={len(residuals[yd_0_indices])}, mean={mean_yd0:.2f}')
ax.scatter(y_pred[yd_1_indices], residuals[yd_1_indices], 
           color='red', alpha=0.7, label=f'yd=1: n={len(residuals[yd_1_indices])}, mean={mean_yd1:.2f}')

# Add horizontal line at y=0
ax.axhline(y=0, color='black', linestyle='--', alpha=0.7)

# Add labels and title
ax.set_xlabel('Forecasted Values (Predicted Probabilities)', fontsize=12)
ax.set_ylabel('Residuals', fontsize=12)
ax.set_title('Residuals vs Forecasted Values', fontsize=14)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# In[74]:


# Graph with standardized residuals and +-1.96 limits for asymptotic 5% threshold
# From residuals to standardized residuals close to a multiplication by 2 after division by standard error
# Create figure
fig, ax = plt.subplots(figsize=(9, 8))

# Get predicted values and residuals
y_pred = model.predict(X)
residuals = y - y_pred

# Calculate standardized residuals
# Estimate standard error: sqrt of residual variance (or use model-based standard error)
mse = np.mean(residuals**2)  # Mean squared error
std_error = np.sqrt(mse)  # Residual standard deviation
standardized_residuals = residuals / std_error
# For a linear probability model, you could use heteroskedasticity-consistent standard errors:
# std_error = np.sqrt(y_pred * (1 - y_pred))  # Uncomment if using LPM variance
# standardized_residuals = residuals / std_error

# Separate data by yd value
yd_0_indices = y_X_train['yd'] == 0
yd_1_indices = y_X_train['yd'] == 1

# Calculate means for each group
mean_yd0 = np.mean(standardized_residuals[yd_0_indices])
mean_yd1 = np.mean(standardized_residuals[yd_1_indices])

# Plot standardized residuals vs forecasted values
ax.scatter(y_pred[yd_0_indices], standardized_residuals[yd_0_indices], 
           color='green', alpha=0.7, label=f'yd=0: n={len(standardized_residuals[yd_0_indices])}, mean={mean_yd0:.2f}')
ax.scatter(y_pred[yd_1_indices], standardized_residuals[yd_1_indices], 
           color='red', alpha=0.7, label=f'yd=1: n={len(standardized_residuals[yd_1_indices])}, mean={mean_yd1:.2f}')

# Add horizontal line at y=0
ax.axhline(y=0, color='black', linestyle='--', alpha=0.7)

# Add horizontal lines at ±1.96 (5% bilateral t-test limits)
ax.axhline(y=1.96, color='blue', linestyle='--', alpha=0.7, label='±1.96 (5% t-test limits)')
ax.axhline(y=-1.96, color='blue', linestyle='--', alpha=0.7)

# Add labels and title
ax.set_xlabel('Forecasted Values (Predicted Probabilities)', fontsize=12)
ax.set_ylabel('Standardized Residuals', fontsize=12)
ax.set_title('Standardized Residuals vs Forecasted Values', fontsize=14)
ax.legend(loc='upper right')

# Display the plot
plt.show()


# In[75]:


import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

# Extract the variables from your DataFrame
y = y_X_train['yd']
X = y_X_train['tdta']

# Add a constant to the explanatory variable
X_with_const = sm.add_constant(X)

# Calculate n0 and n1 (observations in each group)
n0 = (y == 0).sum()
n1 = (y == 1).sum()

# 1. Linear Probability Model (OLS)
lpm_model = sm.OLS(y, X_with_const).fit()
lpm_params = lpm_model.params
lpm_tvalues = lpm_model.tvalues
lpm_r2 = lpm_model.rsquared
lpm_predictions = lpm_model.predict(X_with_const)
lpm_auc = roc_auc_score(y, lpm_predictions)

# 2. Logit Model
logit_model = sm.Logit(y, X_with_const).fit(disp=0)
logit_params = logit_model.params
logit_tvalues = logit_model.tvalues
logit_r2 = logit_model.prsquared
logit_predictions = logit_model.predict(X_with_const)
logit_auc = roc_auc_score(y, logit_predictions)

# 3. Probit Model
probit_model = sm.Probit(y, X_with_const).fit(disp=0)
probit_params = probit_model.params
probit_tvalues = probit_model.tvalues
probit_r2 = probit_model.prsquared
probit_predictions = probit_model.predict(X_with_const)
probit_auc = roc_auc_score(y, probit_predictions)

# Create a comparison table
results_table = pd.DataFrame({
    'Linear Probability': [
        f"{lpm_params[0]:.3f} ({lpm_tvalues[0]:.2f})", 
        f"{lpm_params[1]:.3f} ({lpm_tvalues[1]:.2f})",
        f"{lpm_auc:.3f}",  # Added AUC
        f"{lpm_r2:.3f}",
        f"{n0}",
        f"{n1}"
    ],
    'Probit': [
        f"{probit_params[0]:.3f} ({probit_tvalues[0]:.2f})", 
        f"{probit_params[1]:.3f} ({probit_tvalues[1]:.2f})",
        f"{probit_auc:.3f}",  # Added AUC
        f"{probit_r2:.3f}",
        f"{n0}",
        f"{n1}"
    ],
    'Logit': [
        f"{logit_params[0]:.3f} ({logit_tvalues[0]:.2f})", 
        f"{logit_params[1]:.3f} ({logit_tvalues[1]:.2f})",
        f"{logit_auc:.3f}",  # Added AUC
        f"{logit_r2:.3f}",
        f"{n0}",
        f"{n1}"
    ]
}, index=['Intercept', 'tdta', 'AUC', 'R²/Pseudo-R²', 'n₀', 'n₁'])

# Display the table
print("Comparison of Linear Probability, Logit, and Probit Models")
print("Parameter estimates with t-statistics in parentheses")
print(results_table)


# In[76]:


# Create a figure
plt.figure(figsize=(12, 8))

# Plot original data with requested colors
plt.scatter(df.loc[df['yd'] == 0, 'tdta'], df.loc[df['yd'] == 0, 'yd'], 
           color='green', label='Actual yd=0', alpha=0.6)
plt.scatter(df.loc[df['yd'] == 1, 'tdta'], df.loc[df['yd'] == 1, 'yd'], 
           color='red', label='Actual yd=1', alpha=0.6)

# Fit the models
X = sm.add_constant(df['tdta'])
lpm_model = sm.OLS(df['yd'], X).fit()
logit_model = sm.Logit(df['yd'], X).fit(disp=0)
probit_model = sm.Probit(df['yd'], X).fit(disp=0)

# Get predictions for each model
lpm_preds = lpm_model.predict(X)
logit_preds = logit_model.predict(X)
probit_preds = probit_model.predict(X)

# Plot LPM forecasts as crosses - matching the color to the actual yd value
# For yd=0 points
plt.scatter(df.loc[df['yd'] == 0, 'tdta'], lpm_preds[df['yd'] == 0], 
           color='green', marker='x', s=50, label='LPM forecast for yd=0')
# For yd=1 points
plt.scatter(df.loc[df['yd'] == 1, 'tdta'], lpm_preds[df['yd'] == 1], 
           color='red', marker='x', s=50, label='LPM forecast for yd=1')

# Plot Logit forecasts as crosses
# For yd=0 points
plt.scatter(df.loc[df['yd'] == 0, 'tdta'], logit_preds[df['yd'] == 0], 
           color='green', marker='x', s=40, alpha=0.7)
# For yd=1 points
plt.scatter(df.loc[df['yd'] == 1, 'tdta'], logit_preds[df['yd'] == 1], 
           color='red', marker='x', s=40, alpha=0.7)

# Plot Probit forecasts as crosses
# For yd=0 points
plt.scatter(df.loc[df['yd'] == 0, 'tdta'], probit_preds[df['yd'] == 0], 
           color='green', marker='x', s=30, alpha=0.7)
# For yd=1 points
plt.scatter(df.loc[df['yd'] == 1, 'tdta'], probit_preds[df['yd'] == 1], 
           color='red', marker='x', s=30, alpha=0.7)

# Create a range of tdta values for smooth curves
tdta_range = np.linspace(df['tdta'].min(), df['tdta'].max(), 100)
X_curve = sm.add_constant(tdta_range)

# Generate predictions for the smooth curves
lpm_curve = lpm_model.predict(X_curve)
logit_curve = logit_model.predict(X_curve)
probit_curve = probit_model.predict(X_curve)

# Plot smooth curves with requested colors
plt.plot(tdta_range, lpm_curve, color='grey', linewidth=2, label='LPM Curve')
plt.plot(tdta_range, logit_curve, color='lightblue', linewidth=2, label='Logit Curve')
plt.plot(tdta_range, probit_curve, color='mediumpurple', linewidth=2, label='Probit Curve')

plt.xlabel('tdta')
plt.ylabel('Probability of Default (yd)')
plt.title('Default Probability Models with Varying tdta')
plt.legend()
plt.grid(True)
plt.show()


# In[77]:


# Extract the variables from your DataFrame
y = y_X_train['yd']
X = y_X_train['tdta']

# Add a constant to the explanatory variable
X_with_const = sm.add_constant(X)

# Calculate n0 and n1 (observations in each group)
n0 = (y == 0).sum()
n1 = (y == 1).sum()

# 1. Linear Probability Model (OLS)
lpm_model = sm.OLS(y, X_with_const).fit()
lpm_params = lpm_model.params
lpm_tvalues = lpm_model.tvalues
lpm_r2 = lpm_model.rsquared
lpm_predictions = lpm_model.predict(X_with_const)
lpm_auc = roc_auc_score(y, lpm_predictions)

# 2. Logit Model
logit_model = sm.Logit(y, X_with_const).fit(disp=0)
logit_params = logit_model.params
logit_tvalues = logit_model.tvalues
logit_r2 = logit_model.prsquared
logit_predictions = logit_model.predict(X_with_const)
logit_auc = roc_auc_score(y, logit_predictions)

# 3. Probit Model
probit_model = sm.Probit(y, X_with_const).fit(disp=0)
probit_params = probit_model.params
probit_tvalues = probit_model.tvalues
probit_r2 = probit_model.prsquared
probit_predictions = probit_model.predict(X_with_const)
probit_auc = roc_auc_score(y, probit_predictions)

# Create a comparison table
results_table = pd.DataFrame({
    'Linear Probability': [
        f"{lpm_params[0]:.3f} ({lpm_tvalues[0]:.2f})", 
        f"{lpm_params[1]:.3f} ({lpm_tvalues[1]:.2f})",
        f"{lpm_auc:.3f}",  # Added AUC
        f"{lpm_r2:.3f}",
        f"{n0}",
        f"{n1}"
    ],
    'Probit': [
        f"{probit_params[0]:.3f} ({probit_tvalues[0]:.2f})", 
        f"{probit_params[1]:.3f} ({probit_tvalues[1]:.2f})",
        f"{probit_auc:.3f}",  # Added AUC
        f"{probit_r2:.3f}",
        f"{n0}",
        f"{n1}"
    ],
    'Logit': [
        f"{logit_params[0]:.3f} ({logit_tvalues[0]:.2f})", 
        f"{logit_params[1]:.3f} ({logit_tvalues[1]:.2f})",
        f"{logit_auc:.3f}",  # Added AUC
        f"{logit_r2:.3f}",
        f"{n0}",
        f"{n1}"
    ]
}, index=['Intercept', 'tdta', 'AUC', 'R²/Pseudo-R²', 'n₀', 'n₁'])

# Display the table
print("Comparison of Linear Probability, Logit, and Probit Models")
print("Parameter estimates with t-statistics in parentheses")
print(results_table)


# In[78]:


# Extract dependent variable (yd) from y_X_train
y = y_X_train['yd']

# Extract independent variables from X_train
X = X_train[['tdta', 'gempl', 'opita', 'invsls', 'lsls']]

# Add a constant to the explanatory variables
X_with_const = sm.add_constant(X)

# Calculate n0 and n1 (observations in each group)
n0 = (y == 0).sum()
n1 = (y == 1).sum()

# 1. Linear Probability Model (OLS)
lpm_model = sm.OLS(y, X_with_const).fit()
lpm_params = lpm_model.params
lpm_tvalues = lpm_model.tvalues
lpm_r2 = lpm_model.rsquared
lpm_predictions = lpm_model.predict(X_with_const)
lpm_auc = roc_auc_score(y, lpm_predictions)

# 2. Logit Model
logit_model = sm.Logit(y, X_with_const).fit(disp=0)
logit_params = logit_model.params
logit_tvalues = logit_model.tvalues
logit_r2 = logit_model.prsquared
logit_predictions = logit_model.predict(X_with_const)
logit_auc = roc_auc_score(y, logit_predictions)

# 3. Probit Model
probit_model = sm.Probit(y, X_with_const).fit(disp=0)
probit_params = probit_model.params
probit_tvalues = probit_model.tvalues
probit_r2 = probit_model.prsquared
probit_predictions = probit_model.predict(X_with_const)
probit_auc = roc_auc_score(y, probit_predictions)

# Create a comparison table
results_table = pd.DataFrame({
    'Linear Probability': [
        f"{lpm_params['const']:.3f} ({lpm_tvalues['const']:.2f})",
        f"{lpm_params['tdta']:.3f} ({lpm_tvalues['tdta']:.2f})",
        f"{lpm_params['gempl']:.3f} ({lpm_tvalues['gempl']:.2f})",
        f"{lpm_params['opita']:.3f} ({lpm_tvalues['opita']:.2f})",
        f"{lpm_params['invsls']:.3f} ({lpm_tvalues['invsls']:.2f})",
        f"{lpm_params['lsls']:.3f} ({lpm_tvalues['lsls']:.2f})",
        f"{lpm_auc:.3f}",
        f"{lpm_r2:.3f}",
        f"{n0}",
        f"{n1}"
    ],
    'Probit': [
        f"{probit_params['const']:.3f} ({probit_tvalues['const']:.2f})",
        f"{probit_params['tdta']:.3f} ({probit_tvalues['tdta']:.2f})",
        f"{probit_params['gempl']:.3f} ({probit_tvalues['gempl']:.2f})",
        f"{probit_params['opita']:.3f} ({probit_tvalues['opita']:.2f})",
        f"{probit_params['invsls']:.3f} ({probit_tvalues['invsls']:.2f})",
        f"{probit_params['lsls']:.3f} ({probit_tvalues['lsls']:.2f})",
        f"{probit_auc:.3f}",
        f"{probit_r2:.3f}",
        f"{n0}",
        f"{n1}"
    ],
    'Logit': [
        f"{logit_params['const']:.3f} ({logit_tvalues['const']:.2f})",
        f"{logit_params['tdta']:.3f} ({logit_tvalues['tdta']:.2f})",
        f"{logit_params['gempl']:.3f} ({logit_tvalues['gempl']:.2f})",
        f"{logit_params['opita']:.3f} ({logit_tvalues['opita']:.2f})",
        f"{logit_params['invsls']:.3f} ({logit_tvalues['invsls']:.2f})",
        f"{logit_params['lsls']:.3f} ({logit_tvalues['lsls']:.2f})",
        f"{logit_auc:.3f}",
        f"{logit_r2:.3f}",
        f"{n0}",
        f"{n1}"
    ]
}, index=['Intercept', 'tdta', 'gempl', 'opita', 'invsls', 'lsls', 'AUC', 'R²/Pseudo-R²', 'n₀', 'n₁'])

# Display the table
print("Comparison of Linear probability, Logit and Probit models")
print("The dependent variable is default=1, non-default=0")
print("Parameter estimates with t-statistics in parentheses")
print(results_table)


# In[79]:


# Comment all the differences of the above customized table with the following default table
# Extract dependent variable (yd) from y_X_train
y = y_X_train['yd']

# Extract independent variables from X_train
X = X_train[['tdta', 'gempl', 'opita', 'invsls', 'lsls']]

# Add constant term for intercept
X = sm.add_constant(X)

# Fit linear probability model
model = sm.OLS(y, X).fit()

# Print regression summary
print(model.summary())


# In[80]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Assuming X_train, X_test, y_train, y_test are already defined
# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[['tdta', 'gempl', 'opita', 'invsls', 'lsls']])
X_test_scaled = scaler.transform(X_test[['tdta', 'gempl', 'opita', 'invsls', 'lsls']])

# Linear Probability Model (LPM)
lpm_model = LinearRegression()
lpm_model.fit(X_train_scaled, y_train)
lpm_pred = lpm_model.predict(X_test_scaled)

# Probit Model
probit_model = sm.Probit(y_train, sm.add_constant(X_train_scaled)).fit(disp=0)
probit_pred = probit_model.predict(sm.add_constant(X_test_scaled))

# Logit Model
logit_model = LogisticRegression(solver='liblinear')
logit_model.fit(X_train_scaled, y_train)
logit_pred = logit_model.predict_proba(X_test_scaled)[:, 1]

# Compute ROC curves and AUC
lpm_fpr, lpm_tpr, _ = roc_curve(y_test, lpm_pred)
probit_fpr, probit_tpr, _ = roc_curve(y_test, probit_pred)
logit_fpr, logit_tpr, _ = roc_curve(y_test, logit_pred)

lpm_auc = auc(lpm_fpr, lpm_tpr)
probit_auc = auc(probit_fpr, probit_tpr)
logit_auc = auc(logit_fpr, logit_tpr)

# Plot ROC curves
plt.figure(figsize=(8, 6))
plt.plot(lpm_fpr, lpm_tpr, label=f'LPM (AUC = {lpm_auc:.2f})')
plt.plot(probit_fpr, probit_tpr, label=f'Probit (AUC = {probit_auc:.2f})')
plt.plot(logit_fpr, logit_tpr, label=f'Logit (AUC = {logit_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('roc_comparison.png')
plt.close()


# In[81]:


# Fit the analysis of variance robability model
model = sm.OLS(y, X).fit()
# Print the summary of the model
print(model.summary())


# In[82]:


# Calculate ROC curves for each model with tdta only 
fpr_lpm, tpr_lpm, _ = roc_curve(y, lpm_predictions)
fpr_probit, tpr_probit, _ = roc_curve(y, probit_predictions)
fpr_logit, tpr_logit, _ = roc_curve(y, logit_predictions)

# Calculate AUC for each model (already computed, but included for legend)
lpm_auc = roc_auc_score(y, lpm_predictions)
probit_auc = roc_auc_score(y, probit_predictions)
logit_auc = roc_auc_score(y, logit_predictions)

# Create the ROC plot
plt.figure(figsize=(8, 6))
plt.plot(fpr_lpm, tpr_lpm, label=f'Linear Probability (AUC = {lpm_auc:.3f})', color='blue', linewidth=2)
plt.plot(fpr_probit, tpr_probit, label=f'Probit (AUC = {probit_auc:.3f})', color='green', linewidth=2)
plt.plot(fpr_logit, tpr_logit, label=f'Logit (AUC = {logit_auc:.3f})', color='red', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)', linewidth=1)  # Diagonal line

# Customize the plot
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curves for Linear Probability, Probit, and Logit Models')
plt.legend(loc='lower right')
plt.grid(True)
plt.axis([0, 1, 0, 1])  # Set axis limits to [0,1] for both FPR and TPR

# Save the plot
plt.savefig('roc_curves.png')
plt.show()


# In[94]:


# Assuming X_train, X_test, y_train, y_test are already defined
X_train_features = X_train[['tdta', 'gempl', 'opita', 'invsls', 'lsls']]
X_test_features = X_test[['tdta', 'gempl', 'opita', 'invsls', 'lsls']]

# Linear Probability Model (LPM)
lpm_model = LinearRegression()
lpm_model.fit(X_train_features, y_train)
lpm_pred = lpm_model.predict(X_test_features)

# Probit Model
X_train_probit = sm.add_constant(X_train_features)
X_test_probit = sm.add_constant(X_test_features)
probit_model = sm.Probit(y_train, X_train_probit).fit(disp=0)
probit_pred = probit_model.predict(X_test_probit)

# Logit Model
logit_model = LogisticRegression(solver='liblinear')
logit_model.fit(X_train_features, y_train)
logit_pred = logit_model.predict_proba(X_test_features)[:, 1]

# Compute ROC curves and AUC
lpm_fpr, lpm_tpr, _ = roc_curve(y_test, lpm_pred)
probit_fpr, probit_tpr, _ = roc_curve(y_test, probit_pred)
logit_fpr, logit_tpr, _ = roc_curve(y_test, logit_pred)

lpm_auc = auc(lpm_fpr, lpm_tpr)
probit_auc = auc(probit_fpr, probit_tpr)
logit_auc = auc(logit_fpr, logit_tpr)

# Plot ROC curves
plt.figure(figsize=(8, 6))
plt.plot(lpm_fpr, lpm_tpr, label=f'LPM (AUC = {lpm_auc:.2f})', color='#1f77b4')
plt.plot(probit_fpr, probit_tpr, label=f'Probit (AUC = {probit_auc:.2f})', color='#ff7f0e')
plt.plot(logit_fpr, logit_tpr, label=f'Logit (AUC = {logit_auc:.2f})', color='#2ca02c')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# In[95]:


lpm_train_pred = lpm_model.predict(X_train_features)
probit_train_pred = probit_model.predict(sm.add_constant(X_train_features))
logit_train_pred = logit_model.predict_proba(X_train_features)[:, 1]
lpm_train_auc = roc_auc_score(y_train, lpm_train_pred)
probit_train_auc = roc_auc_score(y_train, probit_train_pred)
logit_train_auc = roc_auc_score(y_train, logit_train_pred)
print(f"LPM Train AUC: {lpm_train_auc:.2f}, Test AUC: {lpm_auc:.2f}")
print(f"Probit Train AUC: {probit_train_auc:.2f}, Test AUC: {probit_auc:.2f}")
print(f"Logit Train AUC: {logit_train_auc:.2f}, Test AUC: {logit_auc:.2f}")


# In[96]:


print("Train feature means:\n", X_train[['tdta', 'gempl', 'opita', 'invsls', 'lsls']].mean())
print("Test feature means:\n", X_test[['tdta', 'gempl', 'opita', 'invsls', 'lsls']].mean())


# In[98]:


from scipy.stats import ttest_ind
for feature in ['tdta', 'gempl', 'opita', 'invsls', 'lsls']:
    stat, p = ttest_ind(X_train[feature], X_test[feature])
    print(f"{feature} t-test p-value: {p:.4f}")


# In[101]:


print("Train feature std:\n", X_train[['tdta', 'gempl', 'opita', 'invsls', 'lsls']].std())
print("Test feature std:\n", X_test[['tdta', 'gempl', 'opita', 'invsls', 'lsls']].std())


# In[100]:


import matplotlib.pyplot as plt
for feature in ['tdta', 'gempl', 'opita', 'invsls', 'lsls']:
    plt.figure(figsize=(6, 4))
    plt.hist(X_train[feature], bins=30, alpha=0.5, label='Train', color='blue')
    plt.hist(X_test[feature], bins=30, alpha=0.5, label='Test', color='orange')
    plt.title(f'Distribution of {feature}')
    plt.legend()
    plt.show()


# In[84]:


# Correlation matrix default shape with Python
y_X_train[['yd','reta', 'tdta']].corr()


# In[85]:


"""
More readable correlation matrix with heat map -1 and 1, lower triangular, t-statistics below and two decimals
I already ordered the variables with respect to their correlation with the dependent in absolute value 
followed by their highly correlated close accounting ratio 
"""
# step1, degrees of freedom may change depending on number of missing observations for some variables for FTAT LTDTA
n_samples = y_X_train.shape[0]
print(n_samples)


# In[86]:


"""
Compute t-statistics for testing H_0: rho = 0 for each correlation coefficient,
t_stats[i, j] = (r * np.sqrt(n - 2)) / np.sqrt(1 - r**2)
accounting for missing (NaN) values in each variable pair.
Parameters:
corr_matrix: Correlation matrix (pandas DataFrame)
y_X_train: DataFrame with target and feature variables
Returns:
t_stats: Array of t-statistics with the same shape as corr_matrix
n_valid: Array of number of non-NaN observations for each pair
"""
def compute_t_statistics(corr_matrix, y_X_train):
    t_stats = np.full_like(corr_matrix, np.nan, dtype=float)
    n_valid = np.zeros_like(corr_matrix, dtype=int)
    
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            # Get non-NaN pair counts for variables i and j
            valid_rows = y_X_train.iloc[:, [i, j]].dropna()
            n = len(valid_rows)
            n_valid[i, j] = n
            
            if n < 3:  # Need at least 3 observations for valid t-statistic
                t_stats[i, j] = np.nan
                continue
                
            r = corr_matrix.iloc[i, j]
            if np.isnan(r):
                t_stats[i, j] = np.nan
            elif abs(r) >= 0.99999:  # Avoid division by zero for r = ±1
                t_stats[i, j] = np.inf if r > 0 else -np.inf
            else:
                t_stats[i, j] = (r * np.sqrt(n - 2)) / np.sqrt(1 - r**2)
    
    return t_stats, n_valid


# In[87]:


"""
Lower triangular correlation matrix of dependent and explanatory variables with (t-statistics) and heatmap
"""
corr_matrix = y_X_train.corr()
# Calculate t-statistics and valid observation counts
t_stats, n_valid = compute_t_statistics(corr_matrix, y_X_train)
# Create custom annotations (correlation + t-statistic)
annot = np.array([
        [f"{corr_matrix.iloc[i, j]:.2f}\n({t_stats[i, j]:.2f})" 
         if not np.isnan(corr_matrix.iloc[i, j]) and n_valid[i, j] >= 3
         else "NaN" if np.isnan(corr_matrix.iloc[i, j]) else f"{corr_matrix.iloc[i, j]:.2f}"
         for j in range(corr_matrix.shape[1])]
        for i in range(corr_matrix.shape[0])
    ])
# Facilitate the visualisation of correlations
# Lower triangular correlation matrix with heatmap for absolute values of correlation
from seaborn import heatmap
plt.figure(figsize=(12, 10))
# Create a mask for the upper triangle and main diagonal
mask = np.triu(np.ones_like(y_X_train.corr(), dtype=bool))
# Create custom diverging colormap where red is near -1 and 1
cmap = sns.diverging_palette(0, 0, s=75, l=40, n=9, center="light", as_cmap=True)
sns.heatmap( 
    y_X_train.corr(), 
    fmt='',  # Use empty format since annotations are pre-formatted
    annot=annot, 
    vmin=-1, 
    vmax=1,
    center=0,
    cbar=True,
    square=True,
    cmap=cmap,
    mask=mask
    )
plt.title("Lower triangular correlation matrix of dependent and explanatory variables with (t-statistics)")
plt.show()
   


# In[88]:


# Bivariate graphs of correlation matrix dependent and first 7 explanatory variables 
# with kde histograms by groups on the main diagonal
new_df = y_X_train.copy()
new_df["target"] = y_train
vars = ["yd", "tdta", "reta", "opita", "ebita", "lsls", "lta", "gempl"]

# Check for zero variance in the variables
for var in vars:
    if new_df[var].var() == 0:
        print(f"Warning: Variable {var} has zero variance. KDE may fail.")

# Set up the PairGrid
g = sns.PairGrid(new_df, vars=vars, hue="target", palette={0: "green", 1: "red"})

# Map KDE to diagonal, split by hue, with filled areas
g.map_diag(sns.kdeplot, hue_order=[0, 1], common_norm=False, warn_singular=False, fill=True)

# Custom function for scatter plot and single regression line
def plot_scatter_reg(x, y, **kwargs):
    # Extract data from kwargs
    data = new_df
    # Scatter plot with hue
    sns.scatterplot(x=data[x.name], y=data[y.name], hue=data["target"], palette={0: "green", 1: "red"})
    # Single regression line (ignoring hue)
    sns.regplot(x=data[x.name], y=data[y.name], color="blue", scatter=False, ci=None, line_kws={"linewidth": 2})

# Map the custom function to off-diagonal
g.map_offdiag(plot_scatter_reg)

# Add legend
g.add_legend()

# Show plot
plt.show()


# In[89]:


# Bivariate graphs of correlation matrix dependent and other 7 explanatory variables 
# with kde histograms by groups on the main diagonal
new_df = y_X_train.copy()
new_df["target"] = y_train
vars = ["yd", "invsls","nwcta","cacl","qacl","fata","ltdta","mveltd"]

# Check for zero variance in the variables
for var in vars:
    if new_df[var].var() == 0:
        print(f"Warning: Variable {var} has zero variance. KDE may fail.")

# Set up the PairGrid
g = sns.PairGrid(new_df, vars=vars, hue="target", palette={0: "green", 1: "red"})

# Map KDE to diagonal, split by hue, with filled areas
g.map_diag(sns.kdeplot, hue_order=[0, 1], common_norm=False, warn_singular=False, fill=True)

# Custom function for scatter plot and single regression line
def plot_scatter_reg(x, y, **kwargs):
    # Extract data from kwargs
    data = new_df
    # Scatter plot with hue
    sns.scatterplot(x=data[x.name], y=data[y.name], hue=data["target"], palette={0: "green", 1: "red"})
    # Single regression line (ignoring hue)
    sns.regplot(x=data[x.name], y=data[y.name], color="blue", scatter=False, ci=None, line_kws={"linewidth": 2})

# Map the custom function to off-diagonal
g.map_offdiag(plot_scatter_reg)

# Add legend
g.add_legend()

# Show plot
plt.show()


# In[90]:


# Visualization of KDE distributions of explanatory variables by binary dependent and correlations first 7 variables
new_df=y_X_train.copy()
new_df["target"]=y_train
vars = ["yd","tdta", "reta", "opita", "ebita", "lsls", "lta", "gempl"]
p=pairplot(
    new_df, vars=vars, hue="target", palette = {0: "green", 1: "red"}, kind="reg"
    )


# In[91]:


# Visualization of distributions of variables by groups and correlations 7 other variables
new_df=y_X_train.copy()
new_df["target"]=y_train
vars = [ "invsls","nwcta","cacl","qacl","fata","ltdta","mveltd"]
p=pairplot(
    new_df, vars=vars, hue="target", palette = {0: "green", 1: "red"}, kind="reg"
    )


# In[92]:


# Zoom Visualization of distributions of variables by groups and correlations 4 first variables
vars = ["tdta", "reta", "opita", "ebita"]
p=pairplot(
    new_df, vars=vars, hue="target", palette = {0: "green", 1: "red"}, kind="reg"
    )


# In[ ]:





# In[ ]:





# In[93]:


#Export en Latex
latex_table_descstats = merged_descriptive.to_latex(float_format="%.2f", caption="Descriptive Statistics of Financial ratio", 
                                           label="tab:descriptive_stats")


# In[ ]:




