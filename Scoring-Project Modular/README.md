# Credit Scoring and Default Prediction

This project builds a Python workflow for credit scoring and corporate default prediction. The objective is to use borrower-level financial ratios to estimate default risk, compare alternative probability models, and evaluate their predictive performance.

The project applies several standard models used in credit-risk analysis, including a Linear Probability Model, Logit, and Probit. It also compares benchmark financial ratios with transformed variables and evaluates the models using ROC curves, AUC, confusion matrices, residual diagnostics, and threshold-based loss functions.

## Project Structure

```text
├── main.py
├── data_cleaning.py
├── data_statistical_analysis.py
├── regression_estimation.py
├── model_evaluation.py
├── default2000.csv
```

## Main Files

### `main.py`

Runs the full workflow from data preparation to model estimation and evaluation.

### `data_cleaning.py`

Loads and prepares the raw dataset. The cleaning process includes handling missing or abnormal values, converting variables into usable numeric formats, and preparing the final estimation and validation samples.

### `data_statistical_analysis.py`

Produces the exploratory analysis used to understand the financial ratios before modeling. This includes descriptive statistics, distribution plots, boxplots, normality checks, tests of mean differences, simple correlations with the default dummy, and correlation analysis between explanatory variables.

### `regression_estimation.py`

Estimates the main default-prediction models. The project compares:

- Linear Probability Model
- Logit
- Probit

The models are estimated using both benchmark financial ratios and transformed variables.

### `model_evaluation.py`

Evaluates model performance using classification and ranking metrics. This includes ROC curves, AUC, confusion matrices, standardized Pearson residuals, outlier diagnostics, and threshold selection using a credit-loss framework.

## Methodology

The project follows a credit-scoring workflow:

1. Clean the raw dataset and handle missing or abnormal observations.
2. Analyze each financial ratio using visual inspection and summary statistics.
3. Compare defaulting and non-defaulting firms using tests of distribution, mean differences, and correlations.
4. Identify variables with stronger predictive content for default.
5. Check for multicollinearity between financial ratios.
6. Estimate default-prediction models using LPM, Logit, and Probit.
7. Compare benchmark variables with transformed specifications.
8. Evaluate in-sample and out-of-sample model performance.
9. Use a loss-function approach to choose lending thresholds.

## Model Evaluation

The project evaluates each model using:

- ROC curves
- Area Under the Curve
- confusion matrices
- classification accuracy
- standardized Pearson residuals
- type I and type II error analysis
- credit-loss-based threshold selection

AUC is used as the main ranking metric because it measures how well the model separates defaulting from non-defaulting firms across all possible thresholds.

## Economic Interpretation

The results are consistent with standard credit-risk intuition. Higher leverage is associated with higher default risk, while stronger profitability and firm growth are associated with lower default risk.

The comparison between LPM, Logit, and Probit shows that the models are broadly consistent in terms of coefficient signs and economic interpretation. Logit and Probit are more appropriate for probability modeling because they constrain predicted probabilities between 0 and 1 and allow for nonlinear changes in default probability.
