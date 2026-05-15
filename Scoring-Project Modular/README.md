# Credit Scoring and Default Prediction

This project builds a Python workflow for credit scoring and corporate default prediction. The objective is to use borrower-level financial ratios to estimate default risk, compare alternative probability models, and evaluate their predictive performance.

The project applies several standard models used in credit-risk analysis, including a Linear Probability Model, Logit, and Probit. It also compares benchmark financial ratios with transformed variables and evaluates the models using ROC curves, AUC, confusion matrices, residual diagnostics, and threshold-based loss functions.

## Project Structure

```text
.
├── main.py
├── data_cleaning.py
├── data_statistical_analysis.py
├── regression_estimation.py
├── model_evaluation.py
├── default2000.csv
