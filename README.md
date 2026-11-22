# Loan Approval Predictor (Decision Tree)

A binary classification project that predicts whether a home loan application will be **approved or rejected**, tests whether new features and class weights improve a decision tree, and exports the final tree to a web page that shows the exact question it asked.

**Live demo:** https://anju-raman1.github.io/Loan_Approval_Predictor/

> **Educational project.** It must not be used to make real lending decisions.

## What this project shows

- How a decision tree works, and how deeper trees **overfit** (train accuracy rises, cross-validated accuracy falls).
- Looking beyond accuracy: a model with 85% test accuracy still approved 17 of 38 applications that should have been rejected.
- Testing two improvements fairly: **new features** (total income, monthly instalment, instalment-to-income ratio) and **class weights**, using 5-fold cross-validation repeated 3 times instead of the small test set.
- Reporting an honest result: **the simplest tree won**.
- A fairness decision: gender and marital status are not used as features.

## Data

Loan Prediction dataset (a public practice dataset from a housing finance company): 614 applications, 12 input columns. About 69% of applications were approved. A missing credit history (50 applications) is kept as its own group, **Unknown**, instead of guessing a value. Other missing values are filled inside the model pipeline, using only training data.

## Method

- 80/20 stratified train/test split (491 train, 123 test), `random_state=42`.
- Pipeline: median imputation for numbers, most-common imputation and one-hot encoding for categories, then `DecisionTreeClassifier(min_samples_leaf=5)`.
- **Step 1:** depth 1 to full, using the original features, compared by train, cross-validated and test accuracy.
- **Step 2:** all 20 combinations of features (original or with the 3 new ones), class weight (none or balanced) and depth (1 to 5), scored by cross-validated balanced accuracy. Ties go to the simpler tree.
- The test set was used once, at the end.

## Results

**Step 1: depth (original features, no class weights)**

| Max depth | Leaves | Train accuracy | CV accuracy | Test accuracy |
|---|---|---|---|---|
| **1** | 2 | 0.798 | **0.798** | 0.854 |
| 2 | 4 | 0.798 | 0.788 | 0.854 |
| 3 | 8 | 0.798 | 0.774 | 0.854 |
| 4 | 13 | 0.817 | 0.754 | 0.821 |
| 5 | 18 | 0.825 | 0.745 | 0.821 |
| 6 | 25 | 0.829 | 0.741 | 0.813 |
| 8 | 39 | 0.841 | 0.731 | 0.756 |
| 10 | 47 | 0.855 | 0.713 | 0.756 |
| None (full) | 52 | 0.857 | 0.709 | 0.764 |

**Step 2: selected setups (cross-validated, 15 scores each)**

| Features | Class weight | Depth | CV accuracy | CV balanced accuracy | Rejected caught | Approved found |
|---|---|---|---|---|---|---|
| **Original** | **none** | **1** | 0.798 | **0.689** | 0.396 | 0.982 |
| With new features | none | 1 | 0.798 | 0.689 | 0.396 | 0.982 |
| With new features | none | 2 | 0.792 | 0.686 | 0.401 | 0.970 |
| Original | balanced | 3 | 0.735 | 0.661 | 0.459 | 0.862 |
| Original | balanced | 5 | 0.678 | 0.639 | 0.535 | 0.743 |

**Final comparison on the 123 test applications**

| Model | Accuracy | Balanced accuracy | Rejected caught | Approved found | Bad loans wrongly approved | Good loans wrongly rejected |
|---|---|---|---|---|---|---|
| Baseline (always approve) | 0.691 | 0.500 | 0.000 | 1.000 | 38 | 0 |
| Credit-history rule (depth 1) | 0.854 | 0.770 | 0.553 | 0.988 | 17 | 1 |
| **Best setup (same depth-1 tree)** | 0.854 | 0.770 | 0.553 | 0.988 | 17 | 1 |

### Key findings

- **The best setup was the simple depth-1 tree.** It asks one question: is the credit history Poor? Deeper trees overfit: train accuracy rises from 0.798 to 0.857 while cross-validated accuracy falls to 0.709 for the full 52-leaf tree.
- **The new features did not help.** With and without them, the depth-1 trees score identically, and the deeper trees with new features are no better than those without. The box plots show why: total income, instalment and instalment-to-income overlap heavily between approved and rejected applications.
- **Class weights shifted the trade-off but did not win.** They raise the share of rejected applications caught (from about 0.40 up to 0.54) but reject many more good applications (approved found falls from 0.98 to as low as 0.74), so balanced accuracy drops.
- **Credit history dominates.** About 80% of applicants with a good credit history were approved, against 8% with a poor one; applicants with an unknown credit history were approved about 74% of the time.
- **Test accuracy flatters the model.** The test accuracy of 0.854 is higher than the cross-validated 0.798, most likely luck from a small test set. In cross-validation the tree catches only about 40% of the applications that should be rejected, against 55% on the test set. About 0.80 accuracy is the more reliable estimate.

## Limitations

- The dataset is small (about 600 applications), so the test set holds only 123 and results can change a lot with a different split.
- Most of the predictive power comes from one feature, so the model is essentially a credit-history rule.
- The data comes from one company at one point in time, and real lending decisions depend on many more factors.
- Class weights make the tree more cautious, but a wrongly rejected good applicant is also a real cost. The balance is a business decision, not only a technical one.
- Gender and marital status were left out on purpose for fairness, but other features can still be correlated with them.

## Next steps

- Compare with Random Forest and Gradient Boosting on the same data.
- Tune `min_samples_leaf` and the splitting criterion with `GridSearchCV`.
- Put an explicit cost on each type of mistake and choose the setup that minimizes the total cost.

## Files

| File | Purpose |
|---|---|
| `loan_approval_prediction.ipynb` | Full analysis (open in Google Colab) |
| `model.json` | The final tree: for each node, the feature, threshold and where to go next |
| `index.html` | Web page that walks an application through the tree in JavaScript |

## How the website works

A decision tree is a list of questions, so the whole tree is saved in `model.json`. The page repeats the steps in the browser and shows the path the application followed. Because the final tree has only one question (is the credit history Poor?), **only the credit history changes the decision**; the other fields are calculated and shown but do not affect it. For the check application used to verify the page (credit history Poor, rural property, and so on), both the notebook and the website say **Rejected**, with **9.0%** of similar past applications approved.

## Tools

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, HTML/JavaScript, GitHub Pages.
