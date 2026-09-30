# Student Performance: Linear, Ridge and Lasso Regression

Author: Akshat Shah

Predict the continuous **Performance Index** using Hours Studied, Previous Scores, Extracurricular Activities, Sleep Hours, and Sample Question Papers Practiced.

## Method
- Source: 10,000 synthetic records from Kaggle; no missing values.
- Remove 127 exact duplicate rows, retaining 9,873 rows. There is no student ID; this is a documented deduplication assumption.
- Encode extracurricular activities as Yes=1, No=0.
- Split 80/20 with random_state=42: 7,898 training rows and 1,975 test rows.
- Standardize predictors inside each model pipeline. Scaling is fitted within each training fold, without test-data leakage.
- Tune Ridge and Lasso using five-fold training-only cross-validation and mean validation RMSE. All models share the same test set and CV folds.
- Evaluate MAE, MSE, RMSE and R², and compare standardized and original-unit coefficients.

## Held-out test comparison
| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Linear Regression | 1.646970 | 4.305901 | 2.075066 | 0.988430 |
| Ridge | 1.646970 | 4.305901 | 2.075066 | 0.988430 |
| Lasso | 1.646995 | 4.306034 | 2.075099 | 0.988430 |

## Conclusion
Linear Regression has the lowest held-out RMSE on this split (2.075066) with R² = 0.988430, while training-only cross-validation selects Lasso. The three test RMSEs span only 0.000032 Performance Index points, so this comparison offers little evidence of a meaningful generalization advantage; ordinary Linear Regression is a reasonable simpler choice. Ridge applies L2 shrinkage and Lasso applies L1 shrinkage, but the selected penalties change the coefficients only slightly and Lasso sets 0 of 5 coefficients to zero, indicating little benefit from regularization for this synthetic dataset.

## Files
- `Student_Performance_Regression.ipynb`: notebook with executed outputs, tables, three chart figures, and a three-sentence conclusion.
- `data/Student_Performance.csv`: original source data, unchanged.
- `outputs/model_comparison.csv`: required test-metric comparison.
- `outputs/coefficients_standardized.csv` and `outputs/coefficients_original_units.csv`: coefficient comparisons.
- `outputs/cross_validation_results.csv` and alpha-search CSVs: tuning results.
- `outputs/train_test_rmse.csv`: training/test error comparison.
- `outputs/split_manifest.csv` and `outputs/test_predictions.csv`: split membership and test predictions, with source row indices.
- `outputs/charts/`: coefficient comparison, regularization paths, prediction diagnostics.
- `outputs/conclusion.md`: the required three-sentence conclusion.
- `SUBMISSION_GUIDE.md`: Kaggle, GitHub and Colab instructions.

## Run
Kaggle: import the notebook and attach the linked dataset. Colab: upload the notebook and source CSV. Local Jupyter: keep the directory structure, then run:

```bash
python -m pip install -r requirements.txt
jupyter notebook Student_Performance_Regression.ipynb
```

Run all cells from the top. Saved notebook outputs are included for reading without execution. Generated files are written to `outputs/`.

## Interpretation and limitations
The lowest test-RMSE model is Linear Regression, while the smallest mean training-CV RMSE belongs to Lasso; both differences are negligible. A single split does not establish statistically reliable superiority. Ridge's selected alpha is the lower search-grid endpoint, consistent with almost no regularization; it is not proof of an unconstrained optimum. Alpha scales differ between Ridge and Lasso because their optimization objectives use different normalizations.

These are synthetic educational data, not evidence that changing one factor causes a real student's score to change. The high R² should be interpreted in that context.

## Attribution
Dataset: Nikhil, Kaggle user `nikhil7280`.
https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression

The publisher describes the data as synthetic and permits sharing and use in the dataset description. The original CSV is unchanged; cleaning and modeling steps are documented in the notebook.

Methods: https://scikit-learn.org/stable/modules/linear_model.html and https://scikit-learn.org/stable/common_pitfalls.html
