# Summary Report: Predicting House Sale Prices (Ames Housing)

## 1. Business Problem and Dataset

Agents, buyers and lenders need a fast, consistent estimate of what a home will sell for. This project predicts `SalePrice` from property attributes. The dataset (`train.csv`) has 1,460 Ames, Iowa sales with 79 features covering size, quality, condition, age, garage, basement and neighbourhood. Prices range from $34,900 to $755,000 (median $163,000). Models were scored on a held-out 20% test set.

## 2. Preprocessing and Feature Engineering

- **Target:** Price is right-skewed (skew 1.88), so I modelled `log(price)` (skew 0.12). Errors become proportional and expensive homes don't dominate.
- **Outliers:** Two very large, cheap homes (`GrLivArea` > 4,000 sq ft, price < $300k) were removed because they distort the size-price relationship.
- **Missing values:** In many categorical columns "NA" means the feature is absent (no pool, no garage), so it became a "None" category. `LotFrontage` and `MasVnrArea` were median-imputed.
- **Encoding:** Quality ratings were mapped to an ordered 0-5 scale, other categoricals were one-hot encoded, and numerics were standardised for linear models.
- **Engineered features:** `TotalSF`, `TotalBath`, `HouseAge`, years since remodel, porch/deck area, and pool/garage/fireplace flags. These combine fragmented columns into what buyers actually weigh.
- **Validation:** All steps sit inside a pipeline with 5-fold cross-validation to avoid leakage.

## 3. Best Model

| Model | Test RMSE | Test MAE | Test R² |
|---|---|---|---|
| Median baseline | $76,742 | n/a | -0.07 |
| Ridge | $20,295 | $14,540 | 0.925 |
| Lasso | $19,865 | $14,191 | 0.929 |
| Random Forest | $21,856 | $15,214 | 0.914 |
| **Gradient Boosting** | **$18,080** | **$13,259** | **0.941** |

**Gradient Boosting** won on the test set, with a typical error of about $13,300 (roughly 8% of the median price). Its sequential shallow trees capture non-linear effects and interactions, while subsampling limits overfitting. In cross-validation it is statistically tied with Ridge and Lasso (log-RMSE 0.1135 vs 0.1127, inside one standard deviation), so the linear models are strong, cheaper, more interpretable alternatives.

## 4. Most Predictive Features

By permutation importance on the test set:

1. **`TotalSF` (total finished area):** the strongest driver, with 0.83 correlation with log price.
2. **`OverallQual` (finish quality, 1-10):** median price rises from $133,000 at quality 5 to $269,750 at quality 8.
3. **`GrLivArea` (above-ground living area):** reinforces that size dominates.

For a real estate professional: anchor valuations on square footage and quality first. Quality is partly controllable, so upgrades that raise finish level likely earn more than minor extras. Neighbourhood ranks lower, probably because it overlaps with quality and size, but medians still range from $88,000 to $315,000, so comparables should be local.

## 5. Next Steps for Production

- **More data:** 1,460 sales is small. Add recent transactions, other markets and external signals (schools, transit, mortgage rates), and retrain regularly.
- **Ensembling:** Stack gradient boosting, Lasso and Random Forest, and add XGBoost/LightGBM.
- **Bayesian optimisation:** Tune hyperparameters with Optuna instead of manual settings.
- **Uncertainty:** Provide prediction intervals (quantile regression or conformal prediction).
- **Monitoring:** Track error drift, audit errors by neighbourhood and price tier, and use SHAP for per-property explanations.
