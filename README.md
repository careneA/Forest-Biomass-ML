# Predictive Modeling of Forest Biomass for Carbon Management

**SC 324 — Statistical Machine Learning | Colby College, Spring 2026**  
*Carene Kouassi — Data Science / Economics / Mathematics*

---

## Overview

This project builds a binary classification system to predict whether a land pixel contains forest biomass (carbon presence vs. absence) using remotely sensed environmental data from the USFS Forest Inventory and Analysis (FIA) program.

The motivation is practical: sending field crews to directly measure trees is expensive. A reliable predictive model lets the US Forest Service prioritize field visits, reduce unnecessary surveys, and extend monitoring to remote or inaccessible regions — all using satellite and climate data that is already available at scale.

---

## Dataset

- **Source:** USFS Forest Inventory and Analysis (FIA) program, western United States
- **Unit of observation:** 90 × 90 meter land pixel
- **Response variable:** `CARBON_bin` — Presence (carbon > 0) vs. Absence (carbon = 0)
- **Class distribution:** 83.6% Presence / 16.4% Absence (imbalanced)
- **Predictors:** 21 remotely sensed variables across three categories:

| Category | Variables |
|---|---|
| Vegetation indices | `ndvi`, `evi`, `tcc16` |
| Topography | `elev`, `tri`, `tpi`, `eastness`, `northness` |
| Climate (PRISM) | `ppt`, `tmean`, `tmax`, `tmin`, `tmin01`, `vpdmax`, `tdmean`, `def` |
| Land cover | `tnt` (tree / non-tree classification) |

---

## Methods

Three models were compared, chosen to capture complementary aspects of the data:

### 1. LASSO-Penalized Logistic Regression
- Handles extreme multicollinearity in the climate cluster (`tmean`, `tmax`, `tmin`, `vpdmax`, `tdmean` all r > 0.90)
- λ selected via 10-fold cross-validation using the 1SE rule (λ ≈ 0.018)
- Reduced 21 predictors to **5 key variables**: `tnt`, `tcc16`, `tri`, `ndvi`, `tmax`
- Provides interpretable, parsimonious model

### 2. Support Vector Machine (RBF kernel)
- Captures nonlinear decision boundaries
- Hyperparameters tuned via grid search: Cost = 10, γ = 0.01
- High sensitivity (97.0%) — minimizes missed carbon-positive areas

### 3. Random Forest (500 trees)
- Ensemble of de-correlated decision trees
- Robust to class imbalance and complex interactions
- Variable importance assessed via Mean Decrease in Gini

---

## Results

| Metric | Baseline | LASSO | SVM (RBF) | Random Forest |
|---|---|---|---|---|
| Test Accuracy | 83.6% | 88.4% | 90.8% | **90.8%** |
| Sensitivity | 100% | 95.8% | 97.0% | **97.0%** |
| Specificity | 0.0% | 51.1% | 60.1% | **60.1%** |

All three models significantly outperform the majority-class baseline. The **Random Forest** is the recommended model for operational use — it matches the SVM's accuracy while providing interpretable variable importance scores and robustness to noise.

The **LASSO model** is most useful when interpretability matters: it identifies `tcc16` (canopy cover) and `tnt` (land cover type) as the dominant drivers of carbon presence, with smaller contributions from terrain ruggedness and temperature.

---

## Key Findings

- **Vegetation structure dominates:** `tcc16` is the single most important predictor across all models. Pixels below ~20% canopy cover are strongly associated with carbon absence.
- **Climate multicollinearity is severe:** 6 climate variables have pairwise correlations r > 0.90, making LASSO regularization essential for the parametric approach.
- **Nonlinearity matters:** Carbon presence follows a threshold-like pattern — once canopy cover or NDVI cross a critical level, presence probability rises sharply. Linear models miss this.
- **Sensitivity–specificity tradeoff:** All models prioritize sensitivity (missing carbon areas is more costly than over-predicting). Specificity remains the main challenge, likely due to spectral confusion between forests and grassland/shrubland in NDVI/EVI signals.

---

## Files

| File | Description |
|---|---|
| `final_project.Rmd` | Full R Markdown source — EDA, model fitting, evaluation |
| `final_project.html` | Rendered report with all figures and outputs |
| `SC_324_Final_Project.pdf` | Formatted written paper |

---

## Running the Code

Open `final_project.Rmd` in RStudio and click **Knit**, or run:

```r
rmarkdown::render("final_project.Rmd")
```

**Required R packages:**
```r
install.packages(c("tidyverse", "glmnet", "e1071", "randomForest",
                   "caret", "corrplot", "ggplot2"))
```

---

## Concepts Demonstrated

- Binary classification with class imbalance
- LASSO regularization and variable selection (`glmnet`)
- Support Vector Machines with RBF kernel and hyperparameter tuning
- Random Forest ensemble methods and variable importance
- Cross-validation (10-fold) for model selection
- Sensitivity / specificity tradeoffs in applied ML
- Exploratory data analysis: correlation matrices, boxplots, distributional analysis
- Avoiding data leakage (all EDA on training partition only)

---

*Department of Statistics, Colby College, Waterville ME — Spring 2026*
