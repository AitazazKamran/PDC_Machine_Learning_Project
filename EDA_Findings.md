# Exploratory Data Analysis Findings

## 1. Dataset Overview
- **Size**: 41,000 records with 8 features (7 predictors, 1 target)
- **Features**: 
  - Numerical: feature_1, feature_2, feature_4, feature_6, feature_7
  - Categorical: feature_3, feature_5
  - Target: binary classification (0/1)

## 2. Target Variable Analysis
- **Distribution**:
  - Class 0: 24,682 samples (60.2%)
  - Class 1: 16,318 samples (39.8%)
- The dataset is **moderately imbalanced** with a 60/40 split

## 3. Missing Values Analysis
- Four features have missing values, all around 5%:
  - feature_1: 2,054 missing values (5.01%)
  - feature_2: 2,050 missing values (5.00%)
  - feature_4: 2,054 missing values (5.01%)
  - feature_7: 2,036 missing values (4.97%)

## 4. Categorical Features Analysis
- **feature_3**: Has three categories (A, B, C) with approximately equal distribution
  - Category A: 13,704 samples (33.4%)
  - Category B: 13,616 samples (33.2%)
  - Category C: 13,680 samples (33.4%)
  - **Target association**: Very similar percentages of target=1 across categories
    - C: 40.17% positive class
    - B: 40.08% positive class
    - A: 39.16% positive class

- **feature_5**: Has two categories (Yes, No) with almost exact 50/50 split
  - Yes: 20,519 samples (50.05%)
  - No: 20,481 samples (49.95%)
  - **Target association**: Nearly identical target distributions
    - Yes: 39.97% positive class
    - No: 39.63% positive class

## 5. Numerical Features Analysis

| Feature | Min | Max | Mean | Std Dev | Correlation with Target |
|---------|-----|-----|------|---------|-------------------------|
| feature_1 | 7.67 | 52.4 | 29.99 | 5.00 | -0.0143 |
| feature_2 | 0.0 | 99.0 | 49.44 | 28.96 | 0.0009 |
| feature_4 | -155.62 | 17,643.4 | 1,092.31 | 1,012.05 | 0.0031 |
| feature_6 | 1.0 | 9.0 | 5.01 | 2.58 | 0.0036 |
| feature_7 | 0.0 | 1,660.8 | 23.15 | 47.56 | 0.0045 |

- **Key insight**: All numerical features show extremely weak correlations with the target variable
- **Highest correlation**: feature_7 with just 0.0045 (essentially no correlation)

## 6. Outlier Analysis
- **feature_1**: 264 outliers (0.64%)
  - Outlier target mean: 0.4167 (vs overall: 0.3980)
- **feature_4**: 658 outliers (1.60%)
  - Outlier target mean: 0.4179 (vs overall: 0.3980)
- **feature_7**: 2,065 outliers (5.04%)
  - Outlier target mean: 0.4015 (vs overall: 0.3980)
- Outliers show slightly higher target means but difference is minimal

## 7. Feature Distribution Analysis
- Most numerical features have fairly normal distributions
- feature_7 shows right-skewed distribution with many outliers

## 8. Critical Finding
The most important discovery is that **none of the features show meaningful correlation with the target variable**. This explains why all models performed poorly regardless of hyperparameter tuning or algorithm selection. Without predictive features, no machine learning model can learn meaningful patterns.


## 9. Conclusion
The extremely weak correlations between features and target suggest that this dataset as currently constituted **may not be suitable for predictive modeling**. No amount of model tuning will significantly improve performance without features that have predictive power for the target variable. 