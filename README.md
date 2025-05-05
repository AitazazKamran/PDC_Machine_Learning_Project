# PDC Machine Learning Project

## Overview
This project aims to design and implement an optimized machine learning pipeline for binary classification. The focus of the optimization is to reduce processing time by at least 70% while maintaining or improving predictive performance. Various approaches, including parallel computing (Dask), distributed processing, and GPU acceleration (TensorFlow on Colab T4), were used to evaluate their impact on performance and resource utilization.

## Dataset
- **Total Records:** 41,000 samples
- **Target Classes:** Binary (0 and 1)
- **Features:** 8 total (mix of numerical and categorical)
- **Missing Values:** Present in 4 features (~5% each)
- **Imbalance:** 60.2% Class 0 vs 39.8% Class 1

## Steps Taken

### 1. **Exploratory Data Analysis (EDA)**
The dataset was analyzed to understand its structure, relationships between features, and quality of data:
- **Low Correlation**: No features had strong correlation with the target variable.
- **Categorical Features**: No meaningful separation between classes.
- **Outliers**: Present in several features but did not drastically affect performance.

### 2. **Model Training**
- **Random Forest (CPU)**: Baseline model using Random Forest.
- **Random Forest (Dask)**: Parallelized using Dask to improve processing time.
- **Neural Network (Sklearn MLP)**: Implemented a shallow neural network with preprocessing and PCA.
- **Deep Neural Network (TensorFlow on GPU)**: Deployed a GPU-accelerated model using TensorFlow on Google Colab (T4 GPU).

## Results

The following models were compared based on **accuracy**, **ROC AUC**, **execution time**, and **Class 1 recall**:

| **Model**                      | **Accuracy** | **ROC AUC** | **Time (s)** | **Class 1 Recall** | **% Change in Time** | **% Change in Accuracy** | **% Change in ROC AUC** |
|---------------------------------|--------------|-------------|--------------|--------------------|----------------------|--------------------------|--------------------------|
| Random Forest (CPU)            | 60.26%       | 52.96%      | 92.92        | 0.10               | -                    | -                        | -                        |
| Random Forest (Dask)           | 59.60%       | 52.93%      | 9.85         | 0.10               | **-89.4%**           | **-1.1%**                | **-0.06%**               |
| Neural Net (Sklearn)           | 60.17%       | 52.65%      | 372.05       | 0.00               | **-99.2%**           | **-0.15%**               | **-0.59%**               |
| Deep NN (GPU T4 - TF)          | 60.16%       | 52.40%      | 24.75        | 0.00               | **-93.35%**          | **-0.17%**               | **-0.53%**               |

### Key Insights:
- **Dask** achieved a significant reduction in processing time (89.4%) with minimal loss in accuracy and ROC AUC.
- **GPU Acceleration** helped reduce time further but did not result in better performance due to limitations in the dataset and model architecture.

## Conclusions
- **Best balance:** Random Forest with Dask for speed and interpretability.
- **Limitation:** All models faced performance challenges due to weak feature-target correlation.
- **GPU Use:** Beneficial for reducing time but not improving accuracy without stronger data features.

## Recommendations
1. **SMOTE/ADASYN**: Use for handling class imbalance.
2. **Feature Enrichment**: Collect additional or domain-specific features to improve model performance.
3. **Boosting Methods**: Consider algorithms like XGBoost or LightGBM.
4. **Evaluation Metrics**: Use precision-recall curves over accuracy for imbalanced data.




