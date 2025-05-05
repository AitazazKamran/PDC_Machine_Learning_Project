import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set the style for plots
plt.style.use('ggplot')
sns.set(style="whitegrid")

print("======= EXTENSIVE EXPLORATORY DATA ANALYSIS =======")

# Load the dataset
print("\nLoading dataset...")
data = pd.read_csv('pdc_dataset_with_target.csv')

# Basic dataset information
print("\n1. DATASET OVERVIEW")
print(f"Shape: {data.shape}")
print(f"Features: {data.columns.tolist()}")
print("\nFeature types:")
print(data.dtypes)

print("\nSample data:")
print(data.head())

print("\nBasic statistics:")
print(data.describe().T)

# Target variable analysis
print("\n2. TARGET VARIABLE ANALYSIS")
target_counts = data['target'].value_counts()
print(f"Target distribution:\n{target_counts}")
print(f"Target distribution (%):\n{target_counts / len(data) * 100}")

# Visualize target distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='target', data=data, palette='Blues')
plt.title('Target Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.savefig('target_distribution.png')
print("Saved target_distribution.png")

# Missing values analysis
print("\n3. MISSING VALUES ANALYSIS")
missing_values = data.isnull().sum()
missing_pct = (missing_values / len(data)) * 100
missing_df = pd.DataFrame({
    'Count': missing_values,
    'Percentage': missing_pct
})
print(missing_df[missing_df['Count'] > 0])

# Visualize missing values
plt.figure(figsize=(12, 6))
plt.title('Missing Values Heatmap')
sns.heatmap(data.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.tight_layout()
plt.savefig('missing_values_heatmap.png')
print("Saved missing_values_heatmap.png")

# Identify categorical and numerical features
categorical_features = data.select_dtypes(include=['object']).columns.tolist()
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_features.remove('target')  # Remove target from feature list

print(f"\nCategorical features: {categorical_features}")
print(f"Numerical features: {numerical_features}")

# 4. Categorical features analysis
print("\n4. CATEGORICAL FEATURES ANALYSIS")
for feature in categorical_features:
    print(f"\nAnalysis for {feature}:")
    # Value counts
    counts = data[feature].value_counts()
    print(f"Value counts:\n{counts}")
    
    # Relationship with target
    cross_tab = pd.crosstab(data[feature], data['target'])
    print(f"\nCross-tabulation with target:\n{cross_tab}")
    
    # Calculate percentage of target=1 for each category
    target_pct = pd.crosstab(data[feature], data['target'], normalize='index')
    print(f"\nPercentage of target=1 by category:\n{target_pct[1].sort_values(ascending=False)}")
    
    # Visualize
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.countplot(y=feature, data=data, palette='viridis')
    plt.title(f'Count of {feature}')
    plt.tight_layout()
    
    plt.subplot(1, 2, 2)
    sns.barplot(y=feature, x='target', data=data, palette='viridis')
    plt.title(f'Average Target by {feature}')
    plt.tight_layout()
    plt.savefig(f'categorical_{feature}_analysis.png')
    print(f"Saved categorical_{feature}_analysis.png")

# 5. Numerical features analysis
print("\n5. NUMERICAL FEATURES ANALYSIS")
for feature in numerical_features:
    print(f"\nAnalysis for {feature}:")
    
    # Basic statistics
    stats = data[feature].describe()
    print(f"Statistics:\n{stats}")
    
    # Correlation with target
    correlation = data[feature].corr(data['target'])
    print(f"Correlation with target: {correlation:.4f}")
    
    # Distribution visualization
    plt.figure(figsize=(15, 5))
    
    # Histogram
    plt.subplot(1, 3, 1)
    sns.histplot(data=data, x=feature, kde=True)
    plt.title(f'Distribution of {feature}')
    
    # Box plot
    plt.subplot(1, 3, 2)
    sns.boxplot(x='target', y=feature, data=data)
    plt.title(f'Box Plot of {feature} by Target')
    
    # Distribution by target
    plt.subplot(1, 3, 3)
    sns.histplot(data=data, x=feature, hue='target', kde=True, element='step')
    plt.title(f'Distribution of {feature} by Target')
    
    plt.tight_layout()
    plt.savefig(f'numerical_{feature}_analysis.png')
    print(f"Saved numerical_{feature}_analysis.png")

# 6. Correlation analysis
print("\n6. CORRELATION ANALYSIS")
# Correlation between numerical features
corr_matrix = data[numerical_features + ['target']].corr()
print("Correlation matrix:")
print(corr_matrix['target'].sort_values(ascending=False))

# Visualize correlation matrix
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
print("Saved correlation_heatmap.png")

# 7. Outlier analysis
print("\n7. OUTLIER ANALYSIS")
for feature in numerical_features:
    # Calculate IQR
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define outliers
    outliers = data[(data[feature] < (Q1 - 1.5 * IQR)) | (data[feature] > (Q3 + 1.5 * IQR))]
    outlier_pct = len(outliers) / len(data) * 100
    
    if len(outliers) > 0:
        print(f"{feature}: {len(outliers)} outliers ({outlier_pct:.2f}%)")
        
        # Check if outliers have different target distribution
        overall_target_mean = data['target'].mean()
        outlier_target_mean = outliers['target'].mean()
        print(f"  Overall target mean: {overall_target_mean:.4f}")
        print(f"  Outlier target mean: {outlier_target_mean:.4f}")

# Create box plots for all numerical features to visualize outliers
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features):
    plt.subplot(3, 2, i+1 if i < 5 else 5)
    sns.boxplot(x=feature, data=data)
    plt.title(f'Box Plot - {feature}')
plt.tight_layout()
plt.savefig('outlier_boxplots.png')
print("Saved outlier_boxplots.png")

# 8. Feature distribution analysis
print("\n8. FEATURE DISTRIBUTION ANALYSIS")
# Create a figure with histogram and QQ plot for each numerical feature
for feature in numerical_features:
    plt.figure(figsize=(15, 5))
    
    # Histogram
    plt.subplot(1, 3, 1)
    sns.histplot(data[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    
    # QQ Plot
    plt.subplot(1, 3, 2)
    stats.probplot(data[feature].dropna(), plot=plt)
    plt.title(f'QQ Plot of {feature}')
    
    # Log transform (if positive)
    plt.subplot(1, 3, 3)
    if data[feature].min() > 0:
        sns.histplot(np.log1p(data[feature]), kde=True)
        plt.title(f'Log Distribution of {feature}')
    else:
        plt.text(0.5, 0.5, "Cannot log transform\n(contains zero or negative values)", 
                 horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'distribution_{feature}.png')
    print(f"Saved distribution_{feature}.png")

# 9. Missing values relationship with target
print("\n9. MISSING VALUES RELATIONSHIP WITH TARGET")
for column in data.columns:
    if data[column].isnull().sum() > 0:
        # Create a new column indicating if the value is missing
        data[f'{column}_missing'] = data[column].isnull().astype(int)
        
        # Calculate mean target value for missing and non-missing
        missing_target_mean = data[data[column].isnull()]['target'].mean()
        non_missing_target_mean = data[~data[column].isnull()]['target'].mean()
        
        print(f"{column}:")
        print(f"  Target mean (missing values): {missing_target_mean:.4f}")
        print(f"  Target mean (non-missing): {non_missing_target_mean:.4f}")
        
        # Visualize
        plt.figure(figsize=(8, 6))
        sns.barplot(x=f'{column}_missing', y='target', data=data)
        plt.title(f'Target Mean by {column} Missing Status')
        plt.xlabel(f'{column} is Missing')
        plt.ylabel('Target Mean')
        plt.xticks([0, 1], ['Not Missing', 'Missing'])
        plt.savefig(f'missing_target_{column}.png')
        print(f"Saved missing_target_{column}.png")

# 10. Pair plots for most correlated features
print("\n10. PAIR PLOTS FOR MOST CORRELATED FEATURES")
# Get absolute correlations with target
target_corrs = corr_matrix['target'].abs().sort_values(ascending=False)
# Get top 5 most correlated features (excluding target itself)
top_features = target_corrs[1:6].index.tolist()

# Create pair plot
plt.figure(figsize=(12, 10))
sns.pairplot(data, vars=top_features, hue='target', diag_kind='kde', plot_kws={'alpha': 0.6})
plt.suptitle('Pair Plot of Top Correlated Features', y=1.02)
plt.savefig('pairplot_top_features.png')
print("Saved pairplot_top_features.png")

# 11. Summary of findings
print("\n11. SUMMARY OF KEY FINDINGS")
print("a. Target distribution:")
print(f"   - Class 0: {target_counts[0]} ({target_counts[0]/len(data)*100:.2f}%)")
print(f"   - Class 1: {target_counts[1]} ({target_counts[1]/len(data)*100:.2f}%)")

print("\nb. Missing values:")
missing_summary = missing_df[missing_df['Count'] > 0]
if not missing_summary.empty:
    for idx, row in missing_summary.iterrows():
        print(f"   - {idx}: {row['Count']} missing values ({row['Percentage']:.2f}%)")
else:
    print("   - No missing values found")

print("\nc. Top correlated features with target:")
for feature, corr in target_corrs[1:6].items():
    print(f"   - {feature}: {corr:.4f}")

print("\nd. Categorical feature insights:")
for feature in categorical_features:
    top_category = pd.crosstab(data[feature], data['target'], normalize='index')[1].sort_values(ascending=False).index[0]
    print(f"   - {feature}: Highest target rate in category '{top_category}'")

print("\ne. Most significant outlier features:")
for feature in numerical_features:
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    outliers = data[(data[feature] < (Q1 - 1.5 * IQR)) | (data[feature] > (Q3 + 1.5 * IQR))]
    if len(outliers) > 0:
        outlier_pct = len(outliers) / len(data) * 100
        if outlier_pct > 1:  # Only report if significant
            print(f"   - {feature}: {outlier_pct:.2f}% outliers")

print("\nEDA completed successfully. All visualizations have been saved as PNG files.") 