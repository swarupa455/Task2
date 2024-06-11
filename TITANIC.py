import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Configure visualization
sns.set(style='whitegrid')
plt.style.use('ggplot')
train_data = pd.read_csv('/content/train.csv')
test_data = pd.read_csv('/content/test.csv')
print(train_data.head())
print(test_data.head())
print(train_data.columns)
print(test_data.columns)
train_data.shape
test_data.shape
print(train_data.tail())
print(test_data.tail())
# Check for missing values
print(train_data.isnull().sum())
# Check for missing values
print(test_data.isnull().sum())
# Fill missing 'Age' with median
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)

# Fill missing 'Embarked' with the most frequent value
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

# Drop 'Cabin' due to too many missing values
train_data.drop(columns='Cabin', inplace=True)

# Check for any remaining missing values
print(train_data.isnull().sum())
# Fill missing 'Age' with median
test_data['Age'].fillna(train_data['Age'].median(), inplace=True)

# Fill missing 'Embarked' with the most frequent value
test_data['Fare'].fillna(test_data['Fare'].mode()[0], inplace=True)

# Drop 'Cabin' due to too many missing values
test_data.drop(columns='Cabin', inplace=True)

# Check for any remaining missing values
print(test_data.isnull().sum())
# 4.1 Overview of the Data
print(train_data.info())
print(train_data.describe())
# 4.2 Distribution of Numerical Features
numerical_features = ['Age', 'Fare', 'SibSp', 'Parch']
train_data[numerical_features].hist(bins=15, figsize=(15, 6), layout=(2, 2))
# 4.3 Countplot for Categorical Features
categorical_features = ['Survived', 'Pclass', 'Sex', 'Embarked']
for feature in categorical_features:
    sns.countplot(x=feature, data=train_data)
    plt.title(f'Count of {feature}')
    plt.show()
# 4.4 Correlation Heatmap
# Select only numeric columns for correlation matrix
numeric_columns = train_data.select_dtypes(include=[np.number]).columns.tolist()
plt.figure(figsize=(10, 8))
sns.heatmap(train_data[numeric_columns].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
# 4.5 Survival Rate by Sex
sns.barplot(x='Sex', y='Survived', data=train_data)
plt.title('Survival Rate by Sex')
plt.show()
# 4.6 Survival Rate by Pclass
sns.barplot(x='Pclass', y='Survived', data=train_data)
plt.title('Survival Rate by Pclass')
plt.show()
# 4.7 Survival Rate by Embarked
sns.barplot(x='Embarked', y='Survived', data=train_data)
plt.title('Survival Rate by Embarked')
plt.show()
# 4.8 Age Distribution by Survival Status
sns.histplot(data=train_data, x='Age', hue='Survived', multiple='stack')
plt.title('Age Distribution by Survival Status')
plt.show()
# 4.9 Pairplot for Selected Features
selected_features = ['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
sns.pairplot(train_data[selected_features], hue='Survived', diag_kind='kde')
plt.show()
# 4.10 Boxplot of Age by Pclass and Survival
plt.figure(figsize=(10, 6))
sns.boxplot(x='Pclass', y='Age', hue='Survived', data=train_data)
plt.title('Boxplot of Age by Pclass and Survival')
plt.show()
