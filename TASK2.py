# Titanic Dataset: Data Cleaning and Exploratory Data Analysis (EDA)
# Task: Perform data cleaning and exploratory data analysis (EDA) on a dataset of your choice, such as the Titanic dataset from Kaggle.
# Explore the relationship between variables and identify patterns and trends in the data.

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Section 1: Load and Inspect Data
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic_df = pd.read_csv(url)

# Display the first few rows of the dataset
print(titanic_df.head())

# Display summary information about the dataset
print(titanic_df.info())

# Display descriptive statistics for the dataset
print(titanic_df.describe(include='all'))

# Section 2: Data Cleaning
# Check for missing values
missing_values = titanic_df.isnull().sum()
print("Missing values before cleaning:\n", missing_values)

# Fill missing 'Age' values with the median age
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)

# Fill missing 'Embarked' values with the mode
titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0], inplace=True)

# Drop the 'Cabin' column due to a high number of missing values
titanic_df.drop(columns=['Cabin'], inplace=True)

# Check for missing values after cleaning
missing_values_after_cleaning = titanic_df.isnull().sum()
print("Missing values after cleaning:\n", missing_values_after_cleaning)

# Convert 'Survived' and 'Pclass' to categorical data types
titanic_df['Survived'] = titanic_df['Survived'].astype('category')
titanic_df['Pclass'] = titanic_df['Pclass'].astype('category')

# Display summary information about the cleaned dataset
print(titanic_df.info())

# Section 3: Exploratory Data Analysis (EDA) - Visualizations
# Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(titanic_df['Age'], kde=True, bins=30)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Sex Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Sex', data=titanic_df)
plt.title('Sex Distribution')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()

# Section 4: Survival Analysis
# Survival Rate by Passenger Class
plt.figure(figsize=(8, 6))
sns.countplot(x='Pclass', hue='Survived', data=titanic_df)
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.show()

# Survival Rate by Sex
plt.figure(figsize=(6, 4))
sns.countplot(x='Sex', hue='Survived', data=titanic_df)
plt.title('Survival Rate by Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()

# Section 5: Multivariate Analysis
# Survival Rate by Sex and Passenger Class
sns.catplot(x='Pclass', hue='Sex', col='Survived', kind='count', data=titanic_df, height=6, aspect=1.2)
plt.subplots_adjust(top=0.85)
plt.suptitle('Survival Rate by Sex and Passenger Class', size=16)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
numeric_df = titanic_df.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()

# Section 6: Final Data Summary
# Display summary information about the cleaned dataset
print(titanic_df.info())

# Display descriptive statistics for the cleaned dataset
print(titanic_df.describe(include='all'))

# Save the cleaned dataset to a CSV file
titanic_df.to_csv('cleaned_titanic_dataset.csv', index=False)
