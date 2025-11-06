Titanic Data Analysis using Python

This project performs Exploratory Data Analysis (EDA) on the famous Titanic dataset to uncover patterns related to passenger survival.
The analysis includes data cleaning, visualization, and survival insights based on gender, age, passenger class, and family size.

📌 Project Overview

The objective of this project is to analyze the Titanic dataset and understand the key factors that influenced passenger survival during the tragic event.
The project uses Python libraries to perform data preprocessing and visualize survival trends.

Columns include:
Passenger age, gender, passenger class, fare, embarked location, family relations, deck information, and survival status (0 = Not Survived, 1 = Survived)

🛠️ Tech Stack & Libraries
Library	Purpose
NumPy	Numerical operations
Pandas	Data cleaning & manipulation
Matplotlib	Data visualization
Seaborn	Statistical plots & analysis

🧹 Data Cleaning Performed

Checked missing values
Converted deck column into object type
Filled missing values:
Age → replaced with median
Embarked → replaced with mode
Deck → filled with "Unknown"
Removed rows with missing embark_town values
Created new feature family_size = sibsp + parch + 1

📊 Visualizations & Insights

The following plots are generated:

🔹 Survival Count by Gender
   Displays how survival probability differs between males and females.

🔹 Passenger Class Distribution by Gender
   Shows distribution of male & female passengers across Pclass (1, 2, 3).

🔹 Age Distribution
   Histogram with KDE to analyze age pattern of passengers on board.

🔹 Correlation Heatmap
   Shows correlation among numerical features and survival.

🔹 Survival Rate

✅ By Gender – Females had significantly higher survival rate
✅ By Passenger Class – 1st class passengers survived more
✅ By Gender & Class Combined – Female 1st class had highest survival
✅ By Family Size – Moderate family size had a positive survival impact

🔹 KDE Plot for Age vs Survival
   Visual representation of ages for survived vs non-survived passengers.

📎 Sample Code Snippet

df["age"].fillna(df["age"].median(), inplace=True)
df["embarked"].fillna(df["embarked"].mode()[0], inplace=True)
df["deck"].fillna("Unknown", inplace=True)
df.dropna(subset=['embark_town'], inplace=True)

Future Enhancements

Apply Machine Learning models (Logistic Regression, Random Forest) for prediction
Add detailed interactive dashboards with Plotly or Power BI
Perform feature engineering for improved insights
