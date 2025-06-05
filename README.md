# SCT_DS_2
🧾 Introduction
The sinking of the RMS Titanic in the early hours of April 15, 1912, remains one of the most infamous and studied maritime disasters in modern history. Deemed "unsinkable" before her maiden voyage, the Titanic was the largest ship afloat at the time and a marvel of early 20th-century engineering. However, after hitting an iceberg in the North Atlantic Ocean, the vessel sank, resulting in the deaths of more than 1,500 of the approximately 2,224 passengers and crew on board. This tragedy not only shocked the world but also revealed the limitations of technology and the human cost of overconfidence.

From a data science perspective, the Titanic disaster presents a unique opportunity to analyze historical passenger data and uncover patterns related to survival. The Titanic dataset, provided by Kaggle, has become a classic case study for data analysis, machine learning, and feature engineering. It contains detailed information about each passenger, such as their age, gender, class, fare paid, family aboard, and survival outcome. As such, the dataset offers a structured and multi-dimensional view of a real-world scenario in which human behavior, social norms, and chance all played a role.

This project focuses on performing Exploratory Data Analysis (EDA) on the Titanic dataset to better understand the factors that influenced passenger survival. Rather than jumping directly into predictive modeling, EDA allows us to gain familiarity with the dataset, identify trends, explore correlations, and visually communicate findings. It is a critical first step in the data science process, one that bridges raw data and actionable insight.

At the core of our analysis is the question: What factors most influenced whether a passenger survived the Titanic disaster? Was it age or gender? Wealth or class? Did traveling with family increase or decrease the chance of survival? By exploring the dataset, we aim to answer such questions using empirical evidence.

The dataset includes the following key columns:
PassengerId: Unique identifier for each passenger.
Survived: Binary value (0 = No, 1 = Yes) indicating survival.
Pclass: Ticket class (1st, 2nd, or 3rd).
Name: Full name, which may contain useful titles.
Sex: Gender of the passenger.
Age: Age in years.
SibSp: Number of siblings/spouses aboard.
Parch: Number of parents/children aboard.
Ticket: Ticket number.
Fare: Fare paid for the ticket.
Cabin: Cabin number (heavily missing).
Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

🧠 Why Titanic?
The Titanic dataset is frequently used in data science not only because of its historical relevance but also because it contains a mix of categorical and numerical features, missing data, and a clearly defined binary outcome. These elements make it an ideal playground for practicing data wrangling, visualization, and interpretation.
It’s also interesting from a social science point of view. The decisions made during the evacuation reflected societal structures of the time—such as the "women and children first" policy, or how socioeconomic status influenced access to lifeboats. This makes it possible to extract real-life meaning from the data, rather than analyzing abstract or synthetic datasets.

🛠 Tools and Technologies Used

This project leverages various tools from the Python data science ecosystem:
Pandas for data loading, cleaning, and transformation.
NumPy for handling numerical operations.
Matplotlib and Seaborn for basic and advanced plotting, including bar plots, histograms, and heatmaps.
Plotly Express for interactive and 3D visualizations.
Google Colab as the coding environment, allowing for convenient, browser-based development without the need for local installation.

📌 Project Objectives
The core objectives of this project are as follows:

1. Load and explore the Titanic dataset.
    -Understand the structure, shape, and types of features.
    -Preview initial rows to gain intuition about data composition.

2. Handle missing data and clean the dataset.
    -Fill missing values for age and embarked location.
    -Drop or ignore columns with excessive missing data (e.g., Cabin).
    -Encode categorical variables such as sex and embarkation port.

3. Perform univariate and bivariate analysis.
   -Study the distribution of variables like age, fare, and survival.
   -Explore how survival varies across different classes and genders.

4. Visualize patterns using static and interactive plots.
   -Use bar plots, box plots, and distribution curves.
   -Create a 3D scatter plot to visualize Age vs. Fare vs. Class, colored by survival and symbolized by gender.

5. Identify insights and patterns.
   -Determine which variables correlate most strongly with survival.
   -Provide interpretations based on social or historical context.

🔍 Initial Observations
At first glance, the dataset shows several notable issues that need to be addressed before meaningful analysis can begin:
-Missing Values: Columns like "Age," "Cabin," and "Embarked" contain missing entries. Especially "Cabin," which is missing for most passengers, may not contribute   useful information and is often dropped.
-Categorical Encoding: Columns like "Sex" and "Embarked" must be converted into numeric values to allow mathematical operations and plotting.
-Outliers: Variables like "Fare" contain extreme values that can distort distributions and need careful handling during visualization.
 By addressing these challenges through thoughtful preprocessing, we prepare the dataset for analysis and visualization, improving the accuracy and clarity of our   findings.

📊 Value of Visualizations
Visualizations play a central role in this project. A well-crafted plot can reveal complex relationships between variables that are not easily seen through raw numbers or summary tables. For instance, plotting survival rates across classes and genders simultaneously allows us to confirm historical accounts of evacuation policies. Similarly, a 3D scatter plot using `Plotly` lets us explore multiple dimensions at once—adding depth and interactivity to our findings.These visual tools are not just aesthetic; they are integral to forming hypotheses, spotting trends, and communicating insights clearly.

Certainly, Aryan! Building upon the comprehensive introduction you've crafted, let's delve into the **Data Cleaning** section of your Titanic Exploratory Data Analysis (EDA) project. This segment is pivotal in ensuring the dataset's integrity and reliability for subsequent analyses.


🧹 Data Cleaning
Data cleaning is a fundamental step in any data analysis project. It involves identifying and rectifying inaccuracies, inconsistencies, and missing values within the dataset. For the Titanic dataset, this process is crucial to ensure that our analyses yield meaningful and accurate insights.

🔍 Assessing Missing Values

Upon initial inspection using `df.info()` and `df.isnull().sum()`, we observe the following missing values:
Age: 177 missing entries
Cabin: 687 missing entries
Embarked: 2 missing entries
The 'Cabin' column has a substantial number of missing values, making it challenging to impute accurately. Therefore, we decide to drop this column from our analysis.

🧠 Handling Missing Data
Age: Given the right-skewed distribution of the 'Age' variable, we opt to fill missing values with the median age to minimize the impact of outliers.
  ```python
  df['Age'].fillna(df['Age'].median(), inplace=True)
  ```

Embarked: With only two missing entries, we fill these with the mode of the 'Embarked' column.
  ```python
  df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
  ```

Cabin: Due to the high percentage of missing values, we drop this column entirely.
  ```python
  df.drop('Cabin', axis=1, inplace=True)
  ```

🔄 Encoding Categorical Variables
Machine learning algorithms require numerical input. Therefore, we convert categorical variables into numerical formats:
Sex: Map 'male' to 0 and 'female' to 1.
  ```python
  df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
  ```

Embarked: Map 'S' to 0, 'C' to 1, and 'Q' to 2.
  ```python
  df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
  ```

🧹 Dropping Irrelevant Features
Certain columns do not contribute to our analysis and may introduce noise:
PassengerId: Serves as a unique identifier with no predictive value.
Name and Ticket: Contain high-cardinality textual data, which are not directly useful in our current analysis.
  ```python
  df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
  ```

✅ Final Dataset Overview
After cleaning, our dataset comprises the following columns:
Survived: Survival status (0 = No, 1 = Yes)
Pclass: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)
Sex: Gender (0 = Male, 1 = Female)
Age: Age in years
SibSp: Number of siblings/spouses aboard
Parch: Number of parents/children aboard
Fare: Ticket fare
Embarked: Port of embarkation (0 = Southampton, 1 = Cherbourg, 2 = Queenstown)

📊 Exploratory Data Analysis (EDA)

With the Titanic dataset now cleaned and structured, we can begin our exploratory data analysis. EDA is essential for summarizing the main characteristics of the data, discovering relationships among features, and generating hypotheses. We use both statistical summaries and visualizations to examine the patterns and trends that may have influenced passenger survival.

🎯 Target Variable: Survival Distribution
We begin by examining the overall survival distribution to understand the class imbalance.
```python
sns.countplot(data=df, x='Survived', palette='pastel')
plt.title('Survival Count')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Number of Passengers')
plt.show()
```

Insight: Around 38% of passengers survived while 62% did not, indicating an imbalanced target variable.

🏷️ Survival by Gender
Gender was a major factor in evacuation policies (“women and children first”). Let’s examine this:
```python
sns.countplot(data=df, x='Sex', hue='Survived', palette='Set2')
plt.title('Survival Count by Gender')
plt.xlabel('Sex (0 = Male, 1 = Female)')
plt.ylabel('Count')
plt.show()
```

Insight: A much higher proportion of females survived compared to males.

🏨 Survival by Passenger Class
Passenger class might reflect access to lifeboats or cabins near deck exits.
```python
sns.countplot(data=df, x='Pclass', hue='Survived', palette='coolwarm')
plt.title('Survival by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.show()
```

Insight: First-class passengers had the highest survival rate, while third-class had the lowest.

📈 Age Distribution and Survival
We analyze the age distribution and how it correlates with survival.
```python
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Age', hue='Survived', bins=30, kde=True, palette='crest')
plt.title('Age Distribution by Survival')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.show()
```

Insight: Children (particularly under 10) had a higher chance of survival. Many adults aged 20–40 did not survive.

💵 Fare vs. Survival
Fare may indicate wealth or class. Let’s explore its effect.
```python
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Survived', y='Fare', palette='Set3')
plt.title('Fare Paid vs Survival')
plt.xlabel('Survived')
plt.ylabel('Fare')
plt.yscale('log')
plt.show()
```

Insight: Survivors generally paid higher fares, especially those in the first class.

🧒 SibSp and Parch
We evaluate family size on board—whether traveling with family improved survival.
```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.countplot(data=df, x='SibSp', hue='Survived', ax=axes[0], palette='magma')
axes[0].set_title('Survival by Siblings/Spouses Aboard')

sns.countplot(data=df, x='Parch', hue='Survived', ax=axes[1], palette='viridis')
axes[1].set_title('Survival by Parents/Children Aboard')

plt.tight_layout()
plt.show()
```

Insight: Passengers with 1–2 family members had better survival odds. Those alone or with large families fared worse.

🌎 Survival by Embarkation Port
Let’s visualize survival rates by port of embarkation.
```python
sns.countplot(data=df, x='Embarked', hue='Survived', palette='Accent')
plt.title('Survival by Port of Embarkation')
plt.xlabel('Embarked (0 = S, 1 = C, 2 = Q)')
plt.ylabel('Count')
plt.show()
```

Insight: Passengers embarking from Cherbourg (C) had the highest survival rate. Southampton (S) had the lowest.

🔗 Correlation Matrix
Let’s calculate correlations between numeric variables.
```python
corr_df = df.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(10, 8))
sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
```

Insight: 'Sex' (female) and 'Fare' positively correlate with survival. 'Pclass' negatively correlates.

🌐 Interactive 3D Visualization
Using `plotly`, we create a 3D scatter plot to visualize age, fare, and class with color-coded survival and gender:
```python
import plotly.express as px

fig = px.scatter_3d(df,
                    x='Age',
                    y='Fare',
                    z='Pclass',
                    color='Survived',
                    symbol='Sex',
                    hover_data=['Sex', 'Fare', 'Embarked'])

fig.update_layout(title='3D Titanic Passenger Visualization')
fig.show()
```

Insight: This interactive 3D plot allows you to explore survival patterns more intuitively across features.
👤 Visualizing Titanic Passenger Layout (Creative Visualization)
While there is no official floor map of Titanic in the dataset, we can creatively illustrate an approximate layout using icons or abstract distribution plots:
import plotly.graph_objects as go

survived_male = df[(df['Survived'] == 1) & (df['Sex'] == 0)].shape[0]
survived_female = df[(df['Survived'] == 1) & (df['Sex'] == 1)].shape[0]
died_male = df[(df['Survived'] == 0) & (df['Sex'] == 0)].shape[0]
died_female = df[(df['Survived'] == 0) & (df['Sex'] == 1)].shape[0]

fig = go.Figure(data=[
    go.Bar(name='Survived Male', x=['Male'], y=[survived_male], marker_color='lightblue'),
    go.Bar(name='Survived Female', x=['Female'], y=[survived_female], marker_color='pink'),
    go.Bar(name='Died Male', x=['Male'], y=[died_male], marker_color='darkblue'),
    go.Bar(name='Died Female', x=['Female'], y=[died_female], marker_color='deeppink')
])
fig.update_layout(barmode='stack', title='Total Passengers by Gender and Survival Status')
fig.show()
Insight: This bar chart presents a quick "snapshot" of how the Titanic's demographic fared in terms of survival, offering a pseudo-layout of male and female survival representation.

✅ Summary of EDA Insights
  Women and children had significantly higher survival rates.
  First-class passengers were more likely to survive.
  Passengers who paid higher fares had better survival odds.
  Smaller families (1–2 members) improved chances of survival.
  Passengers from Cherbourg (port C) had higher survival rates.
  Age and sex had strong correlations with survival.

🧠 Key Takeaways
From our analysis, we observe several compelling patterns:
Gender Bias in Survival: Females had a significantly higher chance of survival, possibly due to evacuation protocols prioritizing women and children.
Class Divide: First-class passengers were much more likely to survive than those in third class.
Age and Family Factor: Children and passengers with small families had better survival odds than solo travelers or those with large families.
Fare as Wealth Indicator: Higher ticket fares generally correlated with higher survival, reinforcing the socioeconomic divide on the ship.
Embarkation Port: Passengers who boarded at Cherbourg showed higher survival, potentially due to their class or cabin location.


🧾 Conclusion
The Titanic dataset is an excellent resource for practicing real-world data cleaning, visualization, and interpretation. Our analysis uncovered key factors—such as gender, passenger class, fare, and family size—that heavily influenced survival rates. Through EDA, we not only explored numerical relationships but also crafted interactive and creative visualizations to make the findings more engaging.
This project demonstrates the importance of data preparation, visual storytelling, and domain understanding in deriving value from raw datasets. In real-world data science, similar approaches are taken before applying machine learning models, where these insights would inform model feature selection.
