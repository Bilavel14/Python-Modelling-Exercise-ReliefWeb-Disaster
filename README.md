This exercise aimed to explore disaster data, clean it, and build a predictive model for classifying disaster types using Python, focusing on both exploratory analysis and model building. 
Disaster Data Modeling Project - Summary
1. Data Loading and Inspection
Objective: Load disaster data and inspect its structure.

Code:
import pandas as pd

# Load the dataset
reliefweb_disaster_data = pd.read_csv("reliefweb-disasters-list.csv")

# Inspecting the first few rows
reliefweb_disaster_data.head()
2. Data Cleaning
Objective: Clean and prepare the dataset for analysis.

Extract relevant columns (country, disaster type, and year).

Handle missing values, format the year column, and filter irrelevant entries.
# Drop rows with NaN in critical columns
reliefweb_disaster_data = reliefweb_disaster_data.dropna(subset=['primary_country-name', 'primary_type-name'])

# Extract year from 'date-event'
reliefweb_disaster_data['date-event'] = pd.to_datetime(reliefweb_disaster_data['date-event'], errors='coerce')
reliefweb_disaster_data['year'] = reliefweb_disaster_data['date-event'].dt.year

# Create a subset for modeling
modeling_data = reliefweb_disaster_data[['primary_country-name', 'year', 'primary_type-name']].dropna()

# Save the cleaned dataset
modeling_data.to_csv("modeling_data_subset.csv", index=False)
 Exploratory Data Analysis (EDA)
Objective: Perform exploratory data analysis on the cleaned data.

Identify the distribution of disaster types, countries most affected, and analyze trends over time.

Code Example for EDA:
import matplotlib.pyplot as plt

# Count the frequency of disaster types
disaster_count = modeling_data['primary_type-name'].value_counts()
disaster_count.plot(kind='bar', title='Disaster Type Frequency')
plt.show()
 Feature Engineering for Modelling
Objective: Prepare data for modeling.

Convert categorical data (like disaster type) into numerical format.

Encode categorical columns using LabelEncoder for machine learning algorithms.
from sklearn.preprocessing import LabelEncoder

# Label encode categorical columns
le_country = LabelEncoder()
modeling_data['country_encoded'] = le_country.fit_transform(modeling_data['primary_country-name'])

le_type = LabelEncoder()
modeling_data['type_encoded'] = le_type.fit_transform(modeling_data['primary_type-name'])
Model Building (Classification)
Objective: Train a classification model to predict disaster types based on country and year.

Split the data into training and testing sets.

Use Random Forest Classifier to predict the disaster type.
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Define features and target variable
X = modeling_data[['country_encoded', 'year']]
y = modeling_data['type_encoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le_type.inverse_transform(np.unique(y_test))))
Model Evaluation
Objective: Assess the model's performance.

The classification report provides insights into precision, recall, and F1-score for each disaster type.

Output Example:
precision    recall  f1-score   support

Cold Wave       0.00      0.00      0.00         6
Drought         0.00      0.00      0.00        11
Earthquake      0.33      0.22      0.27         9
Epidemic        0.26      0.25      0.25        36
Flood           0.42      0.54      0.47        76
Improving the Model
Objective: Improve model performance by trying different models, handling class imbalance, or including additional features.

Consider using:

Cross-validation for better evaluation.

SMOTE (Synthetic Minority Over-sampling Technique) to handle imbalanced classes.

Additional features such as regional data, population data, or weather-related factors.
Conclusion:
The model demonstrates that country and year are insufficient features for predicting disaster types effectively due to class imbalance and limited feature sets.

A better approach could involve collecting more data, handling imbalanced classes, and considering other features like population or geographical characteristics.

Data Cleaning: Essential to handle missing data and convert datetime columns into usable formats.

Feature Engineering: Label encoding categorical columns is crucial before using them in machine learning models.

Modeling: Random Forest was used as a classifier, but better features and data preprocessing are needed for better performance.

Model Evaluation: Understanding precision, recall, and F1-score for different disaster types is vital for interpreting model performance.
