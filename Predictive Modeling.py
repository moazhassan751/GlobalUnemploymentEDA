import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load Data
data = pd.read_csv("global_unemployment_data.csv")

# Step 2: Data Preprocessing
# Reshape data (convert from wide to long format)
data_long = pd.melt(data, id_vars=['country_name', 'indicator_name', 'sex', 'age_group', 'age_categories'], 
                    var_name='Year', value_name='Unemployment Rate')

# Convert 'Year' to integer and 'Unemployment Rate' to float
data_long['Year'] = data_long['Year'].astype(int)
data_long['Unemployment Rate'] = data_long['Unemployment Rate'].astype(float)

# Feature Engineering (example: year-over-year unemployment change)
data_long['Unemployment Change'] = data_long.groupby(['country_name', 'age_group'])['Unemployment Rate'].diff()

# Drop missing values
data_long.dropna(subset=['Unemployment Rate', 'Unemployment Change'], inplace=True)

# Step 3: Model Selection and Training
# Select relevant features (e.g., year, age_group, sex, and unemployment change)
X = data_long[['Year', 'age_group', 'sex', 'Unemployment Change']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variables into dummy variables
y = data_long['Unemployment Rate']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Step 4: Model Evaluation
y_pred = rf.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display metrics
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared: {r2}")

# Step 5: Display True vs Predicted Values in Tabular Format
comparison_df = pd.DataFrame({
    'True Values': y_test.values,
    'Predicted Values': y_pred
})

print("\nTrue vs Predicted Unemployment Rates:")
print(comparison_df.head(20))  # Displaying the first 20 rows

# Step 6: Visualization - Predicted vs True Values
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='True Values', color='blue', linestyle='-', marker='o', alpha=0.7)
plt.plot(y_pred, label='Predicted Values', color='red', linestyle='--', marker='x', alpha=0.7)
plt.title('True vs Predicted Unemployment Rates')
plt.xlabel('Index')
plt.ylabel('Unemployment Rate')
plt.legend()
plt.show()
