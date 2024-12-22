
# Global Unemployment Data Analysis and Predictive Modeling

## Overview
This project consists of two main components:

1. **Exploratory Data Analysis (EDA)**: Analyzing global unemployment data from 2014 to 2024 to identify trends, patterns, and relationships.
2. **Predictive Modeling**: Building machine learning models to forecast future unemployment rates.

The dataset includes unemployment rates segmented by country, gender, and age group, providing valuable insights for researchers, policymakers, and stakeholders.


## Key Features

### 1. Exploratory Data Analysis (EDA)
- **Objective**: Analyze global unemployment trends and provide actionable insights.
- **Key Highlights**:
  - Handle missing data, remove duplicate rows, and detect outliers.
  - Visualize the data using bar charts, pie charts, KDE plots, box plots, and correlation heatmaps.
  - Perform statistical analysis including mean, median, standard deviation, skewness, kurtosis, and central moments.
  - Identify gender disparities and geographic trends.
  - Analyze the impact of the COVID-19 pandemic on unemployment.

### 2. Predictive Modeling
- **Objective**: Use machine learning techniques to predict unemployment rates.
- **Key Steps**:
  - **Data Preprocessing**: Reshape the data, create new features (e.g., year-over-year unemployment changes), and handle missing values.
  - **Model Training**: Train a Random Forest Regressor using one-hot encoding for categorical variables.
  - **Evaluation Metrics**:
    - Mean Absolute Error (MAE): ~0.847
    - Mean Squared Error (MSE): ~1.732
    - R-squared (R²): ~0.89
  - **Results Visualization**: Compare true vs. predicted unemployment rates using line plots.


## Dataset
- **Source**: Kaggle
- **Description**: Unemployment rates from 2014–2024, segmented by country, gender, and age group.
- **Size**: ~1200 rows, 16 columns.

