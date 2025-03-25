# Boston_Housing_Price_Prediction
Boston Housing Price Prediction is a common machine learning project that involves predicting house prices based on various features such as crime rate, number of rooms, tax rates, and more. It is typically done using the Boston Housing Dataset, which is available in sklearn.datasets.

# Technology Used:
Programming Language: Python
Libraries/Frameworks:
            scikit-learn (for model development)
            Pandas, NumPy (for data manipulation)
            Matplotlib, Seaborn (for data visualization)
GUI: Tkinter

# Functional Requirements
1.Data Preprocessing
Input: Raw Boston Housing dataset.
Functionality:
o Handle missing data.
o Normalize or standardize features for effective model performance.
o Split the dataset into training and testing sets.
2.Model Building
Input: Preprocessed data.
Functionality:
o Implement Linear Regression and Ridge Regression models.
o Tune hyperparameters for the Ridge Regression model.
o Train the model on the training set.
3.Model Evaluation
Input: Model predictions and actual target values from the test set.
Functionality:
o Calculate and display the Mean Squared Error (MSE) and R-squared scores.
o Visualize predictions against actual values for analysis.
4.Data Visualization
Input: Preprocessed data and model outputs.
Functionality:
o Create correlation heatmaps to identify significant features.
o Plot predictions vs. actual prices.

# Regression Models
1. Linear Regression - This model will predict housing prices by estimating the linear relationship between the dependent variable (price) and the independent variables (features).
2. Ridge Regression - A regularized version of Linear Regression that includes a penalty term to prevent overfitting, providing more robust predictions.
