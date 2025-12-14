# house-price-prediction-ml

End-to-end machine learning project to predict house prices using regression models (Linear, Ridge, Lasso) with EDA, feature engineering, and model evaluation using scikit-learn.

## Dataset

The project uses the Boston Housing dataset, which contains information about housing characteristics in Boston suburbs. The target variable is `MEDV`, representing the median value of owner-occupied homes in thousands of dollars.

The dataset consists of 506 samples with 13 numerical features, including indicators related to crime rate, property size, number of rooms, environmental factors, and accessibility to highways. House prices range from 5 to 50 (in $1000 units), with most values concentrated between approximately 15 and 30.

The dataset is moderately noisy and contains correlated features (e.g., `RAD` and `TAX`), making it suitable for studying linear regression and the impact of regularization techniques such as Ridge and Lasso regression.

Dataset source: https://www.kaggle.com/datasets/fedesoriano/the-boston-houseprice-data

## Conclusion

This project explored house price prediction as a supervised regression problem using the Boston Housing dataset. Linear Regression was used as a baseline model, followed by Ridge and Lasso regression to study the impact of regularization.

The Linear Regression model achieved an RMSE of approximately 5.05, indicating that predictions deviate from actual house prices by about $5,000 on average. Ridge Regression provided a marginal improvement (RMSE ≈ 5.051), suggesting mild multicollinearity among features and improved stability through coefficient shrinkage. Lasso Regression performed worse (RMSE ≈ 5.52), likely due to excessive feature elimination and increased bias.

Overall, the results indicate that the dataset does not suffer from severe overfitting, and regularization offers limited but meaningful improvement. Ridge Regression was the most suitable model for this problem, balancing bias and variance while retaining all informative features.

Further improvements would require richer feature engineering, non-linear models, or additional data, as the remaining error is likely due to inherent noise in the dataset.
