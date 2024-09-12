
# Binance Coin Price Prediction
<img src="https://i.ytimg.com/vi/4BQ-yM-joH4/maxresdefault.jpg">

## Overview

In this project involves predicting the closing price of DOGE Coin (DOGE) using historical market data. The analysis and model building are carried out using a Jupyter notebook, where various machine learning techniques and visualizations are employed.

## Dataset

The dataset contains historical price data for DOGE Coin (DOGE) with the following columns:
- `SNo`: Serial Number
- `Name`: Coin Name
- `Symbol`: Coin Symbol
- `Date`: Date of the data entry
- `High`: Highest price of the coin on that date
- `Low`: Lowest price of the coin on that date
- `Open`: Opening price of the coin on that date
- `Close`: Closing price of the coin on that date
- `Volume`: Trading volume of the coin on that date
- `Marketcap`: Market capitalization of the coin on that date

## Requirements

To run the notebook, you need the following Python libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

You can install the required libraries using pip:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Notebook Details

1. **Data Preparation and Exploration**
   - **Loading Data**: The dataset is loaded into a pandas DataFrame.
   - **Data Cleaning**: Checked for missing values and ensured correct datetime formats.
   - **Feature Selection**: Focused on features: 'Open', 'High', 'Low', 'Volume', and 'Marketcap'.

2. **Modeling**
   - **Splitting Data**: The data is divided into training and test sets.
   - **Feature Scaling**: Applied Min-Max scaling for feature normalization.
   - **Model Training**: Trained a RandomForestRegressor model to predict the 'Close' price.
   - **Evaluation**: Used Mean Squared Error (MSE) to evaluate the model’s performance.

3. **Visualization**
   - **Feature Importance**: Visualized feature importance using a bar chart.
   - **Actual vs Predicted Values**: Plotted to compare actual closing prices with predictions.

4. **Additional Plots**
   - **Pie Chart**: Distribution of different columns or categories.
   - **Heatmap**: Correlation matrix of features.
   - **Boxplot**: Distribution of numerical features.
   - **Confusion Matrix**: (If applicable; generally for classification tasks)
   - **Relplot**: Relationship between features or features and target variable.
   - **Cat Plot**: Aggregated statistics or categorical data.
   - **Joint Grid**: Analysis of joint relationships between feature pairs.

## Results

- **Model Accuracy**: The performance of the RandomForestRegressor is evaluated using Mean Squared Error (MSE).
- **Feature Importance**: Insights into the most influential features in predicting the closing price.
- **Visual Analysis**: Various plots to understand data distribution, feature relationships, and model performance.


## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
The dataset is provided by Binance.
Thanks to the contributors of scikit-learn, pandas, matplotlib, and seaborn for their valuable libraries.

Feel free to adjust the sections based on what is most relevant to your project! :)
