
# BigMart Sales Prediction Project

## Introduction
This project aims to predict the sales of products in BigMart stores based on various attributes of the products and stores. The dataset contains information about 1559 products across 10 stores in different cities for the year 2013.

## Problem Statement
The task is to create a regression model that can accurately predict the sales of each product in each store. This model can help BigMart understand the factors that influence sales and make informed decisions to increase sales.

## Dataset
The dataset contains 8,523 records with 12 attributes:

- `Item_Identifier`: Unique product ID
- `Item_Weight`: Weight of the product
- `Item_Fat_Content`: Concentration of fat in the product
- `Item_Visibility`: Percentage of total display area of all similar products in a store
- `Item_Type`: Product category
- `Item_MRP`: Maximum Retail Price for a product
- `Outlet_Identifier`: Store ID
- `Outlet_Establishment_Year`: Year in which the store was established
- `Outlet_Size`: Size of the store (Area Size Category)
- `Outlet_Location_Type`: City tier (Location Type)
- `Outlet_Type`: Type of outlet (Grocery store or supermarket)
- `Item_Outlet_Sales`: Sales of the product in the specific outlet (Target Variable)

## Analysis and Modeling Process
1. **Data Exploration**: Explored the dataset, checked for missing values, and handled them appropriately. Visualized the data to understand distributions and relationships.
   
2. **Feature Engineering**: Encoded categorical variables, dropped irrelevant columns, and created new features like `Outlet_Age` from `Outlet_Establishment_Year`.

3. **Machine Learning Models**: Trained three regression models - Linear Regression, Random Forest Regressor, and Lasso Regressor. Evaluated model performance using metrics like MAE, MSE, R^2 score, and Cross-Validation Score.

4. **Conclusion**: Summarized findings, highlighted best-performing models, and provided insights into the dataset and modeling process.

## Results
- Linear Regression and Lasso Regressor performed the best in most categories, with Random Forest Regressor showing slightly lower performance.
- Item MRP was found to be the most significant predictor of sales, with higher MRPs generally leading to higher sales.
- Feature engineering, including encoding categorical variables and creating new features, improved model performance.

## Next Steps
- Further tuning and optimization of models using techniques like Grid Search.
- Exploration of additional features or data sources that could improve model performance.
- Deployment of the best-performing model for real-time predictions in BigMart stores.

## Repository Structure
- `data/`: Directory containing the dataset files.
- `notebooks/`: Jupyter notebooks for data exploration, preprocessing, and modeling.
- `README.md`: Project overview, instructions, and documentation.
- `results/`: Directory containing model performance metrics and visualization results.

## Dependencies
- Python 3
- Jupyter Notebook
- Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## Usage
1. Clone the repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Navigate to the `notebooks/` directory and run the Jupyter notebooks to explore the data and train the models.
4. Use the trained models to make predictions on new data or deploy them for real-time predictions.

## Credits
- This project is based on the [BigMart Sales Data](https://www.kaggle.com/brijbhushannanda1979/bigmart-sales-data) dataset from Kaggle.
- Inspiration and guidance for data analysis and machine learning techniques were taken from various online resources, including Kaggle kernels, blog posts, and documentation.

