# Data Science Salary Estimator: Project Overview 
* Created a tool that estimates data science salaries (MAE ~ $ 11K) to help data scientists negotiate their income when they get a job.
* Engineered features from the text of each job description to quantify the value companies put on python, excel, aws, and spark. 
* Optimized Lasso, Xgboost and Light GBM using optuna to reach the best model.  

## Code and Resources Used 
**Python Version:** 3.10  
**Packages:** pandas, numpy, sklearn, matplotlib, xgboost, lgb, joblib   
**Data source:** https://github.com/PlayingNumbers/ds_salary_proj 

## Data Structure
Dataset consist of 1000 job postings from glassdoor.com. With each job, we got the following:
*	Job title
*	Salary Estimate
*	Job Description
*	Rating
*	Company 
*	Location
*	Company Headquarters 
*	Company Size
*	Company Founded Date
*	Type of Ownership 
*	Industry
*	Sector
*	Revenue
*	Competitors 

## Data Cleaning
I needed to clean it up so that it was usable for our model. I made the following changes and created the following variables:

*	Parsed numeric data out of salary 
*	Made columns for employer provided salary and hourly wages 
*	Removed rows without salary 
*	Parsed rating out of company text 
*	Made a new column for company state 
*	Added a column for if the job was at the company’s headquarters 
*	Transformed founded date into age of company 
*	Made columns for if different skills were listed in the job description:
    * Python  
    * R  
    * Excel  
    * SQL 
*	Column for simplified job title and Seniority 
*	Column for description length
*	Parse number of employees
*	Parse average revenue amount

## EDA
I looked at the distributions of the data and the value counts for the various categorical variables. Below are a few highlights from the pivot tables. 

![alt text](https://github.com/HalyshAnton/ds_salary/blob/main/salary_by_job_title.PNG "Salary by Position")
![alt text](https://github.com/HalyshAnton/ds_salary/blob/main/positions_by_state.png "Job Opportunities by State")
![alt text](https://github.com/HalyshAnton/ds_salary/blob/main/correlation_visual.png "Correlations")

## Model Building 

First, I transformed data using sklearn pipeline. I also split the data into train, validation and tests sets with 60%/20%/20%.   

I tried three different models and evaluated them using different metrics(R2, MAE, MSE). I choose R2 as the main metric for this problem 

I tried three different models:
*	**Lasso Regression** – Baseline for the model with a lot of featurse
*	**XGBoost Regression** – Because of the sparse data from the many categorical variables, I thought a treebased algorithm would be effective.
*	**Light Gradient Boosting** – Again, with the sparsity associated with the data, I thought that this would be a good fit. 

## Model performance
The XGBoost model outperformed the other approaches on the validation set. 
*	**Lasso Regression** – R2: 0.52
*	**XGBoost Regression** – R2: 0.72
*	**Light Gradient Boosting** – R2: 0.61

However, the difference between the results for train and validation sets is much larger for xgboost than for lgbm, indicating a poorer robustness to outliers. Therefore, I have chosen lgbm as the final model.

For hyperparameter tuning I have used optuna. After 1000 trials I gave reeived folowing results
* r2:  0.7501
* MAE:  10.4381
* MSE:  307.2425
* RMSE:  17.5283
