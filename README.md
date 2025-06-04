Objective:
The goal of this project is to build a machine learning model that predicts the load type of a power system based on historical data. The target variable, "Load_Type", has three categories: Light_Load, Medium_Load, and Maximum_Load. The project involves data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and evaluation.

Dataset Description:
The dataset includes the following features:

Date: Timestamp data taken on the first day of each month

Usage_kWh: Continuous industry energy consumption in kilowatt-hours

Lagging Current Reactive Power: Continuous in kVarh

Leading Current Reactive Power: Continuous in kVarh

CO2: Continuous carbon dioxide concentration in ppm

NSM: Number of seconds from midnight

Load_Type: The target variable with values Light_Load, Medium_Load, or Maximum_Load

Methodology:

Data Preprocessing:

Converted date column to datetime format

Extracted month and year from the date

Handled missing values

Encoded the target labels

Exploratory Data Analysis (EDA):

Analyzed the distribution of features

Checked correlation between variables

Visualized data trends over time

Feature Engineering:

Created new time-based features from the Date column

Scaled or normalized the continuous features

Considered dimensionality reduction or selection if necessary

Model Building:

Trained multiple classification models such as Random Forest, XGBoost, Logistic Regression, and SVM

Performed hyperparameter tuning using GridSearchCV or RandomizedSearchCV

Validation and Testing:

Used the last month of the dataset as the test set

Evaluated model using classification metrics including accuracy, precision, recall, and F1-score

Plotted confusion matrix and feature importance

Results:
Models were compared based on their performance on the test set using all classification metrics. The best model was selected based on balanced performance across all classes.

Project Structure:
The project is organized into folders for data, notebook, saved models, and a README file describing the project.

How to Run the Project:
To run the project, clone the repository, install the required libraries using the requirements.txt file, and open the Jupyter Notebook to view the analysis and model predictions.

Evaluation Metrics Used:

Accuracy

Precision

Recall

F1-score

Confusion matrix
These metrics were used to evaluate how well the model performs on the unseen test set.# E_D_A
