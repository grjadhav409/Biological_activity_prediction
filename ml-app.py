import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_boston
from training_Validation_Insertion import train_validation
from best_model_finder import tuner
from application_logging import logger
from file_operations import file_methods
import matplotlib.pyplot as plt

# ---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='Property prediction',
                   layout='wide')


# ---------------------------------#
# Model building
# Model building
def build_model(df):
    train_valObj = train_validation(df)
    X, Y = train_valObj.data_cleansing()
    X = X.iloc[:, :10]

    # X = df.iloc[:, :-1]  # Using all column except for the last column as X
    # Y = df.iloc[:, -1]  # Selecting the last column as Y

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3)

    st.markdown('**1.2. Data splits**')
    st.write('Training set')
    st.info(X_train.shape)
    st.write('Test set')
    st.info(X_test.shape)

    st.markdown('**1.3. Variable details**:')
    st.write('X variable')
    st.info(list(X.columns))
    st.write('Y variable')
    st.info(Y.name)

    # model building

    model_finder = tuner.Model_Finder(open("Training_Logs/ModelTrainingLog.txt", 'a+'), logger.App_Logger())
    best_model_name, rf = model_finder.get_best_model(X_train, Y_train, X_test, Y_test)

    st.write('best_model:')
    st.info(best_model_name)

    # RandomForest = RandomForestRegressor()

    rf.fit(X_train, Y_train)

    st.subheader('2. Model Performance')

    st.markdown('**2.1. Training set**')
    Y_pred_train = rf.predict(X_train)
    st.write('Coefficient of determination ($R^2$):')
    st.info(r2_score(Y_train, Y_pred_train))

    st.write('Error (MSE or MAE):')
    st.info(mean_squared_error(Y_train, Y_pred_train))

    st.markdown('**2.2. Test set**')
    Y_pred_test = rf.predict(X_test)
    st.write('Coefficient of determination ($R^2$):')
    st.info(r2_score(Y_test, Y_pred_test))

    st.write('Error (MSE or MAE):')
    st.info(mean_squared_error(Y_test, Y_pred_test))

    st.subheader('3. Model Parameters')
    st.write(rf.get_params())


# ---------------------------------#
st.write("""
# ML for Molecular Property Prediction 

## IIIT Hyderabad

""")

# ---------------------------------#
# Sidebar - Collects user input features into dataframe

with st.sidebar.header('1. Upload your CSV data:'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/JiajunZhou96/ML-for-LSD1/main/datasets/ChEMBL_original_dataset.csv)
""")

# ---------------------------------#
# Main panel

# Displays the dataset
st.subheader('1. Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df)
    build_model(df)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        # Boston housing dataset
        df = pd.read_csv(
            "https://raw.githubusercontent.com/JiajunZhou96/ML-for-LSD1/main/datasets/ChEMBL_original_dataset.csv",delimiter=";")

        st.markdown('Biological activity against LSD1 dataset used as example from CHEMBL database.')
        st.write(df.head(5))

        build_model(df)
