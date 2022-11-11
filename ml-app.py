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
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from utils import save_dataset, get_parameters
from rdkit_utils import smiles_dataset
import mols2grid
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw


from file_operations import file_methods
import matplotlib.pyplot as plt

# ---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='Property prediction',
                   layout='wide')
# ---------------------------------#
# smiles to structure
def getstr_drg(df1):
    df2 = df1.sort_values(by=["Predicted values"], ascending=False)
    df3 = df2.head(5).drop_duplicates()
    smiles_list = df3["smiles"].to_list()
    mols = []
    for i in smiles_list:
        mols.append(Chem.MolFromSmiles(i))
    df3["rdkit_object"] = mols
    df3["smiles_list"] = smiles_list
    two_decimals = lambda x: f"{x:.2f}"
    return mols2grid.display(df3, mol_col='rdkit_object', subset=["img", 'Predicted values'],
                             transform={"Predicted values": two_decimals})


# ---------------------------------#
# Model building
def build_model(df):
    # descriptor calculations

    train_valObj = train_validation(df, descriptor)
    X, Y = train_valObj.data_cleansing()
    # X = X.iloc[:, :10]

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 / 3)

    st.markdown('**1.2. Data splits**')
    st.write('Training set')
    st.info(X_train.shape)
    st.write('Test set')
    st.info(X_test.shape)

    st.markdown(f"**1.3.'{descriptor}'**:")
    st.write(X_train.head(4))

    # st.info(list(X.columns))

    st.markdown('Target')
    st.write(Y_train.head(4))
    # st.info(Y_train)

    # model building
    rf = SVR(kernel=parameter_kernel,
             gamma=parameter_gamma,
             C=parameter_C
             )
    rf.fit(X_train, Y_train)

    st.subheader('2. Model Performance')
    st.markdown('**2.1. Training set**')
    Y_pred_train = rf.predict(X_train)

    st.write('Coefficient of determination ($R^2$):')
    st.info(r2_score(Y_train, Y_pred_train))

    st.write('Error (MSE):')
    st.info(mean_squared_error(Y_train, Y_pred_train))

    st.markdown('**2.2. Test set**')
    Y_pred_test = rf.predict(X_test)

    st.write('Coefficient of determination ($R^2$):')
    st.info(r2_score(Y_test, Y_pred_test))

    st.write('Error (MSE):')
    st.info(mean_squared_error(Y_test, Y_pred_test))

    st.subheader('3. Model Parameters')
    st.write(rf.get_params())

    # vertual screening
    database2 = pd.read_csv('screening_base/drugs_smiles.csv')
    dic = get_parameters(path='fp_settings.json', print_dict=False)
    database_fp = smiles_dataset(dataset_df=database2, smiles_loc='smiles',
                                 fp_radius=dic.get("fp_radius"), fp_bits=dic.get("fp_bits"))
    # database_fp = database_fp.iloc[:, :10] # for less training time
    screen_result2 = rf.predict(database_fp[:])
    screen_result_fp2 = pd.DataFrame({'Predicted values': screen_result2})
    predictions_on_zinc15 = pd.concat([database2, screen_result_fp2], axis=1)
    predictions_on_zinc15_new = predictions_on_zinc15.sort_values(by=["Predicted values"],
                                                                  ascending=False)
    st.subheader('Virtual screening results on FDA approved Drugs')
    st.write(predictions_on_zinc15_new.head(10))


# ---------------------------------#
st.write("""
# Machine Learning for Biological activity prediction

## IIIT Hyderabad

""")

# ---------------------------------#
# Sidebar - Collects user input features into dataframe

with st.sidebar.header('Upload CSV data from ChEMBL database for Model training :'):
    uploaded_file = st.sidebar.file_uploader("Upload your training CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/JiajunZhou96/ML-for-LSD1/main/datasets/ChEMBL_original_dataset.csv)
""")

# Sidebar
with st.sidebar.header('Molecular Featurization Method :'):
    descriptor = st.sidebar.select_slider(' ',
                                          options=['Morgan fingerprints', 'Mordred descriptors'])
with st.sidebar.subheader('Hyperparameters :'):
    # parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
    # parameter_max_features = st.sidebar.select_slider('Max features (max_features)', options=['auto', 'sqrt', 'log2'])
    # parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
    # parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)
    parameter_kernel = st.sidebar.select_slider('kernel :', options=['rbf', 'sigmoid', 'linear', 'poly'])
    parameter_gamma = st.sidebar.select_slider('gamma :', options=['auto', 'scale'])
    parameter_C = st.sidebar.select_slider('C :', options=[ 10, 1, 20,50,100])

# ---------------------------------#
# Main panel

# Displays the dataset
st.subheader('1. Dataset')

# train
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, delimiter=";")
    df = df[df['Smiles'].notna()]
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df)
    build_model(df)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        df = pd.read_csv(
            "datasets/ChEMBL_original_dataset.csv")  # ,delimiter=";")
        st.markdown('Biological activity against LSD1 dataset used as example from CHEMBL database.')
        st.write(df.head(5))
        build_model(df)
