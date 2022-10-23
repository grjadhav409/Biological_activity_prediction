### calculates fingerprints
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import time
import os
from sklearn.svm import SVR
import joblib

import warnings

warnings.filterwarnings("ignore")

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

from rdkit_utils import smiles_dataset
from utils import save_dataset, get_parameters


class pred_validation():
    model_load = joblib.load('models/SVR/SVR.sav')
    database = pd.read_csv('screening_base/drugs_smiles.csv')

    dic = get_parameters(path='./settings/fp_settings.json', print_dict=False)
    screen_database = smiles_dataset(dataset_df=database, smiles_loc='smiles',
                                     fp_radius=dic.get("fp_radius"), fp_bits=dic.get("fp_bits"))
    screen_result = model_load.predict(screen_database)

    screen_result_fp = pd.DataFrame({'Predictive Results': screen_result})
    database_result = pd.concat([database, screen_result_fp], axis=1)

    threshold_7_drug = database_result[database_result['Predictive Results'] > 7]

    original_dataset = pd.read_csv('./datasets/all_structures.csv')
    de_threshold_7_drug = threshold_7_drug
    for smile in original_dataset['Smiles']:
        for new_structure in threshold_7_drug['smiles']:
            if smile == new_structure:
                index = threshold_7_drug[threshold_7_drug['smiles'] == smile].index[0]
                print('overlap found at position: {:01d}'.format(index))
                de_threshold_7_drug = de_threshold_7_drug.drop(index=index, axis=0)
            else:
                pass

    # save_dataset(threshold_7_drug, path='results/', file_name='threshold_7_drug', idx=False)  # more than 7 pchembl
    # save_dataset(de_threshold_7_drug, path='results/', file_name='de_threshold_7_drug',idx=False)  # more than 7 excluding mols from training data
