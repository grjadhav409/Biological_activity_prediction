from datetime import datetime
import pandas as pd
import numpy as np
from application_logging import logger
from utils import save_dataset , get_parameters
from rdkit_utils import smiles_dataset

class train_validation:

    def __init__(self,df):
        self.dataset_original = df

        #self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.log_writer = logger.App_Logger()

    def data_cleansing(self):
        try:
            #self.log_writer.log(self.file_object, 'Start of Validation on files for prediction!!')
            # Take out all values that have pChEMBL values
            dataset_v1 = self.dataset_original[self.dataset_original['pChEMBL Value'].notna()]

            # Check out the duplicates and take their mean values
            dataset_v2 = dataset_v1.groupby('Molecule ChEMBL ID').mean()['Standard Value'].reset_index()

            # calculate pChEMBL values
            s_value = dataset_v2['Standard Value'].values
            p_value = np.around(- np.log10(s_value / (10 ** (9))), 2)
            dataset_v2['Calculated pChEMBL'] = p_value.tolist()

            for i in range(0, dataset_v2.shape[0]):
                index = dataset_v2['Molecule ChEMBL ID'][i]
                smile = dataset_v1.loc[dataset_v1['Molecule ChEMBL ID'] == index]['Smiles'].drop_duplicates()
                dataframe = pd.DataFrame(smile)

                if i == 0:
                    concat_df = dataframe
                else:
                    concat_df = pd.concat([concat_df, dataframe], axis=0)

            concat_df = concat_df.reset_index()

            all_structures = pd.concat([dataset_v2, concat_df], axis=1)

            save_dataset(all_structures)

            # change the parameters in .json file
            dic = get_parameters(path='/settings/fp_settings.json', print_dict=False)

            x = smiles_dataset(dataset_df=all_structures, smiles_loc='Smiles',
                           fp_radius=dic.get("fp_radius"), fp_bits=dic.get("fp_bits"))

            y = all_structures['Calculated pChEMBL']

            # change file_name to save as different datasets
            save_dataset(x, file_name=dic.get("dataset_name"), idx=False)
            save_dataset(y, file_name=dic.get("label_name"), idx=False)

            return x,y


        except Exception as e:
            raise e
