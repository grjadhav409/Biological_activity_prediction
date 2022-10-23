import os
import warnings
import joblib
from flask import Response
from trainingModel import trainModel
from utils import get_parameters

warnings.filterwarnings("ignore")
from rdkit_utils import smiles_dataset
from utils import save_dataset
import pandas as pd

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

df = pd.read_csv('datasets/ChEMBL_original_dataset.csv')

def trainRouteClient(df):
    try:
        # path = 'datasets/ChEMBL_original_dataset.csv'

        df = df

        trainModelObj = trainModel()  # object initialization
        trainModelObj.trainingModel(df)  # training the model for the files in the table

    except ValueError:

        return Response("Error Occurred! %s" % ValueError)

    except KeyError:

        return Response("Error Occurred! %s" % KeyError)

    except Exception as e:

        return Response("Error Occurred! %s" % e)
    return Response("Training successfull!!")


trainRouteClient(df)


def PredictRouteClient():
    try:
        path = 'screening_base/drugs_smiles.csv'

        model_load = joblib.load('models/SVR/SVR.sav')

        # for drugs

        database2 = pd.read_csv(path)

        dic = get_parameters(path='./settings/fp_settings.json', print_dict=False)
        database_fp = smiles_dataset(dataset_df=database2, smiles_loc='smiles',
                                     fp_radius=dic.get("fp_radius"), fp_bits=dic.get("fp_bits"))

        screen_result2 = model_load.predict(database_fp)
        screen_result_fp2 = pd.DataFrame({'Predictive Results': screen_result2})
        predictions_on_zinc15_drugs = pd.concat([database2, screen_result_fp2], axis=1)
        predictions_on_zinc15_drugs_new = predictions_on_zinc15_drugs.sort_values(by=["Predictive Results"],
                                                                                  ascending=False)

        save_dataset(predictions_on_zinc15_drugs_new.head(10), path='results/', file_name='drugs_result', idx=False)

    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)

# PredictRouteClient()
