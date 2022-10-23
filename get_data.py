import pandas as pd
import numpy as np
from chembl_webresource_client.new_client import new_client
import warnings
warnings.filterwarnings("ignore")

def get_data_class(target_name):
    try:
        # Target search
        target = new_client.target
        target_query = target.search(target_name)
        targets0 = pd.DataFrame.from_dict(target_query)
        targets = targets0.loc[(targets0["organism"] == "Homo sapiens") & (
                    (targets0["target_type"] == ("SINGLE PROTEIN")) | (targets0["target_type"] == ("PROTEIN COMPLEX")))]
        selected_targets = targets.target_chembl_id[0:]

        df_dict = {}  # Use a dict to save all df
        for target in selected_targets:
            activity = new_client.activity
            res = activity.filter(target_chembl_id=target).filter(standard_type="IC50")
            df = pd.DataFrame.from_dict(res)
            df_dict[target] = df

        df1 = pd.concat(df_dict, axis=0)
        df2 = df1.reset_index()
        return df2
    except Exception as e:
        print(e)

a = get_data_class("LSD1")
print(a.shape)

