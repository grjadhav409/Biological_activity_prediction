import pandas as pd
import numpy as np
from operator import index
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
import mols2grid
from rdkit.Chem import Draw


def getstr_drg(filepath):
    df1 = pd.read_csv(filepath, delimiter=',')
    df2 = df1.sort_values(by=["Predicted values"], ascending=False)
    df3 = df2.head(5).drop_duplicates()
    smiles_list = df3["smiles"].to_list()
    mols = []
    for i in smiles_list:
        mols.append(Chem.MolFromSmiles(i))

    df3["rdkit_object"] = mols
    df3["smiles_list"] = smiles_list
    img = Draw.MolsToGridImage(mols,molsPerRow=3)
    img.save('images/cdk2_molgrid.o.png')
    print(img)


# best performing drugs
getstr_drg('results/drugs_result.csv')

