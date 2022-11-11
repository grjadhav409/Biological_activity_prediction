import pandas as pd
import numpy as np
from operator import index
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
import mols2grid
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from matplotlib import pyplot as plt
from matplotlib import image as mpimg


ms = [Chem.MolFromSmiles(smi) for smi in ('OCCc1ccn2cnccc12','C1CC1Oc1cc2ccncn2c1','CNC(=O)c1nccc2cccn12')]
img=Draw.MolsToGridImage(ms[:8],molsPerRow=4,subImgSize=(200,200))
print(img)

#mg.save('images/cdk2_molgrid.o.png')