'''
PURPOSE OF THIS SCRIPT IS JUST DO DOWNLOAD THE DATA AND EXPORT IT TO A CSV FILE
'''
from scipy.io import arff
import urllib.request
import io
import pandas as pd
import numpy as np
import os

urlPowerSupply = "https://www.cse.fau.edu/~xqzhu/Stream/powersupply.arff"
ftpstream = urllib.request.urlopen(urlPowerSupply)
data, meta = arff.loadarff(io.StringIO(ftpstream.read().decode('utf-8')))

print(meta)
data_mainGrid = []
data_otherGrids = []
data_hourOfDay = []
for col in data:
    data_mainGrid.append(col[0])
    data_otherGrids.append(col[1])
    data_hourOfDay.append(str(col[2]).replace("b\'", "").replace("\'", ""))

header = {"mainGrid", "otherGrids", "hourOfDay"}
df = pd.DataFrame(np.array((data_mainGrid, data_otherGrids, data_hourOfDay)).T, columns=header)
df.to_csv(os.getcwd() + str("\\powerSupplyStream.csv"))