'''
PURPOSE OF THIS SCRIPT IS JUST DO DOWNLOAD THE DATA AND EXPORT IT TO A CSV FILE
'''
from scipy.io import arff
import urllib.request
import io
import pandas as pd
import numpy as np
import os

##### POWER SUPPLY DATA STREAM #####
header = np.array(["mainGrid", "otherGrids", "hourOfDay"])
path = os.getcwd() + str("\\powerSupplyStream.csv")

def main():
    urlPowerSupply = "https://www.cse.fau.edu/~xqzhu/Stream/powersupply.arff"
    ftpstream = urllib.request.urlopen(urlPowerSupply)
    data, meta = arff.loadarff(io.StringIO(ftpstream.read().decode('utf-8')))

    data_mainGrid = []
    data_otherGrids = []
    data_hourOfDay = []
    for col in data:
        data_mainGrid.append(col[0])
        data_otherGrids.append(col[1])
        data_hourOfDay.append(str(col[2]).replace("b\'", "").replace("\'", ""))

    df = pd.DataFrame(np.array((data_mainGrid, data_otherGrids, data_hourOfDay)).T, columns=header)
    df.to_csv(path)

if __name__ == "__main__":
    main()