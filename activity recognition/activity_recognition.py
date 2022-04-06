import os.path
import pandas as pd
import glob
import numpy as np


def load_data():
    csv_files = glob.glob(os.path.join("Datasets_Healthy_Older_People/S1_Dataset", "*"))
    X_S1 = []
    Y_S1 = []
    for i in csv_files:
        data_tmp = pd.read_csv(i)
        X_tmp = data_tmp.iloc[:, 0:8].values
        Y_tmp = data_tmp.iloc[:, 8].values
        X_S1.append(X_tmp)
        Y_S1.append(Y_tmp)

    X_S2 = []
    Y_S2 = []
    csv_files = glob.glob(os.path.join("Datasets_Healthy_Older_People/S2_Dataset", "*"))
    for i in csv_files:
        data_tmp = pd.read_csv(i)
        X_tmp = data_tmp.iloc[:, 0:8].values
        Y_tmp = data_tmp.iloc[:, 8].values
        X_S2.append(X_tmp)
        Y_S2.append(Y_tmp)

    return X_S1, Y_S1, X_S2, Y_S2


if __name__ == "__main__":
    X_S1, Y_S1, X_S2, Y_S2 = load_data()



