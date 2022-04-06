import os.path
import pandas as pd
import glob
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.python.keras.utils.np_utils import to_categorical


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

    X_S1_new = []
    for i in X_S1:
        for j in i:
            X_S1_new.append(j)

    X_S2_new = []
    for i in X_S2:
        for j in i:
            X_S2_new.append(j)

    Y_S1_new = []
    for i in Y_S1:
        for j in i:
            Y_S1_new.append(j)

    Y_S2_new = []
    for i in Y_S2:
        for j in i:
            Y_S2_new.append(j)

    return np.array(X_S1_new), np.array(Y_S1_new), np.array(X_S2_new), np.array(Y_S2_new)


def majority_class_classifier(Y_train, Y_test):
    classes = np.bincount(Y_train)
    majority_class = np.argmax(classes)

    Y_prediction = np.clip(Y_test, majority_class, majority_class)

    precision = precision_score(Y_test, Y_prediction, average="macro", zero_division=1)
    recall = recall_score(Y_test, Y_prediction, average="macro", zero_division=1)
    f1 = f1_score(Y_test, Y_prediction, average="macro", zero_division=1)

    return precision, recall, f1


if __name__ == "__main__":
    X_S1, Y_S1, X_S2, Y_S2 = load_data()
    Y_S1_encoded = to_categorical(Y_S1)

    X_S1_train, X_S1_test, Y_S1_train, Y_S1_test = train_test_split(X_S1, Y_S1, test_size=0.2)
    X_S2_train, X_S2_test, Y_S2_train, Y_S2_test = train_test_split(X_S2, Y_S2, test_size=0.2)

    majority_precision_S1, majority_recall_S1, majority_f1_S1 = majority_class_classifier(Y_S1_train, Y_S1_test)
    majority_precision_S2, majority_recall_S2, majority_f1_S2 = majority_class_classifier(Y_S1_train, Y_S1_test)

    print("majority class precision S1:", majority_precision_S1, "\nmajority class recall S1:", majority_recall_S1,
          "\nmajority class f1 S1:", majority_f1_S1)
    print("majority class precision S2:", majority_precision_S2, "\nmajority class recall S2:", majority_recall_S2,
          "\nmajority class f1 S2:", majority_f1_S2)


    # X_S1_train, X_S1_test, Y_S1_train, Y_S1_test = train_test_split(X_S1, Y_S1_encoded, test_size=0.2)
    # X_S1_train, X_S1_val, Y_S1_train, Y_S1_val = train_test_split(X_S1_train, Y_S1_train, test_size=0.2)

    # model = Sequential()
    # model.add(Dense(25, input_shape=(8,), activation='gelu'))
    # model.add(Dense(10, activation='tanh'))
    # model.add(Dense(5, activation='softmax'))
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #
    # model.summary()
    #
    # history = model.fit(X_S1_train, Y_S1_train, validation_data=(X_S1_val, Y_S1_val), epochs=100, batch_size=500)
    # y_pred = model.predict(X_S1_test)
    # predictions = np.argmax(y_pred, axis=1)
    #
    # Y_test_one_label = []
    # for i in range(len(Y_S1_test)):
    #     Y_test_one_label.append(np.where(Y_S1_test[i] == 1)[0][0])
    #
    # precision = precision_score(Y_test_one_label, predictions, average='macro')
    # recall = recall_score(Y_test_one_label, predictions, average='macro')
    # f1 = f1_score(Y_test_one_label, predictions, average='macro')
    # print("precision:", precision, "recall:", recall, "f1:", f1)
