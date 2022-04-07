import os.path
import pandas as pd
import glob
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
import numpy as np
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


def simple_baseline_classifier(Y_test):
    Y_prediction = np.random.randint(1, 5, size=(np.shape(Y_test)))

    precision = precision_score(Y_test, Y_prediction, average="macro", zero_division=1)
    recall = recall_score(Y_test, Y_prediction, average="macro", zero_division=1)
    f1 = f1_score(Y_test, Y_prediction, average="macro", zero_division=1)

    return precision, recall, f1


def neural_network_classifier(X_train, X_test, Y_train, Y_test, X_val, Y_val):
    model = Sequential()
    model.add(Dense(25, input_shape=(8,), activation='gelu'))
    model.add(Dropout(0.8))
    model.add(Dense(10, activation='tanh'))
    model.add(Dropout(0.8))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=100, batch_size=75000)
    y_pred = model.predict(X_test)
    predictions = np.argmax(y_pred, axis=1)

    Y_test_one_label = []
    for i in range(len(Y_test)):
        Y_test_one_label.append(np.where(Y_test[i] == 1)[0][0])

    precision = precision_score(Y_test_one_label, predictions, average="macro", zero_division=1)
    recall = recall_score(Y_test_one_label, predictions, average="macro", zero_division=1)
    f1 = f1_score(Y_test_one_label, predictions, average="macro", zero_division=1)

    return precision, recall, f1


if __name__ == "__main__":
    X_S1, Y_S1, X_S2, Y_S2 = load_data()

    # X_S1_train, X_S1_test, Y_S1_train, Y_S1_test = train_test_split(X_S1, Y_S1, test_size=0.2)
    # X_S2_train, X_S2_test, Y_S2_train, Y_S2_test = train_test_split(X_S2, Y_S2, test_size=0.2)
    #
    # majority_precision_S1, majority_recall_S1, majority_f1_S1 = majority_class_classifier(Y_S1_train, Y_S1_test)
    # majority_precision_S2, majority_recall_S2, majority_f1_S2 = majority_class_classifier(Y_S1_train, Y_S1_test)

    # print("Majority Class Classifier")
    # print("          S1")
    # print("precision:", majority_precision_S1, "\nrecall:", majority_recall_S1, "\nf1 S1:", majority_f1_S1)
    # print("          S2")
    # print("precision:", majority_precision_S2, "\nrecall:", majority_recall_S2, "\nf1:", majority_f1_S2)
    #
    # print("")
    #
    # random_precision_S1, random_recall_S1, random_f1_S1 = simple_baseline_classifier(Y_S1_test)
    # random_precision_S2, random_recall_S2, random_f1_S2 = simple_baseline_classifier(Y_S2_test)
    #
    # print("Random Class Classifier")
    # print("          S1")
    # print("precision:", random_precision_S1, "\nrecall:", random_recall_S1, "\nf1:", random_f1_S1)
    # print("          S2")
    # print("precision:", random_precision_S2, "\nrecall:", random_recall_S2, "\nf1:", random_f1_S2)

    X = np.concatenate((X_S1, X_S2))
    Y = np.concatenate((Y_S1, Y_S2))
    Y_S1_encoded = to_categorical(Y_S1)
    Y_S2_encoded = to_categorical(Y_S2)
    Y_encoded = to_categorical(Y)

    X_S1_train, X_S1_test, Y_S1_train, Y_S1_test = train_test_split(X_S1, Y_S1_encoded, test_size=0.2)
    X_S1_train, X_S1_val, Y_S1_train, Y_S1_val = train_test_split(X_S1_train, Y_S1_train, test_size=0.2)

    X_S2_train, X_S2_test, Y_S2_train, Y_S2_test = train_test_split(X_S2, Y_S2_encoded, test_size=0.2)
    X_S2_train, X_S2_val, Y_S2_train, Y_S2_val = train_test_split(X_S2_train, Y_S2_train, test_size=0.2)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_encoded, test_size=0.2)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2)

    nn_precision_S1, nn_recall_S1, nn_f1_S1 = neural_network_classifier(
        X_S1_train, X_S1_test, Y_S1_train, Y_S1_test, X_S1_val, Y_S1_val
    )
    nn_precision_S2, nn_recall_S2, nn_f1_S2 = neural_network_classifier(
        X_S2_train, X_S2_test, Y_S2_train, Y_S2_test, X_S2_val, Y_S2_val
    )
    nn_precision_both, nn_recall_both, nn_f1_both = neural_network_classifier(
        X_train, X_test, Y_train, Y_test, X_val, Y_val
    )

    print("3-Layer Densely Connected Neural Network")
    print("                                S1")
    print("precision:", nn_precision_S1, "recall:", nn_recall_S1, "f1:", nn_f1_S1)
    print("                                S2")
    print("precision:", nn_precision_S2, "recall:", nn_recall_S2, "f1:", nn_f1_S2)
    print("                                Both")
    print("precision:", nn_precision_both, "recall:", nn_recall_both, "f1:", nn_f1_both)

