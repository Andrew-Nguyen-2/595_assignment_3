import os.path
import pandas as pd
import glob
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.python.keras.utils.np_utils import to_categorical
from imblearn.over_sampling import SMOTE

evaluation_metric = "macro"
smote = SMOTE(sampling_strategy="minority")
do_smote = False


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


def load_data_gender():
    csv_files = glob.glob(os.path.join("Datasets_Healthy_Older_People/S1_Dataset", "*"))
    csv_files += glob.glob(os.path.join("Datasets_Healthy_Older_People/S2_Dataset", "*"))

    male = []
    female = []

    for i in csv_files:
        if 'M' in i:
            male.append(i)
        else:
            female.append(i)

    X_male = []
    Y_male = []

    for i in male:
        data_tmp = pd.read_csv(i)
        X_tmp = data_tmp.iloc[:, 0:8].values
        Y_tmp = data_tmp.iloc[:, 8].values
        X_male.append(X_tmp)
        Y_male.append(Y_tmp)

    X_female = []
    Y_female = []

    for i in female:
        data_tmp = pd.read_csv(i)
        X_tmp = data_tmp.iloc[:, 0:8].values
        Y_tmp = data_tmp.iloc[:, 8].values
        X_female.append(X_tmp)
        Y_female.append(Y_tmp)

    X_male_new = []
    for i in X_male:
        for j in i:
            X_male_new.append(j)

    X_female_new = []
    for i in X_female:
        for j in i:
            X_female_new.append(j)

    Y_male_new = []
    for i in Y_male:
        for j in i:
            Y_male_new.append(j)

    Y_female_new = []
    for i in Y_female:
        for j in i:
            Y_female_new.append(j)

    return np.array(X_male_new), np.array(Y_male_new), np.array(X_female_new), np.array(Y_female_new)


def majority_class_classifier(Y_train, Y_test):
    classes = np.bincount(Y_train)
    majority_class = np.argmax(classes)

    Y_prediction = np.clip(Y_test, majority_class, majority_class)

    precision = precision_score(Y_test, Y_prediction, average=evaluation_metric, zero_division=1)
    recall = recall_score(Y_test, Y_prediction, average=evaluation_metric, zero_division=1)
    f1 = f1_score(Y_test, Y_prediction, average=evaluation_metric, zero_division=1)

    return precision, recall, f1


def simple_baseline_classifier(Y_test):
    Y_prediction = np.random.randint(1, 5, size=(np.shape(Y_test)))

    precision = precision_score(Y_test, Y_prediction, average=evaluation_metric, zero_division=1)
    recall = recall_score(Y_test, Y_prediction, average=evaluation_metric, zero_division=1)
    f1 = f1_score(Y_test, Y_prediction, average=evaluation_metric, zero_division=1)

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

    precision = precision_score(Y_test_one_label, predictions, average=evaluation_metric, zero_division=1)
    recall = recall_score(Y_test_one_label, predictions, average=evaluation_metric, zero_division=1)
    f1 = f1_score(Y_test_one_label, predictions, average=evaluation_metric, zero_division=1)

    return precision, recall, f1


def classify_by_gender():

    print("----------------------Gender----------------------")

    X_male, Y_male, X_female, Y_female = load_data_gender()

    X_male_train, X_male_test, Y_male_train, Y_male_test = train_test_split(X_male, Y_male, test_size=0.2)
    X_female_train, X_female_test, Y_female_train, Y_female_test = train_test_split(X_female, Y_female, test_size=0.2)

    precision_male_maj, recall_male_maj, f1_male_maj = majority_class_classifier(Y_male_train, Y_male_test)
    precision_female_maj, recall_female_maj, f1_female_maj = majority_class_classifier(Y_female_train, Y_female_test)

    precision_male_rand, recall_male_rand, f1_male_rand = simple_baseline_classifier(Y_male_test)
    precision_female_rand, recall_female_rand, f1_female_rand = simple_baseline_classifier(Y_female_test)

    # apply smote
    if do_smote:
        X_male, Y_male = apply_smote(X_male, Y_male)
        X_female, Y_female = apply_smote(X_female, Y_female)

    Y_male_encoded = to_categorical(Y_male)
    Y_female_encoded = to_categorical(Y_female)

    X_male_train, X_male_test, Y_male_train, Y_male_test = train_test_split(X_male, Y_male_encoded, test_size=0.2)
    X_female_train, X_female_test, Y_female_train, Y_female_test = train_test_split(X_female, Y_female_encoded, test_size=0.2)

    X_male_train, X_male_val, Y_male_train, Y_male_val = train_test_split(X_male_train, Y_male_train, test_size=0.2)
    X_female_train, X_female_val, Y_female_train, Y_female_val = train_test_split(X_female_train, Y_female_train, test_size=0.2)

    precision_male, recall_male, f1_male = neural_network_classifier(
        X_male_train, X_male_test, Y_male_train, Y_male_test, X_male_val, Y_male_val
    )
    precision_female, recall_female, f1_female = neural_network_classifier(
        X_female_train, X_female_test, Y_female_train, Y_female_test, X_female_val, Y_female_val
    )

    print("                      Majority Class Classifier")
    print("                                Male")
    print("precision male:", precision_male_maj, "recall male:", recall_male_maj, "f1 male:", f1_male_maj)
    print("                               Female")
    print("precision female:", precision_female_maj, "recall female:", recall_female_maj, "f1 female:", f1_female_maj)
    print("")

    print("                        Random Class Classifier")
    print("                                Male")
    print("precision male:", precision_male_rand, "recall male:", recall_male_rand, "f1 male:", f1_male_rand)
    print("                               Female")
    print("precision female:", precision_female_rand, "recall female:", recall_female_rand, "f1 female:", f1_female_rand)
    print("")

    print("               3-Layer Densely Connected Neural Network")
    print("                                Male")
    print("precision male:", precision_male, "recall male:", recall_male, "f1 male:", f1_male)
    print("                                Female")
    print("precision female:", precision_female, "recall female:", recall_female, "f1 female:", f1_female)

    print("----------------------Gender----------------------")


def classify_by_room():
    print("----------------------Room-----------------------")
    X_S1, Y_S1, X_S2, Y_S2 = load_data()

    X_S1_train, X_S1_test, Y_S1_train, Y_S1_test = train_test_split(X_S1, Y_S1, test_size=0.2)
    X_S2_train, X_S2_test, Y_S2_train, Y_S2_test = train_test_split(X_S2, Y_S2, test_size=0.2)

    precision_S1_maj, recall_S1_maj, f1_S1_maj = majority_class_classifier(Y_S1_train, Y_S1_test)
    precision_S2_maj, recall_S2_maj, f1_S2_maj = majority_class_classifier(Y_S1_train, Y_S1_test)

    precision_S1_rand, recall_S1_rand, f1_S1_rand = simple_baseline_classifier(Y_S1_test)
    precision_S2_rand, recall_S2_rand, f1_S2_rand = simple_baseline_classifier(Y_S2_test)

    # apply smote
    if do_smote:
        X_S1, Y_S1 = apply_smote(X_S1, Y_S1)
        X_S2, Y_S2 = apply_smote(X_S2, Y_S2)

    Y_S1_encoded = to_categorical(Y_S1)
    Y_S2_encoded = to_categorical(Y_S2)

    X_S1_train, X_S1_test, Y_S1_train, Y_S1_test = train_test_split(X_S1, Y_S1_encoded, test_size=0.2)
    X_S1_train, X_S1_val, Y_S1_train, Y_S1_val = train_test_split(X_S1_train, Y_S1_train, test_size=0.2)

    X_S2_train, X_S2_test, Y_S2_train, Y_S2_test = train_test_split(X_S2, Y_S2_encoded, test_size=0.2)
    X_S2_train, X_S2_val, Y_S2_train, Y_S2_val = train_test_split(X_S2_train, Y_S2_train, test_size=0.2)

    precision_S1_nn, recall_S1_nn, f1_S1_nn = neural_network_classifier(
        X_S1_train, X_S1_test, Y_S1_train, Y_S1_test, X_S1_val, Y_S1_val
    )
    precision_S2_nn, recall_S2_nn, f1_S2_nn = neural_network_classifier(
        X_S2_train, X_S2_test, Y_S2_train, Y_S2_test, X_S2_val, Y_S2_val
    )

    print("                      Majority Class Classifier")
    print("                                S1")
    print("precision:", precision_S1_maj, "recall:", recall_S1_maj, "f1 S1:", f1_S1_maj)
    print("                                S2")
    print("precision:", precision_S2_maj, "recall:", recall_S2_maj, "f1:", f1_S2_maj)
    print("")

    print("                        Random Class Classifier")
    print("                                S1")
    print("precision:", precision_S1_rand, "recall:", recall_S1_rand, "f1:", f1_S1_rand)
    print("                                S2")
    print("precision:", precision_S2_rand, "recall:", recall_S2_rand, "f1:", f1_S2_rand)
    print("")

    print("               3-Layer Densely Connected Neural Network")
    print("                                S1")
    print("precision:", precision_S1_nn, "recall:", recall_S1_nn, "f1:", f1_S1_nn)
    print("                                S2")
    print("precision:", precision_S2_nn, "recall:", recall_S2_nn, "f1:", f1_S2_nn)

    print("----------------------Room-----------------------")


def classify_as_one():
    print("----------------------Both----------------------")
    X_S1, Y_S1, X_S2, Y_S2 = load_data()
    X = np.concatenate((X_S1, X_S2))
    Y = np.concatenate((Y_S1, Y_S2))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    precision_maj, recall_maj, f1_maj = majority_class_classifier(Y_train, Y_test)

    precision_rand, recall_rand, f1_rand = simple_baseline_classifier(Y_test)

    # apply smote
    if do_smote:
        X, Y = apply_smote(X, Y)

    Y_encoded = to_categorical(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_encoded, test_size=0.2)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2)

    nn_precision, nn_recall, nn_f1 = neural_network_classifier(
        X_train, X_test, Y_train, Y_test, X_val, Y_val
    )

    print("                      Majority Class Classifier")
    print("precision:", precision_maj, "recall:", recall_maj, "f1:", f1_maj)
    print("")

    print("                        Random Class Classifier")
    print("precision:", precision_rand, "recall:", recall_rand, "f1:", f1_rand)
    print("")

    print("               3-Layer Densely Connected Neural Network")
    print("precision:", nn_precision, "recall:", nn_recall, "f1:", nn_f1)

    print("----------------------Both----------------------")


def apply_smote(x, y):
    x_smote, y_smote = smote.fit_resample(x, y)

    return x_smote, y_smote


if __name__ == "__main__":
    # classify_by_gender()
    # classify_by_room()
    # classify_as_one()

    X_1, Y_1, X_2, Y_2 = load_data()
    print("len X1:", len(X_1), "len Y1:", len(Y_1), "Class 1:", np.bincount(Y_1))
    print("len X2:", len(X_2), "len Y2:", len(Y_2), "Class 2:", np.bincount(Y_2))
    X_1, Y_1 = apply_smote(X_1, Y_1)
    X_2, Y_2 = apply_smote(X_2, Y_2)
    print("len X1:", len(X_1), "len Y1:", len(Y_1), "Class 1:", np.bincount(Y_1))
    print("len X2:", len(X_2), "len Y2:", len(Y_2), "Class 2:", np.bincount(Y_2))
