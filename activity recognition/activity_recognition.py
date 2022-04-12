import glob
import os.path
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.utils.np_utils import to_categorical

evaluation_metric = 'macro'
smote = SMOTE(sampling_strategy="not majority")
do_smote = False


def clean_sample(arr):
    out = []

    for i in arr:
        for j in i:
            out.append(j-1)

    return out


def load_data():
    csv_files = glob.glob(os.path.join("Datasets_Healthy_Older_People/S1_Dataset", "*"))
    X_S1 = []
    Y_S1 = []
    for i in csv_files:
        data = pd.read_csv(i)
        X = data.iloc[:, 0:8].values
        Y = data.iloc[:, 8].values
        X_S1.append(X)
        Y_S1.append(Y)

    X_S2 = []
    Y_S2 = []
    csv_files = glob.glob(os.path.join("Datasets_Healthy_Older_People/S2_Dataset", "*"))
    for i in csv_files:
        data = pd.read_csv(i)
        X = data.iloc[:, 0:8].values
        Y = data.iloc[:, 8].values
        X_S2.append(X)
        Y_S2.append(Y)

    X_S1_new = clean_sample(X_S1)
    Y_S1_new = clean_sample(Y_S1)
    X_S2_new = clean_sample(X_S2)
    Y_S2_new = clean_sample(Y_S2)

    return np.array(X_S1_new), np.array(X_S2_new), np.array(Y_S1_new), np.array(Y_S2_new)


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
        data = pd.read_csv(i)
        X = data.iloc[:, 0:8].values
        Y = data.iloc[:, 8].values
        X_male.append(X)
        Y_male.append(Y)

    X_female = []
    Y_female = []

    for i in female:
        data_tmp = pd.read_csv(i)
        X_tmp = data_tmp.iloc[:, 0:8].values
        Y_tmp = data_tmp.iloc[:, 8].values
        X_female.append(X_tmp)
        Y_female.append(Y_tmp)

    X_male_new = clean_sample(X_male)
    Y_male_new = clean_sample(Y_male)
    X_female_new = clean_sample(X_female)
    Y_female_new = clean_sample(Y_female)

    return np.array(X_male_new), np.array(X_female_new), np.array(Y_male_new), np.array(Y_female_new)


def load_data_activity():
    csv_files = glob.glob(os.path.join("Datasets_Healthy_Older_People/S1_Dataset", "*"))
    csv_files += glob.glob(os.path.join("Datasets_Healthy_Older_People/S2_Dataset", "*"))

    all_data = []

    for i in csv_files:
        data = pd.read_csv(i)
        all_data.append(data.iloc[:, 0:9].values)

    cleaned_data = clean_sample(all_data)

    X_sit = []
    Y_sit = []
    X_lay = []
    Y_lay = []
    X_walk = []
    Y_walk = []

    for i in cleaned_data:
        features = i[0:8]
        label = i[8]
        if label == 0 or label == 1:
            X_sit.append(features)
            Y_sit.append(int(label))
        if label == 2:
            X_lay.append(features)
            Y_lay.append(int(label))
        if label == 3:
            X_walk.append(features)
            Y_walk.append(int(label))

    return np.array(X_sit), np.array(X_lay), np.array(X_walk), np.array(Y_sit), np.array(Y_lay), np.array(Y_walk)


def majority_class_classifier(Y_train, Y_test):
    classes = np.bincount(Y_train)
    majority_class = np.argmax(classes)

    # clip changes all the values to majority class
    Y_prediction = np.clip(Y_test, majority_class, majority_class)

    precision = precision_score(Y_test, Y_prediction, average=evaluation_metric, zero_division=1)
    recall = recall_score(Y_test, Y_prediction, average=evaluation_metric, zero_division=1)
    f1 = f1_score(Y_test, Y_prediction, average=evaluation_metric, zero_division=1)

    return precision, recall, f1


def simple_baseline_classifier(X_train, X_test, Y_train, Y_test):
    # average each feature sample
    Y_train = Y_train.reshape((Y_train.shape[0], 1))
    data = np.append(X_train, Y_train, axis=1)

    bed = data[np.where(data[:, 8] == 0)]
    chair = data[np.where(data[:, 8] == 1)]
    lay = data[np.where(data[:, 8] == 2)]
    walk = data[np.where(data[:, 8] == 3)]

    x_bed = bed[:, 0:8]
    x_chair = chair[:, 0:8]
    x_lay = lay[:, 0:8]
    x_walk = walk[:, 0:8]

    # average each sample in each class
    x_bed_samp_avg = np.average(x_bed, axis=1)
    x_chair_samp_avg = np.average(x_chair, axis=1)
    x_lay_samp_avg = np.average(x_lay, axis=1)
    x_walk_samp_avg = np.average(x_walk, axis=1)

    # average all samples in each class
    x_bed_avg = np.average(x_bed_samp_avg)
    x_chair_avg = np.average(x_chair_samp_avg)
    x_lay_avg = np.average(x_lay_samp_avg)
    x_walk_avg = np.average(x_walk_samp_avg)

    X_test_averaged = np.average(X_test, axis=1)
    Y_prediction = []

    estimated_averages = np.array([x_bed_avg, x_chair_avg, x_lay_avg, x_walk_avg])

    for i in range(len(X_test_averaged)):
        value = X_test_averaged[i]
        differences = np.abs(estimated_averages - value)
        pred_label = np.argmin(differences)
        Y_prediction.append(pred_label)

    precision = precision_score(Y_test, Y_prediction, average=evaluation_metric, zero_division=1)
    recall = recall_score(Y_test, Y_prediction, average=evaluation_metric, zero_division=1)
    f1 = f1_score(Y_test, Y_prediction, average=evaluation_metric, zero_division=1)

    return precision, recall, f1


def train_learning_rate(x_train, x_test, y_train, y_test, x_val, y_val):
    learning_rate = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    updated_learning_rates = []
    learning_rate_out = 0
    loss = float("inf")
    f1 = float("-inf")
    for lr in learning_rate:
        model = Sequential()
        model.add(Dense(8, input_shape=(8,), activation='gelu'))
        model.add(Dropout(0.8))
        model.add(Dense(6, activation='tanh'))
        model.add(Dropout(0.8))
        model.add(Dense(4, activation='softmax'))
        optimizer = Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=75000)
        tmp_loss = min(history.history['loss'])
        if tmp_loss < loss:
            loss = tmp_loss
            learning_rate_out = lr
            updated_learning_rates.append(lr)

    return learning_rate_out, updated_learning_rates


def neural_network_classifier(X_train, X_test, Y_train, Y_test, X_val, Y_val):
    # fixed one hot encoding to only be 0-3

    learning_rate, all_learning_rates = train_learning_rate(X_train, X_test, Y_train, Y_test, X_val, Y_val)

    model = Sequential()
    model.add(Dense(8, input_shape=(8,), activation='gelu'))
    model.add(Dropout(0.8))
    model.add(Dense(6, activation='tanh'))
    model.add(Dropout(0.8))
    model.add(Dense(4, activation='softmax'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

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

    return precision, recall, f1, learning_rate, all_learning_rates


def classify_by_gender():

    print("----------------------Gender----------------------")

    X_male, X_female, Y_male, Y_female = load_data_gender()

    X_male_train, X_male_test, Y_male_train, Y_male_test = train_test_split(X_male, Y_male, test_size=0.2)
    X_female_train, X_female_test, Y_female_train, Y_female_test = train_test_split(X_female, Y_female, test_size=0.2)

    precision_male_maj, recall_male_maj, f1_male_maj = majority_class_classifier(Y_male_train, Y_male_test)
    precision_female_maj, recall_female_maj, f1_female_maj = majority_class_classifier(Y_female_train, Y_female_test)

    precision_male_rand, recall_male_rand, f1_male_rand = simple_baseline_classifier(
        X_male_train, X_male_test, Y_male_train, Y_male_test)
    precision_female_rand, recall_female_rand, f1_female_rand = simple_baseline_classifier(
        X_female_train, X_female_test, Y_female_train, Y_female_test)

    X_male, Y_male = under_sample_majority_class(X_male, Y_male)
    X_female, Y_female = under_sample_majority_class(X_female, Y_female)

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

    print("                        Baseline Class Classifier")
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
    X_S1, X_S2, Y_S1, Y_S2 = load_data()

    X_S1_train, X_S1_test, Y_S1_train, Y_S1_test = train_test_split(X_S1, Y_S1, test_size=0.2)
    X_S2_train, X_S2_test, Y_S2_train, Y_S2_test = train_test_split(X_S2, Y_S2, test_size=0.2)

    precision_S1_maj, recall_S1_maj, f1_S1_maj = majority_class_classifier(Y_S1_train, Y_S1_test)
    precision_S2_maj, recall_S2_maj, f1_S2_maj = majority_class_classifier(Y_S1_train, Y_S1_test)

    precision_S1_rand, recall_S1_rand, f1_S1_rand = simple_baseline_classifier(
        X_S1_train, X_S1_test, Y_S1_train, Y_S1_test)
    precision_S2_rand, recall_S2_rand, f1_S2_rand = simple_baseline_classifier(
        X_S2_train, X_S2_test, Y_S2_train, Y_S2_test)

    X_S1, Y_S1 = under_sample_majority_class(X_S1, Y_S1)
    X_S2, Y_S2 = under_sample_majority_class(X_S2, Y_S2)

    # over sample minority classes without smote
    X_S1, Y_S1 = over_sample_minority_classes(X_S1, Y_S1)
    X_S2, Y_S2 = over_sample_minority_classes(X_S2, Y_S2)

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

    print("                        Baseline Class Classifier")
    print("                                S1")
    print("precision:", precision_S1_rand, "recall:", recall_S1_rand, "f1:", f1_S1_rand)
    print("                                S2")
    print("precision:", precision_S2_rand, "recall:", recall_S2_rand, "f1:", f1_S2_rand)
    print("")

    print("                  3-Layer Densely Connected Neural Network")
    print("                                S1")
    print("precision:", precision_S1_nn, "recall:", recall_S1_nn, "f1:", f1_S1_nn)
    print("                                S2")
    print("precision:", precision_S2_nn, "recall:", recall_S2_nn, "f1:", f1_S2_nn)

    print("----------------------Room-----------------------")


def classify_as_one():
    print("----------------------Both----------------------")
    X_S1, X_S2, Y_S1, Y_S2 = load_data()

    X = np.concatenate((X_S1, X_S2))
    Y = np.concatenate((Y_S1, Y_S2))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    precision_maj, recall_maj, f1_maj = majority_class_classifier(Y_train, Y_test)

    precision_avg, recall_avg, f1_avg = simple_baseline_classifier(X_train, X_test, Y_train, Y_test)

    # under sample majority class
    X, Y = under_sample_majority_class(X, Y)

    # over sample minority classes
    X, Y = over_sample_minority_classes(X, Y)

    # apply smote
    if do_smote:
        X, Y = apply_smote(X, Y)

    Y_encoded = to_categorical(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_encoded, test_size=0.2)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2)

    nn_precision, nn_recall, nn_f1, learning_rate, all_learning_rates = neural_network_classifier(
        X_train, X_test, Y_train, Y_test, X_val, Y_val
    )

    print("                      Majority Class Classifier")
    print("precision:", precision_maj, "recall:", recall_maj, "f1:", f1_maj)
    print("")

    print("                        Baseline Class Classifier")
    print("precision:", precision_avg, "recall:", recall_avg, "f1:", f1_avg)
    print("")

    print("                    3-Layer Densely Connected Neural Network")
    print("precision:", nn_precision, "recall:", nn_recall, "f1:", nn_f1)
    print("")
    print("optimal learning rate:", learning_rate, "learning rate updates:", all_learning_rates)

    print("----------------------Both----------------------")


def classify_by_activity():
    X_sit, X_lay, X_walk, Y_sit, Y_lay, Y_walk = load_data_activity()

    X = np.concatenate((X_sit, X_lay, X_walk))
    Y = np.concatenate((Y_sit, Y_lay, Y_walk))

    unique, counts = np.unique(Y, return_counts=True)
    print("before class count:", np.column_stack((unique, counts)))

    if do_smote:
        X, Y = apply_smote(X, Y)
        unique, counts = np.unique(Y, return_counts=True)
        print("after smote class count:", np.column_stack((unique, counts)))

    Y = np.where(Y == 2, 1, Y)

    unique, counts = np.unique(Y, return_counts=True)
    print("after over sample class count:", np.column_stack((unique, counts)))

    Y_encoded = to_categorical(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_encoded, test_size=0.2)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2)

    model = Sequential()
    model.add(Dense(25, input_shape=(8,), activation='gelu'))
    model.add(Dropout(0.8))
    model.add(Dense(10, activation='tanh'))
    model.add(Dropout(0.8))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=100, batch_size=75000)
    y_pred = model.predict(X_test)
    predictions = np.argmax(y_pred, axis=1)

    Y_test_one_label = []
    for i in range(len(Y_test)):
        Y_test_one_label.append(np.where(Y_test[i] == 1)[0][0])

    # check what class and if sitting then determine what they are sitting on
    # find all samples that are predicted to be 1 or 2
    X_pred_sit = np.zeros((1, 8))
    Y_pred_sit = np.zeros((1, ))
    sit_pred_indices = []

    unique, counts = np.unique(predictions, return_counts=True)
    print("prediction class count:", np.column_stack((unique, counts)))
    print("predictions[0]:", predictions[0])

    for i in range(len(predictions)):
        # if predictions[i] == 1 or predictions[i] == 2:
        if predictions[i] == 1:
            if np.all(X_pred_sit == X_pred_sit[0]):
                X_pred_sit[0:, ] = X_test[i]
                Y_pred_sit[0] = predictions[i]
                sit_pred_indices.append(i)
            else:
                X_pred_sit = np.append(X_pred_sit, X_test[i], axis=0)
                Y_pred_sit = np.append(Y_pred_sit, predictions[i], axis=0)
                sit_pred_indices.append(i)

    # train model on sitting data samples
    Y_sit_encoded = to_categorical(Y_sit)
    X_sit_train, X_sit_test, Y_sit_train, Y_sit_test = train_test_split(X_sit, Y_sit_encoded, test_size=0.2)
    X_sit_train, X_sit_val, Y_sit_train, Y_sit_val = train_test_split(X_sit_train, Y_sit_train, test_size=0.2)

    print("X_pred:", np.all(X_pred_sit == 0))

    if not np.all(X_pred_sit == 0):
        print("og prediction:", predictions[sit_pred_indices[0]])

        sit_model = Sequential()
        sit_model.add(Dense(25, input_shape=(8,), activation='gelu'))
        sit_model.add(Dropout(0.8))
        sit_model.add(Dense(10, activation='tanh'))
        sit_model.add(Dropout(0.8))
        sit_model.add(Dense(3, activation='softmax'))
        sit_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        sit_model.summary()

        sit_model.fit(X_sit_train, Y_sit_train, validation_data=(X_sit_val, Y_sit_val), epochs=100, batch_size=20000)
        y_prediction_sit = sit_model.predict(X_pred_sit)
        sit_predictions = np.argmax(y_prediction_sit, axis=1)

        # replace previous prediction with new sit prediction after training sit data
        for i in range(len(Y_pred_sit)):
            index = sit_pred_indices[i]
            predictions[index] = sit_predictions[i]

        print("new prediction:", predictions[sit_predictions[0]])

        unique, counts = np.unique(predictions, return_counts=True)
        print("new prediction class count:", np.column_stack((unique, counts)))

    precision = precision_score(Y_test_one_label, predictions, average=evaluation_metric, zero_division=1)
    recall = recall_score(Y_test_one_label, predictions, average=evaluation_metric, zero_division=1)
    f1 = f1_score(Y_test_one_label, predictions, average=evaluation_metric, zero_division=1)

    print("precision:", precision, "recall:", recall, "f1:", f1)


def apply_smote(x, y):
    x_smote, y_smote = smote.fit_resample(x, y)

    return x_smote, y_smote


def over_sample_minority_classes(x, y):
    classes = np.bincount(y)
    minority_class = np.argmin(classes)
    second_minority_class = np.where(classes == np.partition(classes.flatten(), 1)[1])[0][0]
    majority_class = np.argmax(classes)
    amount_to_add_minority = classes[majority_class] - classes[minority_class]
    amount_to_add_second_minority = classes[majority_class] - classes[second_minority_class]
    y = y.reshape((y.shape[0], 1))
    x = np.append(x, y, axis=1)

    index_all_minority_label = np.array(np.where(x[:, 8] == minority_class)).flatten()
    index_all_second_minority_label = np.array(np.where(x[:, 8] == second_minority_class)).flatten()

    random_data_indices_minority = np.random.choice(index_all_minority_label, size=amount_to_add_minority)
    random_data_indices_second_minority = np.random.choice(
        index_all_second_minority_label, size=amount_to_add_second_minority)

    for i in random_data_indices_minority:
        row = np.array(x[i])
        row = row.reshape((1, row.shape[0]))
        x = np.append(x, row, axis=0)

    for i in random_data_indices_second_minority:
        row = np.array(x[i])
        row = row.reshape((1, row.shape[0]))
        x = np.append(x, row, axis=0)

    x_out = np.array(x[:, 0:8])
    y_out = np.array(x[:, 8]).astype(int)

    return x_out, y_out


def under_sample_majority_class(x, y):
    classes = np.bincount(y)
    majority_class = np.argmax(classes)
    second_majority_class = np.where(classes == np.partition(classes.flatten(), -2)[-2])[0][0]
    y = y.reshape((y.shape[0], 1))
    x = np.append(x, y, axis=1)

    index_all_majority_label = np.array(np.where(x[:, 8] == majority_class))
    index_all_second_majority_label = np.array(np.where(x[:, 8] == second_majority_class))
    index_all_majority_label = index_all_majority_label.flatten()
    index_all_second_majority_label = index_all_second_majority_label.flatten()

    amount_to_remove = len(index_all_majority_label) - len(index_all_second_majority_label)

    random_data_indices = np.random.choice(index_all_majority_label, size=amount_to_remove, replace=False)
    random_data_indices = np.sort(random_data_indices)
    x_new = np.delete(x, random_data_indices, axis=0)

    x_out = np.array(x_new[:, 0:8])
    y_out = np.array(x_new[:, 8])
    y_out = y_out.astype(int)

    return x_out, y_out


if __name__ == "__main__":
    # classify_by_gender()
    # classify_by_room()
    classify_as_one()
    # classify_by_activity()
