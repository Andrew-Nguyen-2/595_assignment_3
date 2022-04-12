# 595_assignment_3 - Deep Learning

Part One:
  Using the Iris dataset, I created a 3-layer densely connected neural network using Keras.
  Dataset is split into training data and testing data with training data being 80% of total   dataset and testing data being 20% of total dataset.  Validation data is used during      training and is 20% of training data.  The evaluation metrics used are precision score, recall score, and f1 score.
  
Part Two:
  Using the Activity Recognition with Healthy Older People dataset, I created three classifiers: a majority class classifier, an average class classifier, and a 3-layer densely connected neural network using Keras.  The dataset was recorded in two different rooms with two different antennae set ups and each file is of one subject with their gender identified at the end of the file indicated with an 'M' and an 'F'.
  
  There are four different dataset types for each classifier: male vs. female, room 1 vs. room 2, walking vs. sitting vs. laying, and the combined dataset.  Each one separates the data (or not) by the person the data is collected from, the room the data is collected from, by activity, and the whole dataset.  The walking vs. sitting vs. laying classifier, classifies the test data into one of the three activities then determines what they are sitting on if it is classified as sitting.