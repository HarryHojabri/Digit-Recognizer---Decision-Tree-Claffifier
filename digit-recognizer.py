import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import csv

# The train data: Note that in the file train.csv, every row corresponds to 1 label and 784 digits representing the intensity of 28*28 pixels in an image wjich ranges from 0 to 255.
# Also in the file test.cv, there are no labels and each line has 784 elements.
trainData = pd.read_csv("train.csv").as_matrix()
testData = pd.read_csv("test.csv").as_matrix()

# This section trains the model over the TrainSet (train.csv); The dataset of 42000 pictures in the file train.csv are equally divided into two sets:
# 1) trainData to train the model including the first 21000 labeled pictures; 2) testData to test the precision of the trained model over the last 21000 lebeled pictures
xtrain = trainData[0:21000, 1:]
ytrain = trainData[0:21000, 0]

# The Decision Tree Classifier is used to solve the problem and the the algorithm is trained on trainData
dtc = DecisionTreeClassifier()
dtc.fit(xtrain, ytrain)

# This section tests the model over the TestSet (testData)
xtest = trainData[21000: , 1:]
y_actual = trainData[21000: ,0]
p = dtc.predict(xtest)


# The following lines calculate the accuracy of the model comparing the model's prediction against the actual labels:
count = 0
for i in range(0,21000):
    count+=1 if p[i]==y_actual[i] else 0

print("precision = ", 100*(count/21000))

# Printing the output of the algorithm on the testData in a .csv file named TrainResult.csv:
with open('TrainResult.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([ "ImageId", "Label (Y)"])
    for i in range(0,21000):
        writer.writerow([i+1, p[i]])
file.close()


# This section predicts the labels for the images stored in the file test.csv
xpred = testData[0:, 0:]
pred = dtc.predict(xpred)
# Printing the output (labels) by the algorithm on the test.csv 28000-row input file in a .csv file named sample_submission.csv
with open('sample_submission.csv', 'w', newline ='') as file_pred:
    writer_pred = csv.writer(file_pred)
    writer_pred.writerow(["ImageId", "Label (Y)"])
    for i in range(0,28000):
        writer_pred.writerow([i+1, pred[i]])
file_pred.close()
