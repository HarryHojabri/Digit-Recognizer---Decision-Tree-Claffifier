This project, taken from Kaggle, is about recognizing the digits for images in an unlabeled dataset consisting 28000 images in the field of computer vision:

https://www.kaggle.com/c/digit-recognizer

There are 2 datasets in the project:

"train.csv" which is the labeled training set including linear representations of 42000 images and their labels. Having images of 2828 pixels in each image, a linear representation, including 784 numbers showing the density of the pixel in [0, 255], i+28j stands for the position of a pixel within the linear representation where i and j are the indices of the pixel in the image. Note that each row also contains a label (first one) for each image making a total of 785 elements in each row.
In this project, we use the first 21000 images to train the model, and then use the second 21000 ones to calculate the accuracy of the model which is over 83%.

"test.csv" including 28000 unlabeled images: After training and testing the model on train.csv, the labels are calculated for the this dataset and stored in the file "sample_submission.csv".
The library scikit-learn is imported and Decision Tree Classifier is applied to solve the problem.
