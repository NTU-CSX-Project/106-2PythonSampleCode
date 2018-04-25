#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

def get_testing_labels(training_samples, training_labels, testing_samples):
    normalizer       = Normalizer().fit(training_samples)
    training_samples = normalizer.transform(training_samples)
    testing_samples  = normalizer.transform(testing_samples)

    pca              = PCA(n_components=0.8, whiten=True).fit(training_samples)
    training_samples = pca.transform(training_samples)
    testing_samples  = pca.transform(testing_samples)

    svm_classifier   = SVC(C=2.8).fit(training_samples, training_labels)
    return svm_classifier.predict(testing_samples)

def main():
    # Read training and testing data
    training_df  = pd.read_csv("../input/train.csv")
    testing_df   = pd.read_csv("../input/test.csv")
    print("Training set has {0[0]} rows and {0[1]} columns".format(training_df.shape))
    print("Test set has {0[0]} rows and {0[1]} columns".format(testing_df.shape))

    # Get training samples and labels from training data
    training_labels     = training_df['label'].values
    training_samples    = training_df.drop(['label'], axis = 1).values
    # Get testing samples from testing data
    testing_samples     = testing_df.values
    testing_labels      = get_testing_labels(training_samples, training_labels, testing_samples)

    # Writing testing labels to CSV
    testing_labels_df   = pd.DataFrame(testing_labels, columns=['Label'])
    testing_labels_df.index     += 1
    testing_labels_df.index.name = 'ImageId'
    testing_labels_df.to_csv('sample_submission.csv', sep=',')

if __name__ == "__main__":
    main()
