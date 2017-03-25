#K-bayesian-classifier


""""
This little script is broken down into the following steps:

Handle Data: Load the data from CSV file and split it into training and test datasets.
Summarize Data: summarize the properties in the training dataset so that we can calculate probabilities and make predictions.
Make a Prediction: Use the summaries of the dataset to generate a single prediction.
Make Predictions: Generate predictions given a test dataset and a summarized training dataset.
Evaluate Accuracy: Evaluate the accuracy of predictions made for a test dataset as the percentage correct out of all predictions made.
Tie it Together: Use all of the code elements to present a complete and standalone implementation of the Naive Bayes algorithm.
""""

import random
import csv
import math

def loadCsv(filename):
    lines = csv.reader(open(filename, "rb"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

    filename = 'X_train.csv'
    dataset = loadCsv(filename)
    print('Loaded data file{0} with {1} rows').format(filename, len(dataset))



    def splitDataset(dataset, splitRatio):
            """"
            We need to split the data into a training dataset that Naive Bayes can use to make predictions
            and a test dataset that we can use to evaluate the accuracy of the model.
            We need to split the data set randomly into train and datasets with a ratio of 67% train and 33% test (this is a common ratio for testing an algorithm on a dataset).

            Below is the splitDataset() function that will split a given dataset into a given split ratio.
            """"
        trainSize = int(len(dataset) * splitRatio)
        trainSet = []
        copy = list(dataset)
        while len(trainSet) < trainSize:
            index = random.randrange(len(copy))
            trainSet.append(copy.pop(index))

            return [trainSet, copy]


  def separateByClass(dataset):
      """"
      The first task is to separate the training dataset instances by class value
      so that we can calculate statistics for each class.
      We can do that by creating a map of each class value to
       a list of instances
      that belong to that class and sort
       the entire dataset of instances
       into the appropriate lists.
      """"
      separated = {}
      for i in range(len(dataset)):
          vector = dataset[i]
          if(vector[-1] not in separated):
              separated[vector[-1]] = []
             separated[vector[-1]].append(vector)
             return separated

##calculate the mean
    def mean(numbers):
        return sum(numbers)/float(len(numbers))

    def stdev(numbers):
        avg = mean(numbers)
        variance = sum([pow(x-avg, 2) for x in numbers])/float(len(numbers)-1)
        return math.sqrt(variance)

#summarize

def summarize(dataset):
    """"
    Now we have the tools to summarize a dataset. For a given list of instances (for a class value) we can calculate the mean and the standard deviation
     for each attribute.
    The zip function groups the values for each attribute across our data instances
    into their own lists so that we can compute
    the mean and standard deviation values for the attribute.
    """"
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

def summarizeByClass(dataset)
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.iteritems():
