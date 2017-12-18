import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import os
import time
from scipy import misc
from feature import NPDFeature
import pickle
from ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report 

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

if __name__ == "__main__":
    #load face data
    datafile = 'data'
    #data already preprocessed
    if os.path.exists(datafile):
        input = open(datafile, 'rb')
        X_train = pickle.load(input)
        X_vali = pickle.load(input)
        y_train = pickle.load(input)
        y_vali = pickle.load(input)
        input.close()
    #preprocess data
    else:
        facepath = 'datasets/original/face'
        nonfacepath = 'datasets/original/nonface'

        face = []
        nonface = []

        #for each image, convert it into grayscale presentation
        #scale to 24x24
        #and extract its NPD feature
        facedir = os.listdir(facepath)
        for i in range(0, len(facedir)):
            if facedir[i].endswith('jpg'):
                path = os.path.join(facepath, facedir[i])
                img = mpimg.imread(path)
                img = rgb2gray(img)
                img = misc.imresize(img, [24, 24])
                face.append(NPDFeature(img).extract())

        nonfacedir = os.listdir(nonfacepath)
        for i in range(0, len(nonfacedir)):
            if nonfacedir[i].endswith('jpg'):
                path = os.path.join(nonfacepath, nonfacedir[i])
                img = mpimg.imread(path)
                img = rgb2gray(img)
                img = misc.imresize(img, [24, 24])
                nonface.append(NPDFeature(img).extract())

        X = np.array(face + nonface)
        y = np.ones([1000])
        y[500:999] = -1

        X_train, X_vali, y_train, y_vali = train_test_split(X, y, test_size=0.2, random_state=24)
        output = open(datafile, 'wb')
        pickle.dump(X_train, output)
        pickle.dump(X_vali, output)
        pickle.dump(y_train, output)
        pickle.dump(y_vali, output)
        output.close()

    #create adaboost/weak classifier
    dtc = DecisionTreeClassifier(random_state=0, max_depth=3, max_features="sqrt")
    classifier = AdaBoostClassifier(dtc, 15)
    #train classifiers
    classifier.fit(X_train, y_train)
    dtc.fit(X_train, y_train)
    #do prediction
    result = classifier.predict(X_vali)
    weakresult = dtc.predict(X_vali)

    #calculate predicting accuracy for both
    adacount = 0
    weakcount = 0
    for i in range(0, result.shape[0]):
        if (np.abs(result[i]-1) < np.abs(result[i] + 1)):
            result[i] = 1
        else:
            result[i] = -1
        if result[i] == y_vali[i]:
            adacount = adacount + 1
        if weakresult[i] == y_vali[i]:
            weakcount = weakcount + 1
    print ("adaboost accuracy: " + str(adacount / result.shape[0]))
    print ("weak accuracy: " + str(weakcount / result.shape[0]))

    print(classification_report(y_vali, result))