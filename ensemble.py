import pickle
import math
import numpy as np
import copy

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''
    __base_classifier__ = None
    __classifiers__ = None
    __max_base__ = 0
    __n_base__ = 0
    __alpha__ = None

    def __init__(self, weak_classifier, n_weakers_limit):
        self.__base_classifier__ = weak_classifier
        self.__max_base__ = n_weakers_limit

    def fit(self,X,y):
        self.__alpha__ = []
        self.__classifiers__ = []
        W = np.zeros([self.__max_base__, X.shape[0]])
        W[0, :] = 1 / X.shape[0]
        for m in range(0, self.__max_base__):
            #train m-th classifier
            print ("train the " + str(m + 1) + " base classifier")
            base = copy.deepcopy(self.__base_classifier__)
            base = base.fit(X, y, W[m])
            self.__classifiers__.append(base)

            #predict through the m-th classifier
            y_predict = base.predict(X)
            #calculate error
            h = np.zeros(y_predict.shape)
            for i in range(0, y.shape[0]):
                if y_predict[i] != y[i]:
                    h[i] = 1
                else:
                    h[i] = 0
                h[i] = h[i] * W[m, i]
            epsilon = np.sum(h)

            #calculate alpha value
            self.__alpha__.append(0.5 * math.log(1 / epsilon - 1))

            #reach max number of classifiers or good enough
            if m >= self.__max_base__ - 1 or epsilon < 0.1:
                self.__n_base__ = m + 1
                break

            #calculate weights for the next classifier
            w = np.zeros([X.shape[0]])
            for i in range(0, X.shape[0]):
                w[i] = W[m, i] * math.exp(-self.__alpha__[m] * y[i] * y_predict[i])
            z = np.sum(w)
            for i in range(0, X.shape[0]):
                W[m+1, i] = w[i] / z

    def predict(self, X, threshold=0):
        #sum prediction of all classifiers by their alpha
        alpha = np.array(self.__alpha__)
        h = np.zeros([self.__n_base__, X.shape[0]])
        for m in range(0, self.__n_base__):
            h[m] = alpha[m] * self.__classifiers__[m].predict(X)
        return np.sum(h, axis=0)
