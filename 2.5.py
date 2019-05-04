from sklearn import datasets
import pandas as pd
import pickle

class bayes():
    def __init__(self,train_fname = "train.csv",test_fname = "test.csv",targets = ['none','soft','hard']):
        self.train_data = pd.read_csv(train_fname)
        self.test_data = pd.read_csv(test_fname)
        self.target = targets
        self.columns = list(self.train_data.columns)
        self.params = {}
        for each in targets:
            self.params[each] = self.construct()
        self.train("none","age")
        print(self.params['none'])

    def construct(self)->dict:
        dic = {}
        for i, item in enumerate(self.columns):
            if(i == len(self.columns)-1):
                return dic #last column excluded
            keys = set(self.train_data.iloc[:,i].values)
            dic[item] = {}
            for each in keys:
                dic[item][each] = 0

    def train(self,target,column):
        #Select data with a specific target
        data = self.train_data[self.train_data["contact-lenses"]==target]
        col = data[[column]].values.tolist()
        length = len(col)
        for each in col:
            self.params[target][column][each[0]] += 1
        for each in self.params[target][column].keys():
            self.params[target][column][each] /= length




test = bayes()