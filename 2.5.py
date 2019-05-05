import pandas as pd
import numpy as np
import os
import pickle

from collections import defaultdict
from heapq import nsmallest

class Bayeslearner:
    def __init__(self,train_fname = "train.csv",
                 test_fname = "test.csv",
                 targets = ['none','soft','hard'],
                 calibrate = True):
        self.train_data = pd.read_csv(train_fname)
        self.test_data = pd.read_csv(test_fname)
        self.target = targets
        self.columns = list(self.train_data.columns)
        self.P_x = {}
        self.P_y = defaultdict(int)
        self.calibrate = calibrate
        self.res = []

        temp = self.train_data[["contact-lenses"]].values.tolist()

        for each in temp:
            self.P_y[each[0]]+=1
        for each in self.P_y.keys():
            if(self.calibrate == False):
                self.P_y[each] /= len(temp)
            else:
                self.P_y[each] += 1
                self.P_y[each] /= (len(temp)+len(targets))

        for tar in targets:
            self.P_x[tar] = self.construct()
            print(self.P_x[tar])
            for i,col in enumerate(self.columns):
                if(i == len(self.columns)-1): #The last column is only for prediction
                    break
                self.train(tar,col)

        for i in range(0,len(self.test_data)):
            temp = self.predict(self.test_data[i:i+1])
            self.res.append(temp)

        pd.DataFrame({"result":self.res}).to_csv("Bayes_result.csv")

        #self.showTrainResult()

    def construct(self)->dict:
        dic = {}
        for i, item in enumerate(self.columns):
            if(i == len(self.columns)-1):
                return dic #last column excluded
            keys = set(self.train_data.iloc[:,i].values)
            dic[item] = {}
            for each in keys:
                dic[item][each] = 0
        return dic

    def train(self,target,column):
        #Select data with a specific target
        data = self.train_data[self.train_data["contact-lenses"]==target]
        #the column data for this target
        col = data[[column]].values.tolist()
        length = len(col)
        for each in col:
            self.P_x[target][column][each[0]] += 1
        for each in self.P_x[target][column].keys():
            #calculate the probability
            if(self.calibrate == True):
                self.P_x[target][column][each]+=1
                self.P_x[target][column][each] /= (length+len(self.P_x[target][column].keys()))
            else:
                self.P_x[target][column][each] /= length

    def predict(self,item) -> str:
        probability = dict().fromkeys(self.P_y.keys())
        col_n = self.columns[:-1]
        item = pd.DataFrame(item,columns=col_n)
        item = item.values.tolist()
        #For the three targets
        for key in probability.keys():
            probability[key] = 1
            #for each item in this transaction
            #Here we use the column name indirectly
            for i,each in enumerate(item[0]):
                probability[key] *= self.P_x[key][self.columns[i]][each]
            probability[key] *= self.P_y[key]
        print(probability)
        res = max(probability,key=probability.get)
        print(res)
        return res

    def showTrainResult(self):
        print("===============RESULT===============")
        for each in self.P_x.keys():
            print("For target: "+each)
            for item in self.P_x[each].keys():
                print("\tFor attribute: "+item)
                for child in self.P_x[each][item].keys():
                    print("\t\t"+child+": "+str(self.P_x[each][item][child]))
        print("Class probability: \n"+str(self.P_y))
        print("Prediction result: ")
        print(self.res)

class knnclassifier :
    def __init__(self,train_fname = "train.csv",
                 test_fname = "test.csv",
                 code_fname = "code.pkl",
                 targets = ['none','soft','hard'],
                 k = 2):
        self.train_data_raw = pd.read_csv(train_fname)
        self.test_data_raw = pd.read_csv(test_fname)
        self.target = targets
        self.columns = list(self.train_data_raw.columns)
        self.train_data = []
        self.test_data = []
        self.codes = {}
        self.k = k
        self.load = False
        self.pred = []

        if(os.path.exists(code_fname)):
            f = open(code_fname,"rb")
            self.codes = pickle.load(f)
            f.close()
            self.load = True

        self.encode()
        if(self.load == False):
            f = open(code_fname,"wb")
            pickle.dump(self.codes,f)
            f.close()

        for each in self.test_data:
            self.pred.append(self.findBest(self.predict(each[:-1])))

        pd.DataFrame({"result": self.pred}).to_csv("Knn_result.csv")
        self.show_encode()
        print(self.pred)

    def encode(self):
        for i,each in enumerate(self.columns):
            if(self.load == False):
                #Find all value in this column
                codes = list(set(self.train_data_raw.iloc[:, i].values))
                #Setup a dictionary for encoding
                self.codes[each] = dict(zip(codes,range(0,len(codes))))
                self.codes[each]['?'] = -1 #Add this key manully
            #fatch the raw data of this column
            temp = self.train_data_raw[[each]].values.tolist()
            temp_test = self.test_data_raw[[each]].values.tolist()
            for i in range(0,len(temp)):
                temp[i] = self.codes[each][temp[i][0]]
            for i in range(0,len(temp_test)):
                temp_test[i] = self.codes[each][temp_test[i][0]] #The trancaction for test

            self.train_data.append(temp)
            self.test_data.append(temp_test)
        #Transpose, modify the data format
        self.train_data = np.transpose(self.train_data).tolist()
        self.test_data = np.transpose(self.test_data).tolist()

    def distance(self,vec1,vec2)->int:
        res = 0
        if(not len(vec1)==len(vec2)):
            raise ValueError("The two vectors do not have equal length")
        for i in range(0,len(vec1)):
            res += (vec1[i]-vec2[i])**2
        return res

    def findKey(self,dic,value):
        for each in dic.items():
            if(each[1] == value):
                return each[0]
        return None

    def predict(self,code):
        dist = []
        k_largest = []
        for each in self.train_data:
            dist.append(self.distance(code,each[:-1]))
        k_largest = list(map(dist.index,nsmallest(self.k,dist)))
        #print(dist)
        #print(k_largest)
        res = []
        for each in k_largest:
            res.append(self.train_data[each][-1])
        for i in range(0,len(res)):
            res[i] = self.findKey(self.codes["contact-lenses"],res[i])
        return res

    def findBest(self,vec):
        keys = list(set(vec))
        m = dict().fromkeys(keys)
        for each in m.keys():
            m[each] = 0
        for each in vec:
            m[each] += 1
        return max(m,key=m.get)

    def show_encode(self):
        for each in self.train_data:
            print(each)
        for each in self.test_data:
            print(each)
        print("The code dictionary:\n")
        print(self.codes)

if __name__ == "__main__":
    Bayeslearner()
    knnclassifier()





