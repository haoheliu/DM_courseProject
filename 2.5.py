import pandas as pd
from collections import defaultdict

class Bayes():
    def __init__(self,train_fname = "train.csv",test_fname = "test.csv",targets = ['none','soft','hard'],calibrate = False):
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
            for i,col in enumerate(self.columns):
                if(i == len(self.columns)-1): #The last column is only for prediction
                    break
                self.train(tar,col)


        for i in range(0,len(self.test_data)):
            temp = self.predict(self.test_data[i:i+1])
            self.res.append(temp)

        self.showTrainResult()

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




test = Bayes()