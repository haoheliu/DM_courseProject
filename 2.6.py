import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from numpy.random import randint

random_seed = 5
np.random.seed(random_seed)

class PageRank:
    def __init__(self,max_node = 7,edge_num = 30,alpha = 0.95,stop_margin = 0.00001):
        self.max_node = max_node
        self.edge_num = edge_num
        self.alpha = alpha
        self.stop_margin = stop_margin
        self.e = 100000
        self.itr_times = 0  #iteration times recorded

        self.edges = np.zeros((edge_num,2))
        self.G = nx.MultiDiGraph()
        self.constructGraph()

        self.nodes = []
        for item in self.edges:
            if (item[0] not in self.nodes):
                self.nodes.append(item[0])
            if (item[1] not in self.nodes):
                self.nodes.append(item[1])
        print(len(self.nodes))
        self.pageRank()

    def pageRank(self):
        print("===========Algorithm Started===========")
        node_num = len(self.nodes) #Number of nodes
        print("Number of nodes: "+str(node_num))

        P = np.zeros([node_num, node_num])

        for edge in self.edges:
            P[int(edge[1]), int(edge[0])] = 1 #Add edges

        for j in range(node_num): #Normalize
            sumofcol = sum(P[:,j])
            for i in range(node_num):
                P[i, j] /= sumofcol #For each column normalize with sum 1

        A = self.alpha * P + \
            (1 - self.alpha) / node_num \
            * np.ones([node_num, node_num])

        # initialize page rank value
        rank = np.ones(node_num) / node_num
        previous = np.zeros(node_num)

        while self.e > self.stop_margin:
            previous = np.dot(A, rank)
            self.e = previous - rank
            self.e = max(map(abs, self.e))
            rank = previous
            self.itr_times += 1

        print("After" + str(self.itr_times) + "iterations:\n")

        print('Result:')
        for i,each in enumerate(rank):
            print("node "+str(i)+"'s value: "+str(each))

    def constructGraph(self):
        for i in range(0,self.edges.shape[0]):
            self.edges[i][0] = randint(0,self.max_node+1)
            temp = self.max_node+1;
            while(temp > self.max_node  or temp < 0):
                temp = int(np.random.randn(1) * self.max_node / 2)
            self.edges[i][1] = temp
            # self.edges[i][1] = randint(0, self.max_node + 1)
        print(self.edges)
        for item in self.edges:
            self.G.add_edge(item[0],item[1])

        nx.draw(self.G,with_labels=True,node_size=700,node_color = 'y')
        plt.show()

test = PageRank(max_node = 5,edge_num = 10)