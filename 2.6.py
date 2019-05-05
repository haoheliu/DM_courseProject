import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from numpy.random import randint

random_seed = 5
np.random.seed(random_seed)

class PageRank:
    def __init__(self,max_node = 8,edge_num = 20,alpha = 0.95,stop_margin = 0.00001):
        self.max_node = max_node
        self.edge_num = edge_num
        self.alpha = alpha
        self.stop_margin = stop_margin
        self.e = 100000
        self.itr_times = 0  #iteration times recorded

        self.edges = np.zeros((edge_num,2))
        self.G = nx.Graph()
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
        N = len(self.nodes)
        # 生成初步的S矩阵
        S = np.zeros([N, N])

        for edge in self.edges:
            S[int(edge[1]), int(edge[0])] = 1

        for j in range(N):
            sumofcol = sum(S[:, j])
            for i in range(N):
                S[i, j] /= sumofcol

        A = self.alpha * S + (1 - self.alpha) / N * np.ones([N, N])

        # initialize page rank value
        P_n = np.ones(N) / N
        P_n1 = np.zeros(N)

        while self.e > self.stop_margin:
            P_n1 = np.dot(A, P_n)
            self.e = P_n1 - P_n
            self.e = max(map(abs, self.e))  # 计算误差
            P_n = P_n1
            self.itr_times += 1

        print("After" + str(self.itr_times) + "iterations:\n")

        print('Result:')
        for i,each in enumerate(P_n):
            print("node "+str(i)+": "+str(each))

    def constructGraph(self):
        for i in range(0,self.edges.shape[0]):
            self.edges[i][0] = randint(0,self.max_node)
            self.edges[i][1] = randint(0,self.max_node)

        for item in self.edges:
            self.G.add_edge(item[0],item[1])

        nx.draw(self.G,with_labels=True,node_size=700,node_color = 'y')
        plt.show()

test = PageRank()