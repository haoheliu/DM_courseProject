import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from numpy.random import randint
max_node = 20
edge_num = 1000
alpha = 0.95
random_seed = 5
stop_margin = 0.00001
e = 100000
itr_times = 0  #iteration times recorded

np.random.seed(random_seed)

edges = np.zeros((edge_num,2))
for i in range(0,edges.shape[0]):
    edges[i][0] = randint(0,max_node)
    edges[i][1] = randint(0,max_node)

G = nx.Graph()

for item in edges:
    G.add_edge(item[0],item[1])

nx.draw(G)
plt.show()

nodes = []
for item in edges:
    if(item[0] not in nodes):
        nodes.append(item[0])
    if(item[1] not in nodes):
        nodes.append(item[1])

print(len(nodes))

N = len(nodes)

# 生成初步的S矩阵
S = np.zeros([N, N])

for edge in edges:
    S[int(edge[1]), int(edge[0])] = 1

for j in range(N):
    sumofcol = sum(S[:, j])
    for i in range(N):
        S[i, j] /= sumofcol

# 计算矩阵A

A = alpha * S + (1 - alpha) / N * np.ones([N, N])

#initialize page rank value
P_n = np.ones(N) / N
P_n1 = np.zeros(N)


print('loop...')

while e > stop_margin:
    P_n1 = np.dot(A, P_n)
    e = P_n1 - P_n
    e = max(map(abs, e))  # 计算误差
    P_n = P_n1
    itr_times += 1

print("After"+str(itr_times)+"iterations:\n")

print('final result:', P_n)