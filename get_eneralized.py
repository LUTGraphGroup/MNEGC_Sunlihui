import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import pandas as pd
from collections import OrderedDict
import numpy as np
import networkx as nx
import math
from scipy.stats import kendalltau


# 读取 Excel 文件
df = pd.read_excel('嵌入向量路径', header=None)

# 创建一个空字典用于存储节点嵌入向量
node_embeddings = OrderedDict()

# 遍历 DataFrame 中的每一行，构建节点嵌入向量字典
for index, row in df.iterrows():
    node_id = row[0]  # 节点标识符
    embedding = list(row[1:])  # 节点的嵌入向量
    node_embeddings[node_id] = embedding

# print(node_embeddings)




def euclidean_distance(embedding1, embedding2):
    return euclidean(embedding1, embedding2)
#
def distance(node_embeddings):
    num_nodes = len(node_embeddings)
    distance_matrix = np.zeros([num_nodes, num_nodes])
    for idx1, (node1, embedding1) in enumerate(node_embeddings.items()):
        for idx2, (node2, embedding2) in enumerate(node_embeddings.items()):
            if node1 != node2:
                distance_matrix[idx1][idx2] = euclidean_distance(embedding1, embedding2)
    #print(distance_matrix)
    return distance_matrix


def Cluster(G):
    C=nx.clustering(G)
    Cl=[]
    for i in C.values():
        Cl.append(i)
    return (Cl)

def k_shell(G):
    """计算每个节点的KS值"""
    G1 = G.copy()  # 目的是对G1进行删点和边的操作，对G没有影响

    def k_shell_1(G1):
        importance_dict = {}
        level = 1
        while len(G1.degree):
            importance_dict[level] = []
            while True:
                level_node_list = []
                for item in G1.degree:
                    if item[1] <= level:
                        level_node_list.append(item[0])
                G1.remove_nodes_from(level_node_list)  # 从G中移除节点，移除完后为空，导致后续函数调用G报列表索引越界，k_sheel(G)放到最后
                importance_dict[level].extend(level_node_list)
                if not len(G1.degree):
                    return importance_dict
                if min(G1.degree, key=lambda x: x[1])[1] > level:
                    break
            level = min(G1.degree, key=lambda x: x[1])[1]
        # print('importance_dict',importance_dict)
        return importance_dict

    a = k_shell_1(G1)
    # print('a',a)
    H = {}
    for x, y in a.items():
        for z in y:
            H[z] = x
    # print('H',H)
    H_reverse = sorted(H.items(), key=lambda x: x[0])
    # print(dict(H_reverse))
    KS1 = list(dict(H_reverse).values())
    # print('KS1',KS1)
    return KS1



def calculate_Ei_values(G, A):

    Ei_values = []  # 用于存储计算的Ei值
    num_nodes = A.shape[0]

    for a in np.arange(0.1, 1.1, 0.1):
        LC_values = []  # 存储每个节点的 LC 值

        # 计算每个节点的 LC 值
        for i in G.nodes():
            di = G.degree(i)
            neighbor_nodes = list(G.neighbors(i))
            neighbor_degrees_sum = sum(G.degree(j) for j in neighbor_nodes)
            neighbor_degree_sum_square = sum([G.degree(j) ** 2 for j in neighbor_nodes])
            T = nx.triangles(G, i)

            LC = (a ** 3) * (di ** 3) + 3 * a * ((1 - a) ** 2) * (di ** 2) + (
                    6 * (a ** 2) - 3 * a - 2 * (a ** 3)) * di + 3 * (a ** 3) * (
                     neighbor_degree_sum_square) + 3 * a * (a ** 2 - 4 * a + 2) * neighbor_degrees_sum + 6 * (
                         (1 - a) ** 3) * T

            LC_values.append(LC)
        #print(LC_values)


        NSC = []
        ks=k_shell(G)
        c=Cluster(G)

        for i in range(len(G)):
            sum1=(LC_values[i]*sum(ks[j] for j in list(G.neighbors(i))))/(math.exp(c[i]))
            NSC.append(sum1)
        #print(NSC)

        # H = np.zeros((num_nodes, num_nodes))
        dis = distance(node_embeddings)

        MNEGC_values = []

        for i in range(num_nodes):
            F1 = 0
            # 使用 ego_graph 函数来获取节点的三阶邻域
            neighborhood = nx.ego_graph(G, i, radius=5)
            # 从邻域子图中删除中心节点
            neighborhood.remove_node(i)
            for j in range(num_nodes):
                if j in neighborhood.nodes():
                    F1 += (NSC[i] * NSC[j] / dis[i][j] ** 2)
            MNEGC_values.append(F1)


        # 计算 KSGC
        Ei_values.append(MNEGC_values)
        #print(Ei_values)

    return Ei_values
#
def calculate_Ei_values_Crime(G, A):
    num_nodes = A.shape[0]
    a=0.2
    LC_values = []  # 存储每个节点的 LC 值
    for i in G.nodes():
        di = G.degree(i)
        neighbor_nodes = list(G.neighbors(i))
        neighbor_degrees_sum = sum(G.degree(j) for j in neighbor_nodes)
        neighbor_degree_sum_square = sum([G.degree(j) ** 2 for j in neighbor_nodes])
        T = nx.triangles(G, i)

        LC = (a ** 3) * (di ** 3) + 3 * a * ((1 - a) ** 2) * (di ** 2) + (
                    6 * (a ** 2) - 3 * a - 2 * (a ** 3)) * di + 3 * (a ** 3) * (
                     neighbor_degree_sum_square) + 3 * a * (a ** 2 - 4 * a + 2) * neighbor_degrees_sum + 6 * (
                         (1 - a) ** 3) * T

        LC_values.append(LC)
        #print(LC_values)
    NSC = []
    ks=k_shell(G)
    c=Cluster(G)

    for i in range(len(G)):
        sum1=(LC_values[i]*sum(ks[j] for j in list(G.neighbors(i))))/(math.exp(c[i]))
        NSC.append(sum1)
        #print(NSC)


    dis = distance(node_embeddings)
    MNEGC_values = []
    for i in range(num_nodes):
        F1 = 0
            # 使用 ego_graph 函数来获取节点的三阶邻域
        neighborhood = nx.ego_graph(G, i, radius=3)
            # 从邻域子图中删除中心节点
        neighborhood.remove_node(i)
        for j in range(num_nodes):
            if j in neighborhood.nodes():
                F1 += (NSC[i] * NSC[j] / dis[i][j] ** 2)
        MNEGC_values.append(F1)
    return MNEGC_values
#
def calculate_Ei_values_EEC(G, A):
    num_nodes = A.shape[0]
    a=0.1
    LC_values = []  # 存储每个节点的 LC 值
    for i in G.nodes():
        di = G.degree(i)
        neighbor_nodes = list(G.neighbors(i))
        neighbor_degrees_sum = sum(G.degree(j) for j in neighbor_nodes)
        neighbor_degree_sum_square = sum([G.degree(j) ** 2 for j in neighbor_nodes])
        T = nx.triangles(G, i)

        LC = (a ** 3) * (di ** 3) + 3 * a * ((1 - a) ** 2) * (di ** 2) + (
                    6 * (a ** 2) - 3 * a - 2 * (a ** 3)) * di + 3 * (a ** 3) * (
                     neighbor_degree_sum_square) + 3 * a * (a ** 2 - 4 * a + 2) * neighbor_degrees_sum + 6 * (
                         (1 - a) ** 3) * T

        LC_values.append(LC)
        #print(LC_values)
    NSC = []
    ks=k_shell(G)
    c=Cluster(G)

    for i in range(len(G)):
        sum1=(LC_values[i]*sum(ks[j] for j in list(G.neighbors(i))))/(math.exp(c[i]))
        NSC.append(sum1)
        #print(NSC)


    dis = distance(node_embeddings)
    MNEGC_values = []
    for i in range(num_nodes):
        F1 = 0
            # 使用 ego_graph 函数来获取节点的三阶邻域
        neighborhood = nx.ego_graph(G, i, radius=3)
            # 从邻域子图中删除中心节点
        neighborhood.remove_node(i)
        for j in range(num_nodes):
            if j in neighborhood.nodes():
                F1 += (NSC[i] * NSC[j] / dis[i][j] ** 2)
        MNEGC_values.append(F1)

    return MNEGC_values
#
def calculate_Ei_values_Email(G, A):
    num_nodes = A.shape[0]
    a=0.2
    LC_values = []  # 存储每个节点的 LC 值
    for i in G.nodes():
        di = G.degree(i)
        neighbor_nodes = list(G.neighbors(i))
        neighbor_degrees_sum = sum(G.degree(j) for j in neighbor_nodes)
        neighbor_degree_sum_square = sum([G.degree(j) ** 2 for j in neighbor_nodes])
        T = nx.triangles(G, i)

        LC = (a ** 3) * (di ** 3) + 3 * a * ((1 - a) ** 2) * (di ** 2) + (
                    6 * (a ** 2) - 3 * a - 2 * (a ** 3)) * di + 3 * (a ** 3) * (
                     neighbor_degree_sum_square) + 3 * a * (a ** 2 - 4 * a + 2) * neighbor_degrees_sum + 6 * (
                         (1 - a) ** 3) * T

        LC_values.append(LC)
        #print(LC_values)
    NSC = []
    ks=k_shell(G)
    c=Cluster(G)

    for i in range(len(G)):
        sum1=(LC_values[i]*sum(ks[j] for j in list(G.neighbors(i))))/(math.exp(c[i]))
        NSC.append(sum1)
        #print(NSC)


    dis = distance(node_embeddings)
    MNEGC_values = []
    for i in range(num_nodes):
        F1 = 0
            # 使用 ego_graph 函数来获取节点的三阶邻域
        neighborhood = nx.ego_graph(G, i, radius=3)
            # 从邻域子图中删除中心节点
        neighborhood.remove_node(i)
        for j in range(num_nodes):
            if j in neighborhood.nodes():
                F1 += (NSC[i] * NSC[j] / dis[i][j] ** 2)
        MNEGC_values.append(F1)

    return MNEGC_values

def calculate_Ei_values_Hamster(G, A):
    num_nodes = A.shape[0]
    a=0.3
    LC_values = []  # 存储每个节点的 LC 值
    for i in G.nodes():
        di = G.degree(i)
        neighbor_nodes = list(G.neighbors(i))
        neighbor_degrees_sum = sum(G.degree(j) for j in neighbor_nodes)
        neighbor_degree_sum_square = sum([G.degree(j) ** 2 for j in neighbor_nodes])
        T = nx.triangles(G, i)

        LC = (a ** 3) * (di ** 3) + 3 * a * ((1 - a) ** 2) * (di ** 2) + (
                    6 * (a ** 2) - 3 * a - 2 * (a ** 3)) * di + 3 * (a ** 3) * (
                     neighbor_degree_sum_square) + 3 * a * (a ** 2 - 4 * a + 2) * neighbor_degrees_sum + 6 * (
                         (1 - a) ** 3) * T

        LC_values.append(LC)
        #print(LC_values)
    NSC = []
    ks=k_shell(G)
    c=Cluster(G)

    for i in range(len(G)):
        sum1=(LC_values[i]*sum(ks[j] for j in list(G.neighbors(i))))/(math.exp(c[i]))
        NSC.append(sum1)
        #print(NSC)


    dis = distance(node_embeddings)
    MNEGC_values = []
    for i in range(num_nodes):
        F1 = 0
            # 使用 ego_graph 函数来获取节点的三阶邻域
        neighborhood = nx.ego_graph(G, i, radius=3)
            # 从邻域子图中删除中心节点
        neighborhood.remove_node(i)
        for j in range(num_nodes):
            if j in neighborhood.nodes():
                F1 += (NSC[i] * NSC[j] / dis[i][j] ** 2)
        MNEGC_values.append(F1)

    return MNEGC_values

def calculate_Ei_values_Yeast(G, A):
    num_nodes = A.shape[0]
    a=0.3
    LC_values = []  # 存储每个节点的 LC 值
    for i in G.nodes():
        di = G.degree(i)
        neighbor_nodes = list(G.neighbors(i))
        neighbor_degrees_sum = sum(G.degree(j) for j in neighbor_nodes)
        neighbor_degree_sum_square = sum([G.degree(j) ** 2 for j in neighbor_nodes])
        T = nx.triangles(G, i)

        LC = (a ** 3) * (di ** 3) + 3 * a * ((1 - a) ** 2) * (di ** 2) + (
                    6 * (a ** 2) - 3 * a - 2 * (a ** 3)) * di + 3 * (a ** 3) * (
                     neighbor_degree_sum_square) + 3 * a * (a ** 2 - 4 * a + 2) * neighbor_degrees_sum + 6 * (
                         (1 - a) ** 3) * T

        LC_values.append(LC)
        #print(LC_values)
    NSC = []
    ks=k_shell(G)
    c=Cluster(G)

    for i in range(len(G)):
        sum1=(LC_values[i]*sum(ks[j] for j in list(G.neighbors(i))))/(math.exp(c[i]))
        NSC.append(sum1)
        #print(NSC)


    dis = distance(node_embeddings)
    MNEGC_values = []
    for i in range(num_nodes):
        F1 = 0
            # 使用 ego_graph 函数来获取节点的三阶邻域
        neighborhood = nx.ego_graph(G, i, radius=3)
            # 从邻域子图中删除中心节点
        neighborhood.remove_node(i)
        for j in range(num_nodes):
            if j in neighborhood.nodes():
                F1 += (NSC[i] * NSC[j] / dis[i][j] ** 2)
        MNEGC_values.append(F1)



    return MNEGC_values

def calculate_Ei_values_PGP(G, A):
    num_nodes = A.shape[0]
    a=0.2
    LC_values = []  # 存储每个节点的 LC 值
    for i in G.nodes():
        di = G.degree(i)
        neighbor_nodes = list(G.neighbors(i))
        neighbor_degrees_sum = sum(G.degree(j) for j in neighbor_nodes)
        neighbor_degree_sum_square = sum([G.degree(j) ** 2 for j in neighbor_nodes])
        T = nx.triangles(G, i)

        LC = (a ** 3) * (di ** 3) + 3 * a * ((1 - a) ** 2) * (di ** 2) + (
                    6 * (a ** 2) - 3 * a - 2 * (a ** 3)) * di + 3 * (a ** 3) * (
                     neighbor_degree_sum_square) + 3 * a * (a ** 2 - 4 * a + 2) * neighbor_degrees_sum + 6 * (
                         (1 - a) ** 3) * T

        LC_values.append(LC)
        #print(LC_values)
    NSC = []
    ks=k_shell(G)
    c=Cluster(G)

    for i in range(len(G)):
        sum1=(LC_values[i]*sum(ks[j] for j in list(G.neighbors(i))))/(math.exp(c[i]))
        NSC.append(sum1)
        #print(NSC)


    dis = distance(node_embeddings)
    MNEGC_values = []
    for i in range(num_nodes):
        F1 = 0
            # 使用 ego_graph 函数来获取节点的三阶邻域
        neighborhood = nx.ego_graph(G, i, radius=3)
            # 从邻域子图中删除中心节点
        neighborhood.remove_node(i)
        for j in range(num_nodes):
            if j in neighborhood.nodes():
                F1 += (NSC[i] * NSC[j] / dis[i][j] ** 2)
        MNEGC_values.append(F1)



    return MNEGC_values

def calculate_Ei_values_Sex(G, A):
    num_nodes = A.shape[0]
    a=0.1
    LC_values = []  # 存储每个节点的 LC 值
    for i in G.nodes():
        di = G.degree(i)
        neighbor_nodes = list(G.neighbors(i))
        neighbor_degrees_sum = sum(G.degree(j) for j in neighbor_nodes)
        neighbor_degree_sum_square = sum([G.degree(j) ** 2 for j in neighbor_nodes])
        T = nx.triangles(G, i)

        LC = (a ** 3) * (di ** 3) + 3 * a * ((1 - a) ** 2) * (di ** 2) + (
                    6 * (a ** 2) - 3 * a - 2 * (a ** 3)) * di + 3 * (a ** 3) * (
                     neighbor_degree_sum_square) + 3 * a * (a ** 2 - 4 * a + 2) * neighbor_degrees_sum + 6 * (
                         (1 - a) ** 3) * T

        LC_values.append(LC)
        #print(LC_values)
    NSC = []
    ks=k_shell(G)
    c=Cluster(G)

    for i in range(len(G)):
        sum1=(LC_values[i]*sum(ks[j] for j in list(G.neighbors(i))))/(math.exp(c[i]))
        NSC.append(sum1)
        #print(NSC)


    dis = distance(node_embeddings)
    MNEGC_values = []
    for i in range(num_nodes):
        F1 = 0
            # 使用 ego_graph 函数来获取节点的三阶邻域
        neighborhood = nx.ego_graph(G, i, radius=3)
            # 从邻域子图中删除中心节点
        neighborhood.remove_node(i)
        for j in range(num_nodes):
            if j in neighborhood.nodes():
                F1 += (NSC[i] * NSC[j] / dis[i][j] ** 2)
        MNEGC_values.append(F1)



    return MNEGC_values

def calculate_Ei_values_Power(G, A):
    num_nodes = A.shape[0]
    a=0.2
    LC_values = []  # 存储每个节点的 LC 值
    for i in G.nodes():
        di = G.degree(i)
        neighbor_nodes = list(G.neighbors(i))
        neighbor_degrees_sum = sum(G.degree(j) for j in neighbor_nodes)
        neighbor_degree_sum_square = sum([G.degree(j) ** 2 for j in neighbor_nodes])
        T = nx.triangles(G, i)

        LC = (a ** 3) * (di ** 3) + 3 * a * ((1 - a) ** 2) * (di ** 2) + (
                    6 * (a ** 2) - 3 * a - 2 * (a ** 3)) * di + 3 * (a ** 3) * (
                     neighbor_degree_sum_square) + 3 * a * (a ** 2 - 4 * a + 2) * neighbor_degrees_sum + 6 * (
                         (1 - a) ** 3) * T

        LC_values.append(LC)
        #print(LC_values)
    NSC = []
    ks=k_shell(G)
    c=Cluster(G)

    for i in range(len(G)):
        sum1=(LC_values[i]*sum(ks[j] for j in list(G.neighbors(i))))/(math.exp(c[i]))
        NSC.append(sum1)
        #print(NSC)


    dis = distance(node_embeddings)
    MNEGC_values = []
    for i in range(num_nodes):
        F1 = 0
            # 使用 ego_graph 函数来获取节点的三阶邻域
        neighborhood = nx.ego_graph(G, i, radius=3)
            # 从邻域子图中删除中心节点
        neighborhood.remove_node(i)
        for j in range(num_nodes):
            if j in neighborhood.nodes():
                F1 += (NSC[i] * NSC[j] / dis[i][j] ** 2)
        MNEGC_values.append(F1)



    return MNEGC_values

def calculate_Ei_values_Facebook(G, A):
    num_nodes = A.shape[0]
    a=0.1
    LC_values = []  # 存储每个节点的 LC 值
    for i in G.nodes():
        di = G.degree(i)
        neighbor_nodes = list(G.neighbors(i))
        neighbor_degrees_sum = sum(G.degree(j) for j in neighbor_nodes)
        neighbor_degree_sum_square = sum([G.degree(j) ** 2 for j in neighbor_nodes])
        T = nx.triangles(G, i)

        LC = (a ** 3) * (di ** 3) + 3 * a * ((1 - a) ** 2) * (di ** 2) + (
                    6 * (a ** 2) - 3 * a - 2 * (a ** 3)) * di + 3 * (a ** 3) * (
                     neighbor_degree_sum_square) + 3 * a * (a ** 2 - 4 * a + 2) * neighbor_degrees_sum + 6 * (
                         (1 - a) ** 3) * T

        LC_values.append(LC)
        #print(LC_values)
    NSC = []
    ks=k_shell(G)
    c=Cluster(G)

    for i in range(len(G)):
        sum1=(LC_values[i]*sum(ks[j] for j in list(G.neighbors(i))))/(math.exp(c[i]))
        NSC.append(sum1)
        #print(NSC)


    dis = distance(node_embeddings)
    MNEGC_values = []
    for i in range(num_nodes):
        F1 = 0
            # 使用 ego_graph 函数来获取节点的三阶邻域
        neighborhood = nx.ego_graph(G, i, radius=3)
            # 从邻域子图中删除中心节点
        neighborhood.remove_node(i)
        for j in range(num_nodes):
            if j in neighborhood.nodes():
                F1 += (NSC[i] * NSC[j] / dis[i][j] ** 2)
        MNEGC_values.append(F1)



    return MNEGC_values

def calculate_Ei_values_Netscience(G, A):
    num_nodes = A.shape[0]
    a=0.2
    LC_values = []  # 存储每个节点的 LC 值
    for i in G.nodes():
        di = G.degree(i)
        neighbor_nodes = list(G.neighbors(i))
        neighbor_degrees_sum = sum(G.degree(j) for j in neighbor_nodes)
        neighbor_degree_sum_square = sum([G.degree(j) ** 2 for j in neighbor_nodes])
        T = nx.triangles(G, i)

        LC = (a ** 3) * (di ** 3) + 3 * a * ((1 - a) ** 2) * (di ** 2) + (
                    6 * (a ** 2) - 3 * a - 2 * (a ** 3)) * di + 3 * (a ** 3) * (
                     neighbor_degree_sum_square) + 3 * a * (a ** 2 - 4 * a + 2) * neighbor_degrees_sum + 6 * (
                         (1 - a) ** 3) * T

        LC_values.append(LC)
        #print(LC_values)
    NSC = []
    ks=k_shell(G)
    c=Cluster(G)

    for i in range(len(G)):
        sum1=(LC_values[i]*sum(ks[j] for j in list(G.neighbors(i))))/(math.exp(c[i]))
        NSC.append(sum1)
        #print(NSC)


    dis = distance(node_embeddings)
    MNEGC_values = []
    for i in range(num_nodes):
        F1 = 0
            # 使用 ego_graph 函数来获取节点的三阶邻域
        neighborhood = nx.ego_graph(G, i, radius=3)
            # 从邻域子图中删除中心节点
        neighborhood.remove_node(i)
        for j in range(num_nodes):
            if j in neighborhood.nodes():
                F1 += (NSC[i] * NSC[j] / dis[i][j] ** 2)
        MNEGC_values.append(F1)
    return MNEGC_values



def sckendall_1(a, b):
    kendall_tau_2, p_value = kendalltau(a, b)  # kendalltau：系统自带肯德尔系数
#     # kendall_tau = sckendall(a, b)#sckendall：自己定义的肯德尔系数，好像
#     # print(kendall_tau_2)
#     # print(kendall_tau)
    return kendall_tau_2


