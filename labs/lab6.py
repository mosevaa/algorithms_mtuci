import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy
import time
inf = np.iinfo('int').max

def read_graph(path):
    matrix = []
    with open(path, 'r') as f:
        for line in f:
            numbers = []
            for num in line.split():
                numbers.append(int(num))
            matrix.append(numbers)
    return nx.from_numpy_matrix(np.matrix(matrix), create_using=nx.DiGraph)

def get_matrix_with_inf(matrix):
    inf = np.iinfo('int').max
    new_matrix=[]
    for i in range(len(matrix)):
        new_matrix.append([])
        for j in range(len(matrix)):
            if matrix[i, j] == 0 and j != i:
                new_matrix[i].append(inf-1000)
            else:
                new_matrix[i].append(matrix[i, j])
    return new_matrix

def draw_graph(graph, path = None):
    colors = None
    if path:
        colors = []
        for item in graph.edges():
            try:
                index1 = path.index(item[0])
                index2 = path.index(item[1])
                if index1 + 1 != index2 and index1 - 1 != index2:
                    raise
                colors.append('red')
            except:
                colors.append('black')
    pos = nx.circular_layout(graph)
    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_nodes(graph, node_color = 'green', node_size = 200, pos = pos)
    nx.draw_networkx_edges(graph,edge_color = colors, pos = pos, arrowstyle = 'simple, tail_width = 0.05')
    nx.draw_networkx_labels(graph, pos = pos)
    nx.draw_networkx_edge_labels(graph, pos = pos, edge_labels = labels)

def get_path(for_path, start, end):
    path = [end]
    while end != start:
        end = for_path[path[-1]]
        path.append(end)
    return path

def ford_bellman(graph, start):
    matrix = nx.adjacency_matrix(graph).todense()
    matrix = get_matrix_with_inf(matrix)
    n = len(matrix)
    for_path = [0] * n
    table = [inf for i in range(n)]
    table[start] = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                w = table[j] + matrix[j][k]
                if w < table[k] and w > 0:
                    table[k] = w
                    for_path[k] = j
    return table, for_path

def yen(graph, start, end, k):
    slov = {}
    candidates = []
    table, for_path = ford_bellman(graph, start)
    print(table)
    matrix = nx.adjacency_matrix(graph).todense()
    adj_mat = get_matrix_with_inf(matrix)
    path = get_path(for_path, start, end)
    print(path)
    dist = table[end]
    slov[dist] = list(reversed(path))
    print(slov)
    candidates_length = []
    while len(slov) != k:
        root_path = []
        g = copy.deepcopy(adj_mat)
        for i in range(len(slov[dist]) - 1):
            if i != 0:
                root_path.append(slov[dist][i-1])
            g[i][slov[dist][i+1]] = inf
            spur_table, spur_for_path = ford_bellman(nx.from_numpy_matrix(np.array(g)), slov[dist][i])
            print(spur_table)
            spur_path = get_path(spur_for_path, slov[dist][i], end)
            spur_path = list(reversed(spur_path))
            total_path = root_path + spur_path
            if total_path not in candidates and total_path not in slov.values():
                candidates.append(total_path)
                candidates_length.append(spur_table[end])
    return candidates


graph = read_graph('labs/mat.txt')
start = 0
end = 7
k = 3
print(yen(graph, start, end, k))



