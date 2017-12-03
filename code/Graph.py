import numpy as np


class Edge(object):
    def __init__(self, source, destination, cost):
        self.source = source
        self.destination = destination
        self.cost = cost


class Graph(object):
    def __init__(self, file):
        f = open(file, 'r')
        self.N = int(f.readline())
        # self.edges = []
        self.costs = np.zeros((self.N, self.N))
        for i in range(self.N):
            l = f.readline()
            a = l.split(" ")
            a = [value for value in a if value != '']
            for j in range(i + 1, self.N):
                self.costs[i, j] = a[j]
                self.costs[j, i] = a[j]
                # self.edges.append(Edge(i, j, self.costs[i, j]))
        # self.edges.sort(key=lambda x: x.cost)

    # def get_sorted_edges(self):
    #     return self.edges

    def get_edge(self, i, j):
        if i < j:
            return Edge(i, j, self.costs[i, j])
        else:
            return Edge(j, i, self.costs[i, j])
