import numpy as np



class Edge(object):
    def __init__(self, source, destination, cost):
        self.source = source
        self.destination = destination
        self.cost = cost


class Graph(object):
    def __init__(self, file):
        f = open(file, 'r')
        self.N = int(f.readline())   #dim of matrix, every time read a line
        self.edges = []
        self.costs = np.zeros((self.N, self.N))  #all zeros 2-dimension matrices (10*10)
        for i in range(self.N):   #self.N ==10
            l = f.readline()
            a = l.split(" ")    #split a string l by space
            a = [value for value in a if value != '']
            for j in range(i + 1, self.N):  #only consider half of matrix
                self.costs[i, j] = a[j]
                self.costs[j, i] = a[j]
                self.edges.append(Edge(i, j, self.costs[i, j]))  #the class of Edge
        self.edges.sort(key=lambda x: x.cost)   #sorting according to cost

    def get_sorted_edges(self):
        return self.edges

    def get_edge(self, i, j):
        if i < j:
            return Edge(i, j, self.costs[i, j])
        else:
            return Edge(j, i, self.costs[i, j])
    def get_N(self):
        return self.N
    


a=Graph('N10.data')

