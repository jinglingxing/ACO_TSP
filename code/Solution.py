import numpy as np

from Graph import Graph


class Solution(object):
    def __init__(self, s=None):
        if isinstance(s, Graph):
            self.g = s
            self.cost = 0
            self.visited = []
            self.not_visited = list(range(s.N))
        elif isinstance(s, Solution):
            self.g = s.g
            self.cost = s.cost
            self.visited = s.visited[:]
            self.not_visited = s.not_visited[:]
        else:
            raise ValueError('you should give a graph or a solution')

    def add_edge(self, v, u):
        # print(u)
        self.cost += self.g.get_edge(v, u).cost
        self.visited.append(u)
        self.not_visited.remove(u)

    def printf(self):
        s = '['
        for i in self.visited:
            s += str(i) + ', '
        s = s[:-2]
        s += ']'
        print(s)
        print('cost: ' + str(self.cost))

    def inverser_ville(self, i, j):
        if j < i:
            return self.inverser_ville(j, i)
        vis = np.array(self.visited)
        vis[range(i + 1, j + 1)] = vis[range(j, i, -1)]
        self.visited = list(vis)

    def get_cost(self, Source):
        v = Source
        c = 0
        for i in self.visited:
            c += self.g.costs[v, i]
            v = i
        return c
