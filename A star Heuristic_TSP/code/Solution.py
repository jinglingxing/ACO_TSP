import copy

from Graph import Graph


class Solution(object):
    def __init__(self, s):
        if isinstance(s, Graph):
            ## be used to create the solution
            ## for root node of the search
            self.g = copy.deepcopy(s)
            self.cost = 0
            self.visited = []
            self.not_visited = [x for x in range(1, self.g.get_N())]
            
            
        elif isinstance(s, Solution):
            ## be used to create child node as
            ## copy of its solution's father and 
            ## then new edges can be added
            self.g = copy.deepcopy(s.g)
            self.cost = s.cost
            self.visited = copy.deepcopy(s.visited)
            self.not_visited = copy.deepcopy(s.not_visited)
        else:
            raise ValueError('you should give a graph or a solution')

    def add_edge(self, v, u):
        new_edge = self.g.get_edge(v, u)
        self.visited.append(u)
        if u != 0:
            self.not_visited.remove(u)
        self.cost = self.cost + new_edge.cost
        #raise NotImplementedError()

    def printf(self):
        
        print(0)
        for i in range(len(self.visited)):
            print(self.visited[i])
        print("COST=%d" %(self.cost))
        #raise NotImplementedError()