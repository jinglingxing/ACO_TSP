import numpy as np

kruskal = None

class UnionFind(object):
    #initialise the tree as empty
    #find a cycle in the tree
    def __init__(self, n):
        self.n = n
        self.parent = np.array(range(n))
        self.rnk = np.zeros(n)  #all zero n-dimension matrix

    def reset(self):
        self.parent = np.array(range(self.n))
        self.rnk = np.zeros(self.n)
    
    # check the node's parents nodes
    def find(self, u):
        if u != self.parent[u]:
            return self.find(self.parent[u])
        else:
            return u    
                
        #add the edge e to the tree structure
    def add(self, e):
        x = self.find(e.source)  #X is the source node of arc 'e'
        y = self.find(e.destination)  #y is the destination node of e

        if self.rnk[x] > self.rnk[y]:  #rank of X and Y
            self.parent[y] = x
        else:
            self.parent[x] = y
        if self.rnk[x] == self.rnk[y]:
            self.rnk[y] += 1

    #returns true if and only if a cycle is formed at adding e.
    def makes_cycle(self, e):
        return self.find(e.source) == self.find(e.destination)


class Kruskal(object):
    def __init__(self, g):
            self.uf = UnionFind(g.N)
            self.g = g

    def getMSTCost(self, sol, source):       
       # add all edges of sol to uf 
       # complete the tree using krukal algorithm
       #and compute the cost of the added edges        
       # return this cost
       # assert(sol.visited[0] == source)      
       #minimum_spanning_tree = []  
       #minimum_spanning_tree.insert(0, source)
       #sol=self.g.solution.visited
       #sol.visited[0] == source
        assert(sol.visited[0] == source)
        self.uf.reset()
        mst_cost=0
        
         
        for v in range(len(sol.visited)):
            e=self.g.get_edge(v,source)    
            self.uf.add(e)
            source=v
            
        for e in self.g.get_sorted_edges():  
            if e.source in sol.visited and e.destination in sol.visited:
                continue
            if not (self.uf.makes_cycle(e)):
                self.uf.add(e)
                mst_cost= mst_cost +e.cost      
                #minimum_spanning_tree.add(e)
        #return sorted(minimum_spanning_tree)
        return mst_cost