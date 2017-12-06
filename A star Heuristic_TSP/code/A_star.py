#from heapqueue.binary_heap import BinaryHeap
import Queue as Q

from Graph import Graph

import Kruskal
from Solution import Solution

SOURCE = 0

class Node(object):
    def __init__(self, v, sol, heuristic_cost=0):
        self.v = v
        self.solution = sol
        self.heuristic_cost = heuristic_cost

    def explore_node(self, heap):
        for v in self.solution.not_visited:
            child_sol = Solution(self.solution)
            child_sol.add_edge(self.v, v)
           #child_node = Node(v, child_sol)
            child_node = Node(v, child_sol, Kruskal.kruskal.getMSTCost(child_sol, SOURCE))         
            heap.put(child_node)
        #raise NotImplementedError()

    def __lt__(self, N2):
     #  f1 = self.solution.cost + self.heuristic_cost
        f1 = self.getMSTCost + self.heuristic_cost
    #   f2 = N2.solution.cost + N2.heuristic_cost
        f2 = N2.getMSTCost + N2.heuristic_cost
        if (f1 > f2):
            return False
        else:
            return True
        #raise NotImplementedError

def main():
    g = Graph("N10.data")
    Kruskal.kruskal = Kruskal.Kruskal(g)
    heap = Q.PriorityQueue()
    sol = Solution(g)
    root = Node(SOURCE, sol)
    heap.put(root)
    
    while heap.qsize() > 0:
        node = heap.get()
        print("curCost: %d" %node.solution.cost)
        
  
        
        if len(node.solution.not_visited) == 0:
            if SOURCE in node.solution.visited:
                node.solution.printf()
                return
                
            else:
                child_sol = Solution(node.solution)
                child_sol.add_edge(node.v, SOURCE)
                child_node = Node(SOURCE, child_sol)
                heap.put(child_node)
                
             
        else:
            node.explore_node(heap)
 

def isN2betterThanN1(N1, N2):
    f1 = N1.solution.cost + N1.heuristic_cost
    f2 = N2.solution.cost + N2.heuristic_cost
    if (f1 > f2):
        return True
    else:
        return False
    #raise NotImplementedError


if __name__ == '__main__':
    main()
