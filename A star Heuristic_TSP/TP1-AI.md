
<center><h1>**TP1-Report</h1>**</center>

<center><h5>*Name: Zhenxi LI(1883294) & Jinling XING(1915481)*</h5></center>





### 1. Introduction


The Travelling Salesman Problem (TSP) is defined as follows: given a set of n cities, and the distance between every pair of cities, find the shortest possible route that visits every city exactly once and returns to the starting point.
First, we implement a simple greedy algorithm to solve this problem which visits the best solution at present. Then, we use the MST as our heuristic to implement our heuristic algorithm. Finally, we try to add two values of distance to enhance our heuristic algorithm.

### 2. Implementation

#### 1. Solution

There are those methods required to implement in the solution framework：

- __Solution(g)__: We use a graph to initialize a solution and put all nodes in of the graph into not_visited list.
- __Solution(solution)__: We copy the reference of graph and current cost to the new solution, and make deep copy for visited list and not_visited list.
- __add_edge(v, u)__: We get the edge from graph which contains nodes v,u, add u to visited list and remove u from not_visited list since it is visited now. And add the cost of new edge to solution’s cost.
- __print()__: Print the visiting route with final cost.

```
class Solution(object):
    def __init__(self, s):
        if isinstance(s, Graph):
            ## be used to create the solution
            ## for root node of the search
            self.g = s
            self.cost = 0
            self.visited = []
            self.not_visited = [x for x in range(0, self.g.get_N())]
        elif isinstance(s, Solution):
            ## be used to create child node as
            ## copy of its solution's father and
            ## then new edges can be added
            self.g = s.g
            self.cost = s.cost
            self.visited = copy.deepcopy(s.visited)
            self.not_visited = copy.deepcopy(s.not_visited)
        else:
            raise ValueError('you should give a graph or a solution')

    def add_edge(self, v, u):
        new_edge = self.g.get_edge(v, u)
        self.visited.append(u)
        if u in self.not_visited:
            self.not_visited.remove(u)
        self.cost = self.cost + new_edge.cost
        #raise NotImplementedError()

    def print(self):
        for i in range(len(self.visited)):
            print(self.visited[i])
        print("COST=%d" %(self.cost))
        #raise NotImplementedError()

```



#### 2.A_star
There are those methods required to implement in the A star framework：
- __isN2betterThanN1(N1, N2)__: We use the function f(i)=g(i)+h(i) to compare nodes, used for heap sorting.
- __main()__: Initialize graph, initial solution, Kruskal, root node of the heap and start search. We will keep explore nodes if there is node in heap. Once there is only one node in the heap, it should be the SOURCE NODE, we will explore it directly. When everything is finished, we will print the result.
- __explore_node()__: We traverse all those unvisited cities as current child node, explore the node and put it into heap. There are three kinds of strategies for exploring node,
   - Without heuristic algorithm.
     The heuristic cost will be 0
   - Using MST as heuristic(use flag USEKRUSKAL to control it)
   Using funciton Kruskal.getMSTCost() to calculate the MST cost for remaining nodes
   - Along side with the MST heuristic, we add ***Distance from the current city v to the nearest unvisited cit*** and ***Nearest distance from an unvisited city v to the start city***(use flag TIGHTENBOUND)
Along side with MST cost, we find the nearest unvisited city and the nearest unvisited city for our SOURCE city in current child’s unvisited cities and add these distance to node's heuristic cost.
```

#from heapqueue.binary_heap import BinaryHeap
import queue as Q
from datetime import datetime

from Graph import Graph
import Kruskal
from Solution import Solution

SOURCE = 0
USEKRUSKAL = False
TIGHTENBOUND = False
MAXINT = 2147483647
explored_node = 0
created_node = 0



class Node(object):
    def __init__(self, v, sol, heuristic_cost=0):
        self.v = v
        self.solution = sol
        self.heuristic_cost = heuristic_cost

    def explore_node(self, heap):
        global explored_node, created_node
        explored_node += 1
        for v in self.solution.not_visited:
            if v == SOURCE:
                continue
            child_sol = Solution(self.solution)
            child_sol.add_edge(self.v, v)
            child_heuristic_cost = 0
            if USEKRUSKAL:
                child_heuristic_cost = Kruskal.kruskal.getMSTCost(child_sol, SOURCE)

            if TIGHTENBOUND:
                minimum_next_cost = MAXINT
                for nv in child_sol.not_visited:
                    next_edge = child_sol.g.get_edge(v, nv)
                    if minimum_next_cost > next_edge.cost:
                        minimum_next_cost = next_edge.cost

                nearest_to_start_cost = MAXINT
                for nv in child_sol.not_visited:
                    prestart_edge = child_sol.g.get_edge(SOURCE, nv)
                    if nearest_to_start_cost > prestart_edge.cost:
                        nearest_to_start_cost = prestart_edge.cost
                child_heuristic_cost += (minimum_next_cost + nearest_to_start_cost)

            child_node = Node(v, child_sol, child_heuristic_cost)
            heap.put(child_node)
            created_node += 1
        #raise NotImplementedError()

    def __lt__(self, N2):
        return not isN2betterThanN1(self, N2)
        #raise NotImplementedError

def isN2betterThanN1(N1, N2):
    f1 = N1.solution.cost + N1.heuristic_cost
    f2 = N2.solution.cost + N2.heuristic_cost
    visited_nodes1 = len(N1.solution.visited)
    visited_nodes2 = len(N2.solution.visited)
    if (f1 > f2):
        return True
    # Add visited nodes to determine when same f values get
    elif (f1 == f2) and (visited_nodes1 < visited_nodes2):
        return True
    else:
        return False
    #raise NotImplementedError

def main(data):
    print(data)
    a=datetime.now()
    g = Graph(data)
    Kruskal.kruskal = Kruskal.Kruskal(g)
    heap = Q.PriorityQueue()
    sol = Solution(g)
    sol.visited.append(SOURCE)
    root = Node(SOURCE, sol)
    ## init explored_node created_node
    global explored_node, created_node
    explored_node = 0
    created_node = 1
    heap.put(root)
    while heap.qsize() > 0:
        node = heap.get()
        if len(node.solution.not_visited) == 1:
            child_sol = Solution(node.solution)
            child_sol.add_edge(node.v, SOURCE)
            child_node = Node(SOURCE, child_sol)
            explored_node += 1
            created_node += 1
            heap.put(child_node)
        else:
            node.explore_node(heap)
        if len(node.solution.not_visited) == 0:
            if SOURCE == node.solution.visited[len(node.solution.visited)-1]:
                node.solution.print()
                b=datetime.now()
                print(b-a)
                print("explored_node: %d" % explored_node)
                print("created_node: %d" % created_node)
                return
    

if __name__ == '__main__':

    USEKRUSKAL = False
    print("without kruskal:")
    main("N10.data")
    main("N12.data")
    USEKRUSKAL = True
    print("with kruskal:")
    main("N10.data")
    main("N12.data")
    main("N15.data")
    TIGHTENBOUND = True
    print("with tighten bound:")
    main("N10.data")
    main("N12.data")
    main("N15.data")
    main("N17.data")
    print("over")



```
#### 3. Kruskal

There are those methods required to implement in the kruskal framework：


- __getMSTCost(sol, source)__: Firstly, we add those visited edges into unionfind. Then, search all edges in graph to check whether it can make cycle. If no cycle, it will be added into MST.


```
import numpy as np

kruskal = None


class UnionFind(object):
    def __init__(self, n):
        self.n = n
        self.parent = np.array(range(n))
        self.rnk = np.zeros(n)

    def reset(self):
        self.parent = np.array(range(self.n))
        self.rnk = np.zeros(self.n)

    def add(self, e):
        x = self.find(e.source)
        y = self.find(e.destination)

        if self.rnk[x] > self.rnk[y]:
            self.parent[y] = x
        else:
            self.parent[x] = y
        if self.rnk[x] == self.rnk[y]:
            self.rnk[y] += 1

    def makes_cycle(self, e):
        return self.find(e.source) == self.find(e.destination)

    def find(self, u):
        if u != self.parent[u]:
            return self.find(self.parent[u])
        else:
            return u


class Kruskal(object):

    def __init__(self, g):
        self.uf = UnionFind(g.N)
        self.g = g

    def getMSTCost(self, sol, source):
        assert(sol.visited[0] == source)
        self.uf.reset()
        mstcost = 0
        pre = source

        for v in sol.visited[1:len(sol.visited)]:
            edge = self.g.get_edge(pre, v)
            self.uf.add(edge)
            pre = v
        for edge in self.g.get_sorted_edges():
            if not self.uf.makes_cycle(edge):
                self.uf.add(edge)
                mstcost += edge.cost
        return mstcost
        #raise NotImplementedError()







```



### 3. Results

We answered questions of 3, 4.1, 4.2, and show their result: Without heuristic cost, Using MST as heuristic cost, Using MST and two values of distance as heuristic cost.


__Without heuristic cost__

*(Execution. Now run your A star algorithm to the N10.data.txt and N12.data datasets and report your experiments! (All the results are provided in the Appendix A.))*
- Printed Solution;
- Number of explored nodes;
- Number of created nodes;
- CPU time of execution.


| **Data** | **Printed Solution**   | **Explored Nodes**  | **Created Nodes**   | **CPU Time**  |
|:---------|:-----------------------|:--------------------|:---------------------------|:--------------|
|N10|0-4-9-2-5-3-6-1-8-7-0 COST=135 |41897                |117227| 00.00.05.200996|
|N12|0-5-7-11-3-1-8-6-10-4-2-9-0 COST=1733|1164744|4136575| 00.03.57.183243|

Using Kruskal algorithm to rerun the A_star, from the N10 dataset we can see that the route of nodes is different, but the cost is same as 135. The reason why the N10 dataset has two kinds of routes with Kruskal and without Kruskal is because there exists at least two answers of the minimum cost of this dataset.

*(Add visited nodes to determine when same f values get)*

| **Data** | **Printed Solution**   | **Explored Nodes**  | **Created Nodes**   | **CPU Time**  |
|:---------|:-----------------------|:--------------------|:---------------------------|:--------------|
|N10|0-7-8-1-9-2-5-3-6-4-0 COST=135|41586|116527|0.00.06.108365|
|N12|0-9-2-4-10-6-8-1-3-11-7-5-0 COST=1733| 1163448|4132640|0.04.18.965729|

Explanation: When the A* function get same f values, we use the number of visited nodes in solution to determine which one(N1/N2) is better. And we find that we can get the optimal cost but the solution is different from former result. In addition, we can find that the number of explored_node and created_node are less, which we think is a small improvements.

**************************************************************************************************************************
__Using MST as heuristic cost__

*(Execution. Now run your A star algorithm using the MST as heuristic to the N10.data, N12.data and N15.data. Report all your results and improvements.)*

| **Data** | **Printed Solution**   | **Explored Nodes**  | **Created Nodes**   | **CPU Time**  |
|:---------|:-----------------------|:--------------------|:---------------------------|:--------------|
|N10|0-7-8-1-9-2-5-3-6-4-0 COST=135|16128|45269|0.00.11.209958|
|N12|0-5-7-11-3-1-8-6-10-4-2-9-0 COST=1733|110489|370705|0.01.49.424157|
|N15|0-10-3-5-7-9-13-11-2-6-4-8-14-1-12-0 COST=291|26520|135631|0.00.57.868727|

Improvements: Comparing with the result to the result of algorithm without heuristic cost, we can find that it use less time, explore and create less nodes to achieve the final solution when the number of nodes become larger.


*（Add visited nodes to determine when same f values get）*
| **Data** | **Printed Solution**   | **Explored Nodes**  | **Created Nodes**   | **CPU Time**  |
|:---------|:-----------------------|:--------------------|:---------------------------|:--------------|
|N10|0-4-9-2-5-3-6-1-8-7-0 COST=135|15710|44230|0.00.09.881042|
|N12|0-5-7-11-3-1-8-6-10-4-2-9-0 COST=135|110273|370035|0.01.53.186562|
|N15|0-12-1-14-8-4-6-2-11-13-9-7-5-3-10-0 COST=291|26214|133813|0.01.02.534498|


**************************************************************************************************************************

__Using MST and two values of distance as heuristic cost__

*(Execution. Rerun your experiments and reports your gains.)*

| **Data** | **Printed Solution**   | **Explored Nodes**  | **Created Nodes**   | **CPU Time**  |
|:---------|:-----------------------|:--------------------|:---------------------------|:--------------|
|N10|0-4-6-3-5-2-9-1-8-7-0 COST=135|6660|22221|0.00.05.140642|
|N12|0-9-2-4-10-6-8-1-3-11-7-5-0 COST=1733 |18519|78421|0.00.24.231726|
|N15|0-12-1-14-8-4-6-2-11-13-9-7-5-3-10-0 COST=291|606|3966|0.00.01.842305|

Gains: Comparing with previous results, we can find that cost time, explored nodes and created nodes become less after adding the two values of distance.


*（Add visited nodes to determine when same f values get)*
| **Data** | **Printed Solution**   | **Explored Nodes**  | **Created Nodes**   | **CPU Time**  |
|:---------|:-----------------------|:--------------------|:---------------------------|:--------------|
|N10|0-7-8-1-6-3-5-2-9-4-0 COST=135|6640| 22163|0.00.05.108651|
|N12|0-9-2-4-10-6-8-1-3-11-7-5-0 COST=1733| 18450| 78172|0.00.24.340330|
|N15|0-12-1-14-8-4-6-2-11-13-9-7-5-3-10-0 COST=291|553|3658|0.00.01.809288|
