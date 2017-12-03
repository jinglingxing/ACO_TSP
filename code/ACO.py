import numpy as np
import random
import time
import threading

from Graph import Graph
from Solution import Solution


SOURCE = 0


class ACO(object):
    def __init__(self, q0, beta, rho, phi, K, data):
        self.parameter_q0 = q0
        self.parameter_beta = beta
        self.parameter_rho = rho
        self.parameter_phi = phi
        self.parameter_K = K

        self.graph = Graph(data)
        self.best = Solution(self.graph)
        self.best.cost = 99999999999999
        self.pheromone_init = np.ones((self.graph.N, self.graph.N))
        f = open(data + '_init', 'r')
        self.pheromone_init *= float(f.readline())
        self.pheromone = np.ones((self.graph.N, self.graph.N))

    def get_next_city(self, sol):
        q = ((float)(random.randint(0,100)))/100
        tmpI = SOURCE
        if (len(sol.not_visited) == 1):
            return SOURCE
        if (len(sol.visited) > 0) :
            tmpI = sol.visited[len(sol.visited)-1]
        ratioDic = {}
        for j in sol.not_visited:
            if (j == SOURCE):
                continue
            tmpRatio = 0
            tmpRatio = ((float)(self.pheromone[tmpI,j]))/(self.graph.costs[tmpI,j]**self.parameter_beta)
            ratioDic[j] = tmpRatio
        if q <= self.parameter_q0:
            return max(ratioDic, key=ratioDic.get)
        else:
            mulRatio = {}
            for key, value in ratioDic.items():
                mulRatio[key] = (int)(round(value*1000))
            tmpRan = random.randint(0, sum(mulRatio.values()))
            for key, value in mulRatio.items():
                if (tmpRan <= value):
                    return key
                tmpRan = tmpRan-value  
        # raise NotImplementedError()

    def heuristic2opt(self, sol):
        for i in range(0, len(sol.visited)):
            for j in range(0, len(sol.visited)):
                if i == j or (i-1+len(sol.visited))%len(sol.visited) == j or (i+1)%len(sol.visited) == j:
                    continue
                oldCost = self.graph.costs[sol.visited[i], sol.visited[(i+1)%len(sol.visited)]] + self.graph.costs[sol.visited[j], sol.visited[(j+1)%len(sol.visited)]]
                newCost = self.graph.costs[sol.visited[i], sol.visited[j]] + self.graph.costs[sol.visited[(i+1)%len(sol.visited)], sol.visited[(j+1)%len(sol.visited)]]
                if (newCost < oldCost):
                    sol.inverser_ville(i, j)
                    sol.cost = sol.cost + newCost - oldCost
        # raise NotImplementedError()

        
    def global_update(self, sol):
        bestLen = sol.cost
        for i in range(0, self.graph.N):
            for j in range(i, self.graph.N):
                indexI = sol.visited.index(i)
                indexJ = sol.visited.index(j)
                if ((i == SOURCE and indexJ == SOURCE) or (abs(indexI-indexJ) == 1)):
                    self.pheromone[i,j] = (1-self.parameter_rho)*self.pheromone[i,j]+self.parameter_rho*(float(1)/bestLen)
                    self.pheromone[j,i] = self.pheromone[i,j]
                else:
                    self.pheromone[i,j] = (1-self.parameter_rho)*self.pheromone[i,j]
                    self.pheromone[j,i] = self.pheromone[i,j]
        # raise NotImplementedError()

    def local_update(self, sol):
        start = SOURCE
        end = -1
        for i in range(0, self.graph.N):
            end = sol.visited[i]
            self.pheromone[start,end] = (1-self.parameter_phi)*self.pheromone[start,end]+self.parameter_phi*self.pheromone_init[start,end]
            self.pheromone[end,start] = self.pheromone[start,end]
            start = end
        # raise NotImplementedError()

    def runACO(self, maxiteration):
        startTime = time.time()
        for ite in range(0, maxiteration):
            for k in range(0, self.parameter_K):
                tmpSol = Solution(self.graph)
                start = SOURCE
                while (len(tmpSol.not_visited) != 0):
                    nextNode = self.get_next_city(tmpSol)
                    tmpSol.add_edge(start, nextNode)
                    start = nextNode
                self.local_update(tmpSol)
                if tmpSol.cost < self.best.cost:
                    self.best = tmpSol
            self.heuristic2opt(self.best)
            self.global_update(self.best)
        endTime = time.time()
        return (endTime-startTime)
        # raise NotImplementedError()

    def optimizeGap(self, maxiteration, OPT):
        startTime = time.time()
        lastBest = -1
        lastImp = 0
        for ite in range(0, maxiteration):
            for k in range(0, self.parameter_K):
                tmpSol = Solution(self.graph)
                start = SOURCE
                while (len(tmpSol.not_visited) != 0):
                    nextNode = self.get_next_city(tmpSol)
                    tmpSol.add_edge(start, nextNode)
                    start = nextNode
                self.local_update(tmpSol)
                if tmpSol.cost < self.best.cost:
                    self.best = tmpSol
            self.heuristic2opt(self.best)
            self.global_update(self.best)
            
            if (ite+1)==1 or (ite+1)%100==0:
                print("iteration: " + str(ite+1))
                gapBest = (self.best.cost-OPT)/OPT
                print("gapBest: " + str(gapBest))
                print("without Imp: " + str(ite-lastImp))
                curTime = time.time()
                print("CPU Time: " + str(curTime-startTime))
                
            if self.best.cost!=lastBest:
                lastBest = self.best.cost
                lastImp = ite

default_q0 = 0.9
default_beta = 2
default_rho = 0.1
default_phi = 0.1
default_K = 10
default_iteration = 1000

tune_default_time = None
tune_default_cost = None

tune_q0_time = []
tune_q0_cost = []
tune_beta_time = []
tune_beta_cost = []
tune_rho_time = []
tune_rho_cost = []
tune_phi_time = []
tune_phi_cost = []
tune_K_time = []
tune_K_cost = []

def thread_tune_default():
    global tune_default_time, tune_default_cost
    global default_q0, default_beta, default_rho
    global default_phi, default_K, default_iteration
    
    default_time = []
    default_cost = []
    print("Start default...")
    for i in range(1,6):
        tmpAco = ACO(default_q0, default_beta, default_rho, default_phi, default_K, 'qatar')
        tmpTime = tmpAco.runACO(default_iteration)
        tmpCost = tmpAco.best.cost
        default_time.append(tmpTime)
        default_cost.append(tmpCost)
        print(tmpTime)
        print(tmpCost)
    tune_default_time = np.mean(default_time)
    tune_default_cost = np.mean(default_cost)
    print("tune_default_time: " + str(tune_default_time))
    print("tune_default_cost: " + str(tune_default_cost))


def thread_tune_q0():
    global tune_q0_time, tune_q0_cost
    global default_q0, default_beta, default_rho
    global default_phi, default_K, default_iteration
    
    ### tuning q0
    q0_start = 0
    q0_step = 0.1
    q0_end = 1
    
    print("Start q0...")
    while q0_start <= q0_end:
        cur_q0_time = []
        cur_q0_cost = []
        for i in range(1,6):
            tmpAco = ACO(q0_start, default_beta, default_rho, default_phi, default_K, 'qatar')
            tmpTime = tmpAco.runACO(default_iteration)
            tmpCost = tmpAco.best.cost
            cur_q0_time.append(tmpTime)
            cur_q0_cost.append(tmpCost)
            print(tmpTime)
            print(tmpCost)
        tune_q0_time.append(np.mean(cur_q0_time))
        tune_q0_cost.append(np.mean(cur_q0_cost))
        q0_start = q0_start+q0_step
    print("tune_q0_time: " + str(tune_q0_time))
    print("tune_q0_cost: " + str(tune_q0_cost))


def thread_tune_beta():
    global tune_beta_time, tune_beta_cost
    global default_q0, default_beta, default_rho
    global default_phi, default_K, default_iteration
    
    ### tuning beta
    beta_start = 0
    beta_step = 0.5
    beta_end = 3
    
    print("Start beta...")
    while beta_start <= beta_end:
        cur_beta_time = []
        cur_beta_cost = []
        for i in range(1,6):
            tmpAco = ACO(default_q0, beta_start, default_rho, default_phi, default_K, 'qatar')
            tmpTime = tmpAco.runACO(default_iteration)
            tmpCost = tmpAco.best.cost
            cur_beta_time.append(tmpTime)
            cur_beta_cost.append(tmpCost)
            print(tmpTime)
            print(tmpCost)
        tune_beta_time.append(np.mean(cur_beta_time))
        tune_beta_cost.append(np.mean(cur_beta_cost))
        beta_start = beta_start+beta_step
    print("tune_beta_time: " + str(tune_beta_time))
    print("tune_beta_cost: " + str(tune_beta_cost))
    
    
def thread_tune_rho():
    global tune_rho_time, tune_rho_cost
    global default_q0, default_beta, default_rho
    global default_phi, default_K, default_iteration
    
    ### tuning rho
    rho_start = 0
    rho_step = 0.1
    rho_end = 1
    
    print("Start rho...")
    while rho_start <= rho_end:
        cur_rho_time = []
        cur_rho_cost = []
        for i in range(1,6):
            tmpAco = ACO(default_q0, default_beta, rho_start, default_phi, default_K, 'qatar')
            tmpTime = tmpAco.runACO(default_iteration)
            tmpCost = tmpAco.best.cost
            cur_rho_time.append(tmpTime)
            cur_rho_cost.append(tmpCost)
            print(tmpTime)
            print(tmpCost)
        tune_rho_time.append(np.mean(cur_rho_time))
        tune_rho_cost.append(np.mean(cur_rho_cost))
        rho_start = rho_start+rho_step
    print("tune_rho_time: " + str(tune_rho_time))
    print("tune_rho_cost: " + str(tune_rho_cost))
    
    
def thread_tune_phi():
    global tune_phi_time, tune_phi_cost
    global default_q0, default_beta, default_rho
    global default_phi, default_K, default_iteration
    
    ### tuning phi
    phi_start = 0
    phi_step = 0.1
    phi_end = 1
    
    print("Start phi...")
    while phi_start <= phi_end:
        cur_phi_time = []
        cur_phi_cost = []
        for i in range(1,6):
            tmpAco = ACO(default_q0, default_beta, default_rho, phi_start, default_K, 'qatar')
            tmpTime = tmpAco.runACO(default_iteration)
            tmpCost = tmpAco.best.cost
            cur_phi_time.append(tmpTime)
            cur_phi_cost.append(tmpCost)
            print(tmpTime)
            print(tmpCost)
        tune_phi_time.append(np.mean(cur_phi_time))
        tune_phi_cost.append(np.mean(cur_phi_cost))
        phi_start = phi_start+phi_step
    print("tune_phi_time: " + str(tune_phi_time))
    print("tune_phi_cost: " + str(tune_phi_cost))    
    
    
def thread_tune_K():
    global tune_K_time, tune_K_cost
    global default_q0, default_beta, default_rho
    global default_phi, default_K, default_iteration

    ### tuning K
    K_start = 5
    K_step = 5
    K_end = 40
    
    print("Start K...")
    while K_start <= K_end:
        cur_K_time = []
        cur_K_cost = []
        for i in range(1,6):
            tmpAco = ACO(default_q0, default_beta, default_rho, phi_start, default_K, 'qatar')
            tmpTime = tmpAco.runACO(default_iteration)
            tmpCost = tmpAco.best.cost
            cur_K_time.append(tmpTime)
            cur_K_cost.append(tmpCost)
            print(tmpTime)
            print(tmpCost)
        tune_K_time.append(np.mean(cur_K_time))
        tune_K_cost.append(np.mean(cur_K_cost))
        K_start = K_start+K_step
    print("tune_K_time: " + str(tune_K_time))
    print("tune_K_cost: " + str(tune_K_cost))



if __name__ == '__main__':

        
    ##### Code for tuning, we use multi-executions to run it 
    #if sys.argv[1]=='default':
    #    thread_tune_default()    
    #if sys.argv[1]=='q0':
    #    thread_tune_q0()
    #if sys.argv[1]=='beta':
    #    thread_tune_beta()
    #if sys.argv[1]=='rho':
    #    thread_tune_rho()
    #if sys.argv[1]=='phi':
    #    thread_tune_phi()
    #if sys.argv[1]=='K':
    #    thread_tune_K()

    ## After tuning, we use 
    # q0 = 1.0  beta = 2  rho = 0.2
    # phi = 1.0  K = 20
    ##### Optimize Gap
    UruACO = ACO(1.0, 2, 0.2, 1.0, 20, "uruguay")
    UruACO.optimizeGap(1000, 79114)
    CanACO = ACO(1.0, 2, 0.2, 1.0, 20, "canada")
    CanACO.optimizeGap(1000, 1290319)
            
    print("OVER")
    
    

    

