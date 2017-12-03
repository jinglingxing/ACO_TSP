from ACO import ACO
from Solution import Solution
import copy

def test_heuristic2opt():
    print("testing heuristic 2opt...")
    aco = ACO(0, 0, 0, 0, 0, 'test')
    s = Solution(aco.graph)
    s.add_edge(0, 2)
    s.add_edge(2, 3)
    s.add_edge(3, 1)
    s.add_edge(1, 0)
    aco.heuristic2opt(s)
    assert s.cost == 6
    print('ok')


def test_local_update():
    print("testing local update...")
    phi = 0.5
    aco = ACO(0, 0, 0, phi, 0, 'test')
    s = Solution(aco.graph)
    aco.pheromone[:, :] = 1
    c = copy.copy(aco.pheromone)
    s.add_edge(0, 2)
    s.add_edge(2, 3)
    s.add_edge(3, 1)
    s.add_edge(1, 0)
    aco.local_update(s)
    assert (c != aco.pheromone).sum() == 8
    assert abs(aco.pheromone[0, 1] - 0.833) < 1e-3
    assert abs(aco.pheromone[0, 2] - 0.833) < 1e-3
    assert abs(aco.pheromone[3, 1] - 0.833) < 1e-3
    assert abs(aco.pheromone[3, 2] - 0.833) < 1e-3
    print('ok')


## ??? assert for [0,2],[2,3],[3,1],[1,0] is wrong
def test_global_update():
    print('testing global update...')
    rho = 0.1
    aco = ACO(0, 0, rho, 0, 0, 'test')
    s = Solution(aco.graph)
    s.add_edge(0, 2)
    s.add_edge(2, 3)
    s.add_edge(3, 1)
    s.add_edge(1, 0)
    aco.pheromone[:, :] = 1
    aco.global_update(s)
    print(aco.pheromone)
    assert abs(aco.pheromone[0, 3] - 0.9) < 1e-3
    assert abs(aco.pheromone[1, 2] - 0.9) < 1e-3
    # print(abs(aco.pheromone[0, 2] - 0.9))
    assert abs(aco.pheromone[0, 2] - 0.91) < 1e-3
    assert abs(aco.pheromone[2, 3] - 0.91) < 1e-3
    assert abs(aco.pheromone[3, 1] - 0.91) < 1e-3
    assert abs(aco.pheromone[1, 0] - 0.91) < 1e-3
    print('ok')


def test_next_city():
    print('testing get next city...')
    aco = ACO(0.5, 2, 0, 0, 0, 'test')
    s = Solution(aco.graph)
    c1 = 0
    c3 = 0
    for i in range(1000):
        c = aco.get_next_city(s)
        if c == 3:
            c3 += 1
        elif c == 1:
            c1 += 1
    assert (abs(c3 - 870) < 30)
    assert (abs(c1 - 90) < 30)
    print('ok')


if __name__ == '__main__':
#    test_heuristic2opt()
    test_global_update()
#    test_local_update()
#    test_next_city()

#    aco = ACO(0.9, 2, 0.1, 0.1, 10, 'testa')
#    aco.runACO(10)
