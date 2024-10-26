import numpy as np
import matplotlib.pyplot as plt
from collections import deque

city_coords = np.array([
    [1304, 2312], [3639, 1315], [4177, 2244], [3712, 1399], [3488, 1535],
    [3326, 1556], [3238, 1229], [4196, 1044], [4312, 790], [4386, 570],
    [3007, 1970], [2562, 1756], [2788, 1491], [2381, 1676], [1332, 695],
    [3715, 1678], [3918, 2179], [4061, 2370], [3780, 2212], [3676, 2578],
    [4029, 2838], [4263, 2931], [3429, 1908], [3507, 2376], [3394, 2643],
    [3439, 3201], [2935, 3240], [3140, 3550], [2545, 2357], [2778, 2826],
    [2370, 2975]
])

def distance(city1, city2):
    return np.sqrt((city1[0]-city2[0])**2 + (city1[1]-city2[1])**2)

n = len(city_coords)

distance_matrix = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        distance_matrix[i][j] = distance(city_coords[i], city_coords[j])

def initialize_tour(city_coords):
    initial_tour = []
    initial_tour.append(0)
    visited = set(initial_tour)
    for i in range(1,len(city_coords)):
        min_distance = float('inf')
        min_index = -1
        for j in range(len(city_coords)):
            distance_ij = distance(city_coords[j],city_coords[initial_tour[-1]])
            if distance_ij < min_distance and j not in visited:
                min_distance = distance_ij
                min_index = j
        initial_tour.append(min_index)
        visited.add(min_index)
    initial_tour.append(0)
    return initial_tour


def tour_length(tour,distance_matrix):
    length = 0
    for i in range(len(tour)):
        length += distance_matrix[tour[i-1]][tour[i]]
    return length

def tabu_check(current_tour, tabu_list):
    return tuple(current_tour) in tabu_list

def exchange_choose(current_tour, temperature, distance_matrix, tabu_list=deque(maxlen=10000), tabu_size=10000):
    n = len(current_tour)
    best_delta = float('inf')

    # Initialize best_tour to avoid UnboundLocalError
    best_tour = current_tour.copy()  # Assume current tour is initially the best

    # Wrap the tour
    for i in range(1, n - 2):
        for j in range(i + 1, n - 1):
            temp_tour = current_tour.copy()
            temp_tour[i], temp_tour[j] = temp_tour[j], temp_tour[i]

            if tabu_check(temp_tour, tabu_list):
                continue

            delta = (
                tour_length(temp_tour, distance_matrix) - tour_length(current_tour, distance_matrix)
            )

            if delta < best_delta:
                best_delta = delta
                best_tour = temp_tour

    # Reverse the tour
    for i in range(1, n - 2):
        for j in range(i + 1, n - 1):
            temp_tour = current_tour.copy()
            temp_tour[i:j + 1] = temp_tour[i:j + 1][::-1]

            if tabu_check(temp_tour, tabu_list):
                continue

            delta = (
                tour_length(temp_tour, distance_matrix) - tour_length(current_tour, distance_matrix)
            )

            if delta < best_delta:
                best_delta = delta
                best_tour = temp_tour

    # Insert operation
    for i in range(1, n - 1):
        for j in range(1, n - 2):
            if i != j:
                temp_tour = current_tour.copy()
                temp_tour.insert(j, temp_tour.pop(i))

                if tabu_check(temp_tour, tabu_list):
                    continue

                delta = (
                    tour_length(temp_tour, distance_matrix) - tour_length(current_tour, distance_matrix)
                )
                if delta < best_delta:
                    best_delta = delta
                    best_tour = temp_tour

    tabu_list.appendleft(tuple(best_tour))

    if len(tabu_list) > tabu_size:
        tabu_list.pop()

    if np.exp(-best_delta / temperature) > np.random.rand() or best_delta < 0:
        return best_tour

    return current_tour
    
def simulated_annealing(delta_T,T_max,T_min,n_iter,n,city_coords):
    current_tour = initialize_tour(city_coords)
    current_length = tour_length(current_tour,distance_matrix)
    
    best_tour = current_tour
    best_length = current_length
    
    Length = []
    T = T_max
    for i in range(n_iter):
        current_tour = exchange_choose(current_tour,T,distance_matrix)
        current_length = tour_length(current_tour,distance_matrix)
        if current_length < best_length:
            best_tour = current_tour
            best_length = current_length
        if T > T_min:
            T = T* delta_T
        else:
            break
        Length.append(best_length)
        if i%10 == 0:
            print("Iteration:",i,"Current length:",current_length,"Best length:",best_length,"Temperature:",T)
    return best_tour,best_length,Length

delta_T = 0.99
T_max = 100
T_min = 1
n_iter = 100000
best_tour,best_length,Length = simulated_annealing(delta_T,T_max,T_min,n_iter,n,city_coords)
print("Best tour length:",best_length)

tour_coords = city_coords[best_tour]
plt.plot(tour_coords[:,0],tour_coords[:,1],'o-')
plt.show()

plt.plot(Length)
plt.xlabel("Iteration")
plt.ylabel("Tour length")
plt.show()