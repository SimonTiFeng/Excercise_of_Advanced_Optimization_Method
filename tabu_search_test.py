import numpy as np
import matplotlib.pyplot as plt

city_coords = np.array([
    [48, 21], [52, 26], [55, 50], [50, 50], [41, 46], [51, 42], [55, 45],
    [38, 33], [33, 34], [45, 35], [40, 37], [50, 30], [55, 34], [54, 38],
    [26, 13], [15, 5], [21, 48], [29, 39], [33, 44], [15, 19], [16, 19],
    [12, 17], [50, 40], [22, 53], [21, 36], [20, 30], [26, 29], [40, 20],
    [36, 26], [62, 48], [67, 41], [62, 35], [65, 27], [62, 24], [55, 20],
    [35, 51], [30, 50], [45, 42], [21, 45], [36, 6], [6, 25], [11, 28],
    [26, 59], [30, 60], [22, 22], [27, 24], [30, 20], [35, 16], [54, 10],
    [50, 15], [44, 13], [35, 60], [40, 60], [40, 66], [31, 76], [47, 66],
    [50, 70], [57, 72], [55, 65], [2, 38], [7, 43], [9, 56], [15, 56],
    [17, 64], [55, 57], [62, 57], [70, 64], [64, 4], [59, 5], [50, 4],
    [60, 15], [66, 14], [66, 8], [43, 26]
])

def distance(city1, city2):
    return np.sqrt((city1[0]-city2[0])**2 + (city1[1]-city2[1])**2)

n = len(city_coords)
distance_matrix = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        distance_matrix[i][j] = distance(city_coords[i], city_coords[j])

def tour_length(tour):
    length = 0
    for i in range(len(tour)):
        length += distance(city_coords[tour[i-1]], city_coords[tour[i]])
    return length

def exchange_choose(tabu_list, current_tour,distance_matrix,tabu_length):
    Candidate_list = []
    Candidate_distance = []
    for i in range(len(current_tour)):
        for j in range(i+1,len(current_tour)):
            if (i,j) not in tabu_list and (j,i) not in tabu_list:
                Candidate_list.append((i,j))
                temporary_tour = current_tour.copy()
                temporary_tour[i],temporary_tour[j] = temporary_tour[j],temporary_tour[i]
                new_tour_length = tour_length(temporary_tour)
                Candidate_distance.append(new_tour_length)
    
    best_index = np.argmin(Candidate_distance) 
    (i, j) = Candidate_list[best_index]
    if len(tabu_list) == tabu_length:
        tabu_list.pop(0) 
    tabu_list.append((i,j))  
    current_tour[i], current_tour[j] = current_tour[j], current_tour[i] 
    return current_tour, tabu_list

tabu_length = 60
tabu_list = []
current_tour = np.random.permutation(n)
iteration = 1000

for i in range(iteration):
    current_tour,tabu_list = exchange_choose(tabu_list,current_tour,distance_matrix,tabu_length)
    if i % 10 == 0:
        print("Iteration:", i, "Tour Length:", tour_length(current_tour))

print("Final Tour Length:", tour_length(current_tour))
plt.figure(figsize=(10, 6))
for i in range(len(current_tour)):
        start = city_coords[current_tour[i]]
        end = city_coords[current_tour[(i + 1) % len(current_tour)]]  
        plt.plot([start[0], end[0]], [start[1], end[1]], 'bo-')  

plt.scatter(city_coords[:, 0], city_coords[:, 1], color='red')  
plt.title("Current Tour Route")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid()
plt.show()
