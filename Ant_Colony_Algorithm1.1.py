import numpy as np
import matplotlib.pyplot as plt

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
    return np.sqrt(np.sum((city1 - city2) ** 2))

def tour_length(tour, distance_matrix):
    length = 0
    for i in range(len(tour) - 1):
        length += distance_matrix[tour[i]][tour[i + 1]]
    return length

n = len(city_coords)
distance_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        dist = distance(city_coords[i], city_coords[j])
        distance_matrix[i][j] = dist
        distance_matrix[j][i] = dist

def way_chooser(current_tour, distance_matrix, alpha, beta, information_matrix):
    unvisited = [i for i in range(len(distance_matrix)) if i not in current_tour]
    
    probabilities = []
    for city in unvisited:
        pheromone = information_matrix[current_tour[-1]][city] ** alpha
        visibility = (1 / distance_matrix[current_tour[-1]][city]) ** beta
        probabilities.append(pheromone * visibility)

    probabilities = np.array(probabilities) / np.sum(probabilities)
    next_city = np.random.choice(unvisited, p=probabilities)
    current_tour.append(next_city)
    return current_tour

def update_pheromone(information_matrix, all_tours, distance_matrix, evaporation_rate, Q):
    information_matrix *= (1 - evaporation_rate)  
    for tour in all_tours:
        length = tour_length(tour, distance_matrix)
        for i in range(len(tour) - 1):
            i1, i2 = tour[i], tour[i + 1]
            information_matrix[i1][i2] += Q / length
            information_matrix[i2][i1] += Q / length  
    return information_matrix

def construct_solutions(distance_matrix, alpha, beta, num_ants, information_matrix):
    all_tours = []
    for _ in range(num_ants):
        start_city = np.random.randint(0, len(distance_matrix))
        current_tour = [start_city]
        
        while len(current_tour) < len(distance_matrix):
            current_tour = way_chooser(current_tour, distance_matrix, alpha, beta, information_matrix)

        current_tour.append(current_tour[0]) 
        all_tours.append(current_tour)
    
    return all_tours

def ant_colony_algorithm(distance_matrix, alpha=1, beta=2, num_ants=10, num_iterations=100, evaporation_rate=0.1, Q=1):
    n = len(distance_matrix)
    information_matrix = np.ones((n, n))  

    best_tour = None
    best_length = float('inf')

    for t in range(num_iterations):
        all_tours = construct_solutions(distance_matrix, alpha, beta, num_ants, information_matrix)
        
        for tour in all_tours:
            length = tour_length(tour, distance_matrix)
            if length < best_length:
                best_tour = tour
                best_length = length
        
        if t % 10 == 0:
            print(t,best_length)

        information_matrix = update_pheromone(information_matrix, all_tours, distance_matrix, evaporation_rate, Q)


    return best_tour, best_length

def plot_tour(tour, city_coords):
    plt.figure(figsize=(10, 5))
    tour_coords = city_coords[tour]
    plt.plot(tour_coords[:, 0], tour_coords[:, 1], 'o-', markersize=8, color='blue')
    for i, (x, y) in enumerate(city_coords):
        plt.text(x, y, str(i), fontsize=12, ha='right')
    plt.show()

best_tour, best_length = ant_colony_algorithm(distance_matrix)
print("最优路径：", best_tour)
print("最优路径长度：", best_length)

plot_tour(np.array(best_tour), city_coords)
