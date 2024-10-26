import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import pygad

iris = load_iris()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
X_selected = X[['sepal length (cm)', 'sepal width (cm)']]
X_selected = StandardScaler().fit_transform(X_selected)

def dist_calc(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return distances

def find_closest_centroid(X, centroids):
    distances = dist_calc(X, centroids)
    return np.argmin(distances, axis=1)

def fitness_func(ga_instance, solution, solution_idx, X):
    centroids = solution.reshape((-1, 2))
    index = find_closest_centroid(X, centroids)
    sse = np.sum(np.min(dist_calc(X, centroids), axis=1))
    return -sse

def simulated_annealing_acceptance(old_fitness, new_fitness, temperature):
    if new_fitness > old_fitness:
        return True
    else:
        prob = np.exp((new_fitness - old_fitness) / temperature)
        return np.random.rand() < prob

num_clusters = 5
num_features = X_selected.shape[1]
num_genes = num_clusters * num_features
sol_per_pop = 10
num_gen = 500
initial_temperature = 100.0
cooling_rate = 0.95

ga_instance = pygad.GA(
    num_generations=num_gen,
    num_parents_mating=5,
    fitness_func=lambda ga_instance, solution, solution_idx: fitness_func(
        ga_instance, solution, solution_idx, X_selected),
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    init_range_low=np.min(X_selected),
    init_range_high=np.max(X_selected),
    parent_selection_type="sss",
    keep_parents=1
)

temperature = initial_temperature
for generation in range(num_gen):
    ga_instance.run()
    best_solution, best_fitness, _ = ga_instance.best_solution()
    new_solution, new_fitness, _ = ga_instance.best_solution()
    if generation % 100 == 0:
        print("Generation:", generation, "Best Fitness:", best_fitness)
    temperature *= cooling_rate

centroids = best_solution.reshape(num_clusters, num_features)
cluster_labels = find_closest_centroid(X_selected, centroids)

plt.scatter(X_selected[:, 0], X_selected[:, 1], c=cluster_labels, cmap='viridis', marker='o')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=100)
plt.title('GA with Simulated Annealing on Iris Data')
plt.xlabel('Sepal Length (standardized)')
plt.ylabel('Sepal Width (standardized)')
plt.show()
