import numpy as np
import matplotlib.pyplot as plt
import math

bounds = np.array([-5.12, 5.12]) 

def fitness(x):
    return np.sum(x**2 - np.cos(2*np.pi*x) + 10)

def initialize_particles(num_particles, dim, bounds):
    lower_bounds, upper_bounds = bounds
    
    particles = np.random.uniform(lower_bounds, upper_bounds, size=(num_particles, dim))

    return particles

def update_fitness(particles, fitness_values):
    for i in range(len(particles)):
        fitness_values[i] = fitness(particles[i])
    return fitness_values

def update_particles(particles,particles_volocity,bounds):
    particles += particles_volocity
    for i in range(len(particles)):
        for j in range(len(particles[i])):
            if particles[i][j] < bounds[0]:
                particles[i][j] = bounds[0]
            elif particles[i][j] > bounds[1]:
                particles[i][j] = bounds[1]
                particles_volocity[i][j] = 0
    return particles, particles_volocity

def update_particles_volocity(particles, particles_volocity, best_particle, best_particle_matrix, c1, c2, bounds):
    for i in range(len(particles)):
        particles_volocity[i] += c1 * np.random.rand() * (best_particle - particles[i]) + c2 * np.random.rand() * (best_particle_matrix[i] - particles[i])
        for j in range(len(particles[i])):
            if particles[i][j] < bounds[0] or particles[i][j] > bounds[1]:
                particles_volocity[i][j] = 0
    return particles_volocity

def optimize(num_particles, dim, bounds, max_iter, c1, c2):
    
    particles = initialize_particles(num_particles, dim, bounds)
    particles_volocity = initialize_particles(num_particles, dim, bounds)
    fitness_values = np.zeros(num_particles)


    fitness_values = update_fitness(particles, fitness_values)

    best_particle_matrix = np.copy(particles)
    best_fitness_value = np.min(fitness_values)
    best_particle = particles[np.argmin(fitness_values)]

    for iteration in range(max_iter):
               
        particles, particles_volocity = update_particles(particles, particles_volocity, bounds)
        fitness_values = update_fitness(particles, fitness_values)

        if np.min(fitness_values) < best_fitness_value:
            best_fitness_value = np.min(fitness_values)
            best_particle = particles[np.argmin(fitness_values)]
            best_particle_matrix = np.copy(particles)

        particles_volocity = update_particles_volocity(particles, particles_volocity, best_particle, best_particle_matrix, c1, c2, bounds)

    return best_particle, best_fitness_value

best_particle, best_fitness_value = optimize(10, 2, bounds, 10000, 2, 2)
print("Best Particle:", best_particle)
print("Best Fitness Value:", best_fitness_value)