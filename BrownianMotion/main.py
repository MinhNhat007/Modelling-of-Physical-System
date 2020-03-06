import matplotlib.pyplot as plt
import numpy as np
from numpy import random as rd


def get_random_number_by_normal_distribution():
    mu = 0
    sigma = 1
    return rd.normal(mu, sigma)


def draw_trajectory(number_particle, number_step):
    x_coordinate = np.zeros((number_step, number_particle))
    y_coordinate = np.zeros((number_step, number_particle))

    for step in range(1, number_step):
        for particle in range(0, number_particle):
            x_coordinate[step, particle] = x_coordinate[step - 1, particle] + get_random_number_by_normal_distribution()
            y_coordinate[step, particle] = y_coordinate[step - 1, particle] + get_random_number_by_normal_distribution()

    for particle in range(0, number_particle):
        plt.plot(x_coordinate[:, particle], y_coordinate[:, particle], linewidth=0.5)
    plt.title('Trajectory of {0} particle(s) within {1} steps'.format(number_particle, number_step))
    plt.xlabel('x coordinate')
    plt.ylabel('y coordinate')
    plt.show()


def draw_mean_time_displacement(number_dimension, number_particle, number_step):
    mean_squares = np.zeros(number_step)
    coordinate = np.zeros((number_dimension, number_step, number_particle))

    for step in range(1, number_step):
        for particle in range(0, number_particle):
            for dimension in range(0, number_dimension):
                coordinate[dimension, step, particle] = coordinate[dimension, step - 1, particle] + get_random_number_by_normal_distribution()

        sum_by_step = np.zeros(number_particle)
        for dimension in range(0, number_dimension):
            sum_by_step = sum_by_step + np.power(coordinate[dimension, step], 2)

        mean_squares[step] = np.mean(sum_by_step)

    plt.plot(range(0, number_step), mean_squares)
    plt.title('Mean square of displacement after {0} step for {1} particle(s) in {2} dimensional(s)'.format(number_step, number_particle, number_dimension))
    plt.xlabel('timestep')
    plt.ylabel('mean square displacement')
    plt.show()


def draw_evolution_of_particles_density(number_particle, number_step):
    x_coordinate = np.zeros(number_particle)
    y_coordinate = np.zeros(number_particle)
    for step in range(1, number_step):
        for particle in range(0, number_particle):
            x_coordinate[particle] = x_coordinate[particle] + get_random_number_by_normal_distribution()
            y_coordinate[particle] = y_coordinate[particle] + get_random_number_by_normal_distribution()

    fig, (hist, hist2d) = plt.subplots(1, 2)
    hist.hist(x_coordinate, bins=20)
    hist.set_title('Density over time for {0} particles in {1} steps - 1D'.format(number_particle, number_step))
    hist.set_xlabel('x coordinate')
    hist.set_ylabel('particles')
    h = hist2d.hist2d(x_coordinate, y_coordinate, bins=20)
    hist2d.set_title('Density over time for {0} particles in {1} steps (2D)'.format(number_particle, number_step))
    hist2d.set_xlabel("x coordinate")
    hist2d.set_ylabel("y coordinate")
    colorbar = plt.colorbar(h[3], ax=hist2d)
    colorbar.set_label("particles")
    plt.show()


if __name__ == '__main__':
 draw_trajectory(number_particle=1, number_step=5000)
 draw_trajectory(number_particle=10, number_step=5000)
 draw_trajectory(number_particle=100, number_step=5000)
 draw_trajectory(number_particle=1000, number_step=5000)
 draw_mean_time_displacement(number_dimension=2, number_particle=1, number_step=5000)
 draw_mean_time_displacement(number_dimension=2, number_particle=10, number_step=5000)
 draw_mean_time_displacement(number_dimension=2, number_particle=100, number_step=5000)
 draw_mean_time_displacement(number_dimension=2, number_particle=1000, number_step=5000)
 draw_mean_time_displacement(number_dimension=1, number_particle=1000, number_step=5000)
 draw_mean_time_displacement(number_dimension=3, number_particle=1000, number_step=5000)
 draw_evolution_of_particles_density(number_particle=1000, number_step=5000)
