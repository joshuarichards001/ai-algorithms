from gplearn.genetic import SymbolicRegressor
import numpy as np


def file_parser(file_name):
    with open(file_name) as f:
        lines = [line.rstrip() for line in f]
    for i, line in enumerate(lines[1:]):
        instance_value_list = line.split()
        x_data[i] = float(instance_value_list[0])
        y_data[i] = float(instance_value_list[1])


if __name__ == '__main__':
    x_data = np.empty(20).reshape(-1, 1)
    y_data = np.empty(20).reshape(-1, 1).ravel()
    file_parser("regression")

    est_gp = SymbolicRegressor(population_size=5000,
                               generations=20,
                               stopping_criteria=0.008,
                               p_crossover=0.7,
                               p_subtree_mutation=0.1,
                               p_hoist_mutation=0.05,
                               p_point_mutation=0.1,
                               max_samples=0.9,
                               verbose=1,
                               parsimony_coefficient=0.01,
                               random_state=0)
    est_gp.fit(x_data, y_data)
    print(est_gp._program)
