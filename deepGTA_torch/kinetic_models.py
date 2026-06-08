import numpy as np
import itertools
from functools import lru_cache
from dataclasses import dataclass

@dataclass
class KineticModel:
    id: int
    num_species: int
    
    @property
    def binary_matrix(self):
        return generate_binary_matrix(self.id, self.num_species)

def loguniform(low: float = 0.0, high: float = 1.0, size=None):
    return np.exp(np.random.uniform(np.log(low), np.log(high), size))

def generate_binary_matrix(kinetic_id: int, num_species: int) -> np.ndarray:
    n_permutable = [0, 1, 3, 6, 10, 15][num_species-1]
    bool_list = [bool(kinetic_id & (1 << n)) for n in range(n_permutable)]
    matrix = np.zeros((num_species, num_species))
    for i in range(num_species):
        for j in range(i):
            if bool_list and len(bool_list) > 0:
                if bool_list.pop():
                    matrix[i, j] = 1
                
    for i in range(num_species):
        matrix[i, i] = -np.sum(matrix[:, i])
    return matrix

def generate_matrix(kinetic_id: int, num_species: int, t_min: float, t_max: float) -> np.ndarray:
    n_permutable = [0, 1, 3, 6, 10, 15][num_species-1]
    bool_list = [bool(kinetic_id & (1 << n)) for n in range(n_permutable)]
    matrix = np.zeros((num_species, num_species))
    for i in range(num_species):
        for j in range(i):
            if bool_list and len(bool_list) > 0:
                if bool_list.pop():
                    matrix[i, j] = 1 / loguniform(t_min, t_max)
                
    for i in range(num_species):
        matrix[i, i] = -np.sum(matrix[:, i])
    return matrix

@lru_cache(maxsize=16)
def enumerate_viable_models(num_species: int = 5):
    """
    Creates all different kinetic models for a given number of species and
    excludes them if:
    - isomorphic to an existing model
    - more than 2 species branching into one state
    - a state does not get populated after excitation but has a decay constant

    Default is num_species=5 which gives 103 viable models — matching the
    original published DeepSKAN model.
    """
    n_permutable = [0, 1, 3, 6, 10, 15][num_species-1]
    permutations = 2**n_permutable

    unique_ids = []

    for k_id in range(permutations):
        K = generate_binary_matrix(k_id, num_species)
        K_ = K.copy()
        for i in range(num_species):
            K_[i, i] = 0

        is_unique_matrix = True
        sum_lines = [np.sum(x) for x in K_]
        sum_rows  = [np.sum(x) for x in K_.transpose()]

        # Check for orphan states (not populated but have decay)
        for i in range(1, num_species):
            if sum_lines[i] == 0:
                if np.sum(sum_lines[i:-1]) > 0:
                    is_unique_matrix = False
                if sum_rows[i] > 0:
                    is_unique_matrix = False

        # Check for overbranching (more than 2 species feeding into one state)
        for x in sum_rows:
            if x > 2:
                is_unique_matrix = False

        if is_unique_matrix:
            unique_ids.append(k_id)

    # ── Remove isomorphic duplicates (original proven logic) ─────────────────
    unique_graphs = []
    name_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G'][:num_species]
    permutable_list = name_list[1:-1]
    permutations_list = list(itertools.permutations(permutable_list))
    permuted_name_list = [['A'] + list(p) + [name_list[-1]]
                          for p in permutations_list]

    viable_ids = []

    for k_id in unique_ids:
        is_unique = True
        K = generate_binary_matrix(k_id, num_species)
        K_ = K.copy()
        for i in range(num_species):
            K_[i, i] = 0

        graphs_for_one_model = []
        for p in permuted_name_list:
            graph_list = [[] for _ in range(num_species)]
            for i in range(num_species):
                for j in range(num_species):
                    if K_[i, j] > 0:
                        graph_list[i].append(p[j])
            graphs_for_one_model.append(graph_list)

        for graph in graphs_for_one_model:
            if graph in unique_graphs:
                is_unique = False

        if is_unique:
            for g in graphs_for_one_model:
                unique_graphs.append(g)
            viable_ids.append(k_id)

    return viable_ids
