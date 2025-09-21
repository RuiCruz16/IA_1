import numpy as np
import random
import csv
from utils import *

class TreeNode:
    def __init__(self, asset=None, weight=None, left=None, right=None):
        self.asset = asset
        self.weight = weight
        self.left = left
        self.right = right

    def __eq__(self, value):
        if value is None:
            return False
        return self.asset == value.asset and self.weight == value.weight and self.left == value.left and self.right == value.right

def get_random_portfolio(assets):
    nodes = [TreeNode(asset=asset, weight=0.0) for asset in assets]

    if len(nodes) == 1:
        return nodes[0]

    while len(nodes) > 1:
        left = nodes.pop(random.randint(0, len(nodes) - 1))
        right = nodes.pop(random.randint(0, len(nodes) - 1))

        parent_weight = random.uniform(0.1, 0.9)
        parent = TreeNode(asset=None, weight=parent_weight, left=left, right=right)

        nodes.append(parent)

    return nodes[0]


def compute_weights(node, current_weight=1.0, weights=None):
    if weights is None:
        weights = {}

    if node is None:
        return weights

    if node.asset:
        weights[node.asset] = weights.get(node.asset, 0) + current_weight
    else:
        left_weight = current_weight * node.weight
        right_weight = current_weight * (1.0 - node.weight)

        compute_weights(node.left, left_weight, weights)
        compute_weights(node.right, right_weight, weights)

    return weights

def get_best_solution(population, evaluation_function):
    best_solution = None
    best_sharpe_ratio = float('-inf')

    for portfolio_tree in population:
        portfolio_dict = compute_weights(portfolio_tree)

        sharpe_ratio = evaluation_function(portfolio_dict)

        if sharpe_ratio > best_sharpe_ratio:
            best_sharpe_ratio = sharpe_ratio
            best_solution = portfolio_tree

    return best_solution

def print_tree(node, indent=""):
    if node is None:
        return

    if node.asset:
        print(f"{indent}Asset: {node.asset}, Weight: {node.weight:.4f}")
    else:
        print(f"{indent}Node (Weight: {node.weight:.4f}):")
        print_tree(node.left, indent + "  L-> ")
        print_tree(node.right, indent + "  R-> ")

def get_depth(node):
    if node is None:
        return 0
    return 1 + max(get_depth(node.left), get_depth(node.right))

def clone_tree(node):
    if node is None:
        return None
    new_node = TreeNode(asset=node.asset, weight=node.weight)
    new_node.left = clone_tree(node.left)
    new_node.right = clone_tree(node.right)
    return new_node

def is_subtree(root, target):
    if root is None:
        return False

    if root == target:
        return True
    return is_subtree(root.left, target) or is_subtree(root.right, target)

def swap_subtrees(tree1, subtree1, tree2, subtree2):
    if not is_subtree(tree1, subtree1):
        raise ValueError("subtree1 is not present in tree1.")
    if not is_subtree(tree2, subtree2):
        raise ValueError("subtree2 is not present in tree2.")

    temp = clone_tree(subtree1)
    replaced1 = replace_subtree(tree1, subtree1, clone_tree(subtree2))
    replaced2 = replace_subtree(tree2, subtree2, temp)

    if not (replaced1 and replaced2):
        raise RuntimeError("Warning: One or both subtrees were not replaced successfully.")


def replace_subtree(root, target, new_subtree):
    if root is None:
        return False
    if root.left == target:
        root.left = new_subtree
        return True
    if root.right == target:
        root.right = new_subtree
        return True
    return replace_subtree(root.left, target, new_subtree) or replace_subtree(root.right, target, new_subtree)


def get_subtrees_at_depth(node, target_depth, current_depth=1):
    if node is None:
        return []
    if current_depth == target_depth:
        return [node]
    return get_subtrees_at_depth(node.left, target_depth, current_depth + 1) + \
           get_subtrees_at_depth(node.right, target_depth, current_depth + 1)

def best_subtree_at_depth(node, target_depth):
    subtrees = get_subtrees_at_depth(node, target_depth)
    if not subtrees:
        return None
    best = max(subtrees, key=lambda s: calculate_portfolio_sharpe_ratio(compute_weights(s)))
    return best

def worst_subtree_at_depth(node, target_depth):
    subtrees = get_subtrees_at_depth(node, target_depth)
    if not subtrees:
        return None
    worst = min(subtrees, key=lambda s: calculate_portfolio_sharpe_ratio(compute_weights(s)))
    return worst


def best_worst_subtree_crossover(parent1, parent2):
    depth1 = get_depth(parent1)
    depth2 = get_depth(parent2)

    if min(depth1, depth2) < 2:
        return clone_tree(parent1), clone_tree(parent2)

    d = random.randint(2, min(depth1, depth2))

    best_sub = best_subtree_at_depth(parent1, d)
    worst_sub = worst_subtree_at_depth(parent2, d)

    child1 = clone_tree(parent1)
    child2 = clone_tree(parent2)

    swap_subtrees(child1, best_sub, child2, worst_sub)

    return child1, child2

def tree_mutate(node, mutation_rate=0.2):
    if node is None:
        return node

    if node.asset:
        if random.random() < mutation_rate:
            mutation_value = np.random.uniform(-0.1, 0.1)
            node.weight = max(node.weight + mutation_value, 0.01)
    else:
        if random.random() < mutation_rate:
            if node.left and node.right:
                node.left, node.right = node.right, node.left
                left_weight = np.random.uniform(0.1, 0.9)
                right_weight = 1.0 - left_weight
                node.left.weight = left_weight
                node.right.weight = right_weight

    if node.left:
        tree_mutate(node.left, mutation_rate)
    if node.right:
        tree_mutate(node.right, mutation_rate)

    return node

def local_search_node(node, evaluation_function, meme_speed=0.1, meme_accel=0.333, meme_tresh=0.003):
    if node.asset is not None or not (node.left and node.right):
        return node.weight

    current_weight = node.weight
    current_fitness = evaluation_function(compute_weights(node))
    speed = meme_speed

    while abs(speed) > meme_tresh and 0 < current_weight < 1:
        new_weight = current_weight + speed
        new_weight = max(0, min(1, new_weight))
        original_weight = node.weight
        node.weight = new_weight
        new_fitness = evaluation_function(compute_weights(node))

        if new_fitness < current_fitness:
            speed = speed * meme_accel * -1
        else:
            current_weight = new_weight
            current_fitness = new_fitness

        node.weight = current_weight

    return current_weight

def recursive_local_search(node, evaluation_function, meme_speed=0.1, meme_accel=0.333, meme_tresh=0.003):
    if node is None:
        return

    if node.left:
        recursive_local_search(node.left, evaluation_function, meme_speed, meme_accel, meme_tresh)
    if node.right:
        recursive_local_search(node.right, evaluation_function, meme_speed, meme_accel, meme_tresh)

    if node.asset is None:
        node.weight = local_search_node(node, evaluation_function, meme_speed, meme_accel, meme_tresh)

def mutate_random_subtree(tree, assets, mutation_rate=0.2):
    if random.random() >= mutation_rate:
        return tree

    max_depth = get_depth(tree)
    target_depth = random.randint(1, max_depth)

    if target_depth == 1:
        return get_random_portfolio(assets)

    current = tree
    parent = None
    direction = None
    current_depth = 1

    while current is not None and current_depth < target_depth:
        parent = current

        if current.left and current.right:
            if random.random() < 0.5:
                direction = 'left'
                current = current.left
            else:
                direction = 'right'
                current = current.right
        elif current.left:
            direction = 'left'
            current = current.left
        elif current.right:
            direction = 'right'
            current = current.right
        else:
            break
        current_depth += 1

    new_subtree = get_random_portfolio(assets)

    if parent is None:
        tree = new_subtree
    else:
        if direction == 'left':
            parent.left = new_subtree
        elif direction == 'right':
            parent.right = new_subtree
        else:
            if parent.left == current:
                parent.left = new_subtree
            elif parent.right == current:
                parent.right = new_subtree

    return tree

'''
    Implements a genetic tree algorithm for portfolio optimization.
    The algorithm:
      1. Initializes a population of random tree-based portfolios.
      2. Evaluates each individual using the evaluation_function on computed weights.
      3. Preserves the top 30% of individuals as parents.
      4. Fills the remaining population with offspring generated using crossover (with probability crossover_rate)
         or mutation.
      5. Applies mutation to all offspring and a local search to a random subset.
      6. Logs the best solution for each generation into a CSV file.
      7. Returns the best portfolio (after computing its weights) at the end of the run.
'''
def genetic_tree_algorithm(population_size, generations, mutation_rate, crossover_rate, evaluation_function, local_search_count=5):
    assets = list(stocksRecords.keys())
    population = [get_random_portfolio(assets) for _ in range(population_size)]

    best_solution = get_best_solution(population, evaluation_function)
    print("Initial best fitness:", evaluation_function(compute_weights(best_solution)))

    log_file = 'genetic_tree_algorithm_log.csv'
    with open(log_file, mode='w+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Generation', 'BestScore', 'Portfolio'])
        writer.writerow([0, evaluation_function(compute_weights(best_solution)), compute_weights(best_solution)])

    for generation in range(generations):
        # Evaluate population fitness
        evaluations = [evaluation_function(compute_weights(portfolio)) for portfolio in population]
        sorted_population = [x for _, x in sorted(zip(evaluations, population), key=lambda x: x[0], reverse=True)]

        # Preserve top 30% of parents
        num_parents_to_preserve = int(0.3 * population_size)
        parents_to_preserve = sorted_population[:num_parents_to_preserve]

        # Generate offspring to fill the remaining 70% of the population
        offspring = []
        while len(offspring) < 0.7 * population_size:
            if random.random() < crossover_rate:
                # Perform crossover between two parents
                parent1, parent2 = random.sample(parents_to_preserve, 2)
                child1, child2 = best_worst_subtree_crossover(parent1, parent2)
                offspring.append(child1)
                if len(offspring) < 0.7 * population_size:
                    offspring.append(child2)
            else:
                # Otherwise, perform mutation or direct reproduction
                parent = random.choice(parents_to_preserve)
                child = clone_tree(parent)
                child = mutate_random_subtree(child, assets, mutation_rate)
                offspring.append(child)

        # Apply mutation to all offspring
        for i in range(len(offspring)):
            offspring[i] = mutate_random_subtree(offspring[i], assets, mutation_rate)

        # Apply local search to a random subset of 'n' individuals from the offspring
        local_search_population = random.sample(offspring, local_search_count)  
        for individual in local_search_population:
            recursive_local_search(individual, evaluation_function)

        # Combine parents and offspring, then select the best individuals
        candidates = parents_to_preserve + offspring
        population = sorted(candidates, key=lambda p: evaluation_function(compute_weights(p)), reverse=True)[:population_size]
        best_solution = population[0]

        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([generation + 1, evaluation_function(compute_weights(best_solution)), compute_weights(best_solution)])

        print(f"Generation {generation + 1}/{generations} - Best Fitness: {evaluation_function(compute_weights(best_solution))}")

    return compute_weights(best_solution)
