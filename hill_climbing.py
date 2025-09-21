import numpy as np
import random
import csv
from utils import *

# Generates a random portfolio by selecting a random number of stocks from the available list.
# Each selected stock is assigned a random weight, and the weights are normalized so that their total sum equals 1.
def get_random_portfolio():
    total_stocks = list(stocksRecords.keys())
    num_stocks = np.random.randint(1, len(total_stocks) + 1)
    selected_stocks = np.random.choice(total_stocks, num_stocks, replace=False)
    weights = np.random.rand(num_stocks)
    weights /= np.sum(weights)
    portfolio = {selected_stocks[i]: weights[i] for i in range(num_stocks)}

    return portfolio


# Generates a list of neighboring portfolios by slightly perturbing the weights of a random stock in the current portfolio.
# The perturbation is controlled by the `perturbation` parameter, and the weights are normalized after the adjustment.
# A minimum weight of 0.01 is enforced to prevent stocks from being completely removed from the portfolio.
def get_portfolio_neighbors(current_portfolio, num_neighbors, perturbation=0.1):
    neighbors = []

    for _ in range(num_neighbors):
        new_portfolio = current_portfolio.copy()

        stock_to_mutate = random.choice(list(current_portfolio.keys()))

        mutation = np.random.uniform(-perturbation, perturbation)
        new_weight = new_portfolio[stock_to_mutate] + mutation

        new_weight = np.clip(new_weight, 0.01, 1.0)

        new_portfolio[stock_to_mutate] = new_weight

        total_weight = sum(new_portfolio.values())
        adjustment_factor = 1.0 / total_weight
        new_portfolio = {k: v * adjustment_factor for k, v in new_portfolio.items()}

        neighbors.append(new_portfolio)

    return neighbors


# Implements the hill climbing algorithm to optimize a portfolio.
# The algorithm starts with a random portfolio and iteratively explores its neighbors to find a better solution.
# For each iteration:
# 1. A set of neighboring portfolios is generated.
# 2. The best neighbor is selected based on the evaluation function.
# 3. If the best neighbor improves the score, it becomes the current solution.
# The algorithm logs the best score for each iteration to a CSV file and prints the progress.
# At the end, the best portfolio found during all iterations is returned.
def hill_climbing(max_iterations, num_neighbors, evaluation_function):
    current_solution = get_random_portfolio()
    best_solution = current_solution
    best_score = evaluation_function(best_solution)

    # Prepare CSV file
    log_file = 'hill_climbing_log.csv'
    with open(log_file, mode='w+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Iteration', 'BestScore'])  
        writer.writerow([0, best_score])

    for iteration in range(max_iterations):
        neighbors = get_portfolio_neighbors(current_solution, num_neighbors)

        best_neighbor = None
        best_neighbor_score = -np.inf
        for neighbor in neighbors:
            score = evaluation_function(neighbor)
            if score > best_neighbor_score:
                best_neighbor_score = score
                best_neighbor = neighbor

        if best_neighbor_score > best_score:
            current_solution = best_neighbor
            best_score = best_neighbor_score
            best_solution = best_neighbor

        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([iteration + 1, best_score])

        print(f"Iteration {iteration+1} - Best Sharpe Ratio: {best_score}")

    return best_solution
