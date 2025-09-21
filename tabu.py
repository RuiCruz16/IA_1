import plotly.graph_objects as go
import numpy as np
import random
import csv
import math
from loader import stocksRecords
from utils import *

NEIGHBORHOOD_SIZE = 20
MAX_ITERATIONS = 1000
MAX_NEIGHBORS = 20
TABU_TENURE = 10

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

# Implements an improved version of the Tabu Search algorithm.
# This version includes additional logic to handle repeated solutions and explore new random portfolios when necessary.
# The best solution and its score are logged to a CSV file during the search process.
def tabu_search_improved(max_portfolios_iterations=MAX_ITERATIONS, max_tabu_iterations=MAX_ITERATIONS, num_neighbors=MAX_NEIGHBORS, tabu_tenure=TABU_TENURE, evaluation_function=calculate_portfolio_sharpe_ratio):

    new_solution = get_random_portfolio()
    new_score = evaluation_function(new_solution)

    best_solution = new_solution
    best_score = new_score
    tabu_list = []
    portfolio_list = []

    iteration = 1

    print(f"Initial Portfolio: {best_solution}")
    print(f"Initial Portfolio Score: {best_score}")

    # Prepare CSV file
    log_file = 'tabu_search_improved_log.csv'
    with open(log_file, mode='w+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Iteration', 'BestScore', 'Portfolio'])
        writer.writerow([0, best_score, best_solution])        

    for i in range(max_portfolios_iterations):
        print(f"Portfolio Iteration {i + 1}/{max_portfolios_iterations} - Best Score: {best_score}")
        repeated = 0

        while repeated < max_tabu_iterations:

            neighbors = get_portfolio_neighbors(new_solution, num_neighbors)

            best_neighbor = None
            best_neighbor_score = -np.inf

            # Find the best neighbor
            for neighbor in neighbors:
                if neighbor not in tabu_list:
                    score = evaluation_function(neighbor)
                    if score > best_neighbor_score:
                        best_neighbor = neighbor
                        best_neighbor_score = score

            # Update the best solution
            if best_neighbor_score > new_score:
                if math.floor(new_score) == math.floor(best_neighbor_score):
                    repeated += 1
                else:
                    repeated = 0
                new_solution = best_neighbor
                new_score = best_neighbor_score
                print(f"New Best Neighbor: {new_score}")
                with open(log_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([iteration, new_score, new_solution])
                
                iteration += 1
                
            else:
                repeated += 1

            tabu_list.append(new_solution)
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)

            

            print(f"Tabu Iteration {repeated}/{max_tabu_iterations} - Best Score: {math.floor(new_score)}")

        if (new_score > best_score):
            best_solution = new_solution
            best_score = new_score
        
        # Log the best score
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([iteration, best_score, best_solution])
        
        iteration += 1
        
        portfolio_list.append(new_solution)
        if len(portfolio_list) > tabu_tenure:
            portfolio_list.pop(0)

        while new_solution in portfolio_list:
            new_solution = get_random_portfolio()
            new_score = evaluation_function(new_solution)


    print(f"Best Portfolio: {best_solution}")
    print(f"Best Portfolio Score: {best_score}")
    return best_solution


# Implements the Tabu Search algorithm to optimize a portfolio.
# The algorithm iteratively explores the neighborhood of the current solution to find the best portfolio.
# A tabu list is used to prevent revisiting recently explored solutions.
# The best solution and its score are logged to a CSV file during the search process.
def tabu_search(max_tabu_iterations=MAX_ITERATIONS, num_neighbors=MAX_NEIGHBORS, tabu_tenure=TABU_TENURE, evaluation_function=calculate_portfolio_sharpe_ratio):
    best_solution = get_random_portfolio()
    best_score = calculate_portfolio_sharpe_ratio(best_solution)

    tabu_list = []


    print(f"Initial Portfolio: {best_solution}")
    print(f"Initial Portfolio Score: {best_score}")

    log_file = 'tabu_search_log.csv'
    with open(log_file, mode='w+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Iteration', 'BestScore', 'Portfolio'])
        writer.writerow([0, best_score, best_solution])
    
    for iteration in range(max_tabu_iterations):
        neighbors = get_portfolio_neighbors(best_solution, num_neighbors)
        
        best_neighbor = None
        best_neighbor_score = -np.inf

        for neighbor in neighbors:
            score = evaluation_function(neighbor)
            if score > best_neighbor_score:
                best_neighbor_score = score
                best_neighbor = neighbor

        if best_neighbor_score > best_score:
            best_score = best_neighbor_score
            best_solution = best_neighbor
        
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([iteration + 1, best_score, best_solution])
        
        print(f"Iteration {iteration + 1}/{max_tabu_iterations} - Best Score: {best_score}")
        tabu_list.append(best_solution)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)
    
    return best_solution
