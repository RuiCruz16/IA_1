from simple_genetic import *
from tree_genetic import *
from hill_climbing import *
from tabu import *
from loader import *
from simulated_annealing import *
from utils import *
import numpy as np
import threading
import tkinter as tk
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib import style


style.use("ggplot")

f_simulated_annealing = Figure(figsize=(10, 5), dpi=100)
a_simulated_annealing = f_simulated_annealing.add_subplot(111)

f_genetic = Figure(figsize=(10, 5), dpi=100)
a_genetic = f_genetic.add_subplot(111)

f_genetic_tree = Figure(figsize=(10, 5), dpi=100)
a_genetic_tree = f_genetic_tree.add_subplot(111)

f_tabu_improved = Figure(figsize=(10, 5), dpi=100)
a_tabu_improved = f_tabu_improved.add_subplot(111)

f_tabu = Figure(figsize=(10, 5), dpi=100)
a_tabu = f_tabu.add_subplot(111)

f_hill = Figure(figsize=(10, 5), dpi=100)
a_hill = f_hill.add_subplot(111)

def animate_simulated_annealing(i):
    iteration_data = []
    score_data = []
    temperature_data = []

    if os.path.exists('simulated_annealing_log.csv'):
        with open('simulated_annealing_log.csv', mode='r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                iteration_data.append(int(row['Iteration']))
                score_data.append(float(row['BestScore']))
                temperature_data.append(float(row['Temperature']))
    
    a_simulated_annealing.clear()
    if iteration_data:
        a_simulated_annealing.plot(iteration_data, score_data, label='Best Score')
        a_simulated_annealing.plot(iteration_data, temperature_data, label='Temperature', linestyle='--')
    else:
        a_simulated_annealing.text(0.5, 0.5, "Waiting for data...", 
                                   horizontalalignment='center',
                                   verticalalignment='center',
                                   transform=a_simulated_annealing.transAxes)
    
    a_simulated_annealing.set_title("Simulated Annealing Progress")
    a_simulated_annealing.set_xlabel("Iterations")
    a_simulated_annealing.set_ylabel("Best Score (Sharpe Ratio)")
    a_simulated_annealing.legend()

def animate_genetic(i):
    iteration_data = []
    score_data = []
    average_data = []

    if os.path.exists('genetic_algorithm_log.csv'):
        with open('genetic_algorithm_log.csv', mode='r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                iteration_data.append(int(row['Generation']))
                score_data.append(float(row['BestScore']))
                average_data.append(float(row['AverageScore']))
    
    a_genetic.clear()
    if iteration_data:
        a_genetic.plot(iteration_data, score_data, label='Best Score')
        a_genetic.plot(iteration_data, average_data, label='Average Score', linestyle='--')
    else:
        a_genetic.text(0.5, 0.5, "Waiting for data...", 
                       horizontalalignment='center',
                       verticalalignment='center',
                       transform=a_genetic.transAxes)
    
    a_genetic.set_title("Genetic Algorithm Progress")
    a_genetic.set_xlabel("Generations")
    a_genetic.set_ylabel("Best Score (Sharpe Ratio)")
    a_genetic.legend()

def animate_genetic_tree(i):
    iteration_data = []
    score_data = []

    if os.path.exists('genetic_tree_algorithm_log.csv'):
        with open('genetic_tree_algorithm_log.csv', mode='r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                iteration_data.append(int(row['Generation']))
                score_data.append(float(row['BestScore']))
    
    a_genetic_tree.clear()
    if iteration_data:
        a_genetic_tree.plot(iteration_data, score_data)
    else:
        a_genetic_tree.text(0.5, 0.5, "Waiting for data...", 
                            horizontalalignment='center',
                            verticalalignment='center',
                            transform=a_genetic_tree.transAxes)
    
    a_genetic_tree.set_title("Genetic Tree Algorithm Progress")
    a_genetic_tree.set_xlabel("Generations")
    a_genetic_tree.set_ylabel("Best Score (Sharpe Ratio)")

def animate_tabu_improved(i):
    iteration_data = []
    score_data = []

    if os.path.exists('tabu_search_improved_log.csv'):
        with open('tabu_search_improved_log.csv', mode='r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                iteration_data.append(int(row['Iteration']))
                score_data.append(float(row['BestScore']))
    
    a_tabu_improved.clear()
    if iteration_data:
        a_tabu_improved.plot(iteration_data, score_data)
    else:
        a_tabu_improved.text(0.5, 0.5, "Waiting for data...", 
                             horizontalalignment='center',
                             verticalalignment='center',
                             transform=a_tabu_improved.transAxes)
    
    a_tabu_improved.set_title("Tabu Search Improved Progress")
    a_tabu_improved.set_xlabel("Iterations")
    a_tabu_improved.set_ylabel("Best Score (Sharpe Ratio)")

def animate_tabu(i):
    iteration_data = []
    score_data = []

    if os.path.exists('tabu_search_log.csv'):
        with open('tabu_search_log.csv', mode='r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                iteration_data.append(int(row['Iteration']))
                score_data.append(float(row['BestScore']))
    
    a_tabu.clear()
    if iteration_data:
        a_tabu.plot(iteration_data, score_data)
    else:
        a_tabu.text(0.5, 0.5, "Waiting for data...", 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=a_tabu.transAxes)
    
    a_tabu.set_title("Tabu Search Progress")
    a_tabu.set_xlabel("Iterations")
    a_tabu.set_ylabel("Best Score (Sharpe Ratio)")

def animate_hill(i):
    iteration_data = []
    score_data = []

    if os.path.exists('hill_climbing_log.csv'):
        with open('hill_climbing_log.csv', mode='r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                iteration_data.append(int(row['Iteration']))
                score_data.append(float(row['BestScore']))
    
    a_hill.clear()
    if iteration_data:
        a_hill.plot(iteration_data, score_data)
    else:
        a_hill.text(0.5, 0.5, "Waiting for data...", 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=a_hill.transAxes)
    
    a_hill.set_title("Hill Climbing Progress")
    a_hill.set_xlabel("Iterations")
    a_hill.set_ylabel("Best Score (Sharpe Ratio)")

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Metaheuristics for Stock Portfolio Optimisation")
        self.geometry("1200x1200")
        self.pages = {}
        self.animations = []

        self.addPage("Home", HomePage)
        self.addPage("SimulatedAnnealing", SimulatedAnnealingPage)
        self.addPage("GeneticAlgorithm", GeneticAlgorithmPage)
        self.addPage("GeneticTreeAlgorithm", GeneticTreeAlgorithmPage)
        self.addPage("TabuSearchImproved", ImprovedTabuSearchPage)
        self.addPage("TabuSearch", TabuSearchPage)
        self.addPage("HillClimbing", HillClimbingPage)

        self.showPage("Home")

    def addPage(self, pageName, pageClass):
        page = pageClass(self)
        self.pages[pageName] = page

    def showPage(self, pageName):
        for page in self.pages.values():
            page.pack_forget()
        page = self.pages[pageName]
        page.pack()

class HomePage(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg="#d9d9d9")
        self.title_label = tk.Label(self, text="Select an Algorithm", font=("Helvetica", 40, "bold"), bg="#d9d9d9", fg="#333")
        self.title_label.pack(pady=(300, 20))

        button_frame = tk.Frame(self, bg="#d9d9d9")
        button_frame.pack(pady=10)

        buttons_info = [
            ("Simulated Annealing", "SimulatedAnnealing"),
            ("Genetic Algorithm", "GeneticAlgorithm"),
            ("Genetic Tree Algorithm", "GeneticTreeAlgorithm"),
            ("Tabu Search Improved", "TabuSearchImproved"),
            ("Tabu Search", "TabuSearch"),
            ("Hill Climbing", "HillClimbing")
        ]
        for text, page in buttons_info:
            btn = tk.Button(
                button_frame,
                text=text,
                font=("Helvetica", 20),
                width=25,
                height=2,
                bg="#4CAF50",
                fg="white",
                activebackground="#45a049",
                bd=0,
                relief="flat",
                cursor="hand2",
                command=lambda p=page: master.showPage(p)
            )
            btn.pack(pady=8)

        tk.Label(self, text="", bg="#d9d9d9").pack(pady=220)

class SimulatedAnnealingPage(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.back_button = tk.Button(self, text="Back", font=("Helvetica", 12), command=lambda: master.showPage("Home"))
        self.back_button.pack(pady=5)
        self.label = tk.Label(self, text="Simulated Annealing", font=("Helvetica", 16))
        self.label.pack(pady=5)
        self.iterations_label = tk.Label(self, text="Iterations:")
        self.iterations_label.pack(pady=5)
        self.iterations_entry = tk.Entry(self)
        self.iterations_entry.pack(pady=5)
        self.temp_label = tk.Label(self, text="Initial Temperature:")
        self.temp_label.pack(pady=5)
        self.temp_entry = tk.Entry(self)
        self.temp_entry.pack(pady=5)
        self.reduction_label = tk.Label(self, text="Reduction Rate:")
        self.reduction_label.pack(pady=5)
        self.reduction_entry = tk.Entry(self)
        self.reduction_entry.pack(pady=5)
        self.start_button = tk.Button(self, text="Start Optimisation", font=("Helvetica", 12), command=self.start_task)
        self.start_button.pack(pady=5)
        self.solution_label = tk.Label(self, text="Best Portfolio:", font=("Helvetica", 32))
        self.solution_label.pack(pady=5)

        result_frame = tk.Frame(self)
        result_frame.pack(pady=5, fill=tk.BOTH, expand=False)
        scrollbar = tk.Scrollbar(result_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text = tk.Text(result_frame, height=8, width=60, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.result_text.config(state='disabled')
        scrollbar.config(command=self.result_text.yview)

        canvas = FigureCanvasTkAgg(f_simulated_annealing, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)   

    def start_optimisation(self):
        iterations = int(self.iterations_entry.get())
        temp = float(self.temp_entry.get())
        reduction = float(self.reduction_entry.get())

        best_portfolio = simulated_annealing(iterations, temp, reduction, calculate_portfolio_sharpe_ratio)
        formatted_result = "\n".join([
            f"{str(asset)}: {weight:.5f} ({weight * 100:.2f}%)"
            for asset, weight in sorted(best_portfolio.items())
        ])
        self.result_text.config(state='normal')  
        self.result_text.delete("1.0", tk.END)   
        self.result_text.insert(tk.END, formatted_result)
        self.result_text.config(state='disabled') 

    def start_task(self):
        threading.Thread(target=self.start_optimisation).start()

class GeneticAlgorithmPage(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.back_button = tk.Button(self, text="Back", font=("Helvetica", 12), command=lambda: master.showPage("Home"))
        self.back_button.pack(pady=5)
        self.label = tk.Label(self, text="Genetic Algorithm", font=("Helvetica", 16))
        self.label.pack(pady=5)
        self.population_label = tk.Label(self, text="Population Size:")
        self.population_label.pack(pady=5)
        self.population_entry = tk.Entry(self)
        self.population_entry.pack(pady=5)
        self.generations_label = tk.Label(self, text="Generations:")
        self.generations_label.pack(pady=5)
        self.generations_entry = tk.Entry(self)
        self.generations_entry.pack(pady=5)
        self.mutation_label = tk.Label(self, text="Mutation Rate:")
        self.mutation_label.pack(pady=5)
        self.mutation_entry = tk.Entry(self)
        self.mutation_entry.pack(pady=5)
        self.start_button = tk.Button(self, text="Start Optimisation", font=("Helvetica", 12), command=self.start_task)
        self.start_button.pack(pady=5)
        self.solution_label = tk.Label(self, text="Best Portfolio:", font=("Helvetica", 32))
        self.solution_label.pack(pady=5)

        result_frame = tk.Frame(self)
        result_frame.pack(pady=5, fill=tk.BOTH, expand=False)
        scrollbar = tk.Scrollbar(result_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text = tk.Text(result_frame, height=8, width=60, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.result_text.config(state='disabled')
        scrollbar.config(command=self.result_text.yview)

        canvas = FigureCanvasTkAgg(f_genetic, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def start_optimisation(self):
        population = int(self.population_entry.get())
        generations = int(self.generations_entry.get())
        mutation = float(self.mutation_entry.get())

        best_portfolio = genetic_algorithm(population, generations, mutation, calculate_portfolio_sharpe_ratio)
        formatted_result = "\n".join([
            f"{str(asset)}: {weight:.5f} ({weight * 100:.2f}%)"
            for asset, weight in sorted(best_portfolio.items())
        ])
        self.result_text.config(state='normal')  
        self.result_text.delete("1.0", tk.END)   
        self.result_text.insert(tk.END, formatted_result)
        self.result_text.config(state='disabled') 

    def start_task(self):
        threading.Thread(target=self.start_optimisation).start()

class GeneticTreeAlgorithmPage(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.back_button = tk.Button(self, text="Back", font=("Helvetica", 12), command=lambda: master.showPage("Home"))
        self.back_button.pack(pady=5)
        self.label = tk.Label(self, text="Genetic Tree Algorithm", font=("Helvetica", 16))
        self.label.pack(pady=5)
        self.population_label = tk.Label(self, text="Population Size:")
        self.population_label.pack(pady=5)
        self.population_entry = tk.Entry(self)
        self.population_entry.pack(pady=5)
        self.generations_label = tk.Label(self, text="Generations:")
        self.generations_label.pack(pady=5)
        self.generations_entry = tk.Entry(self)
        self.generations_entry.pack(pady=5)
        self.mutation_label = tk.Label(self, text="Mutation Rate:")
        self.mutation_label.pack(pady=5)
        self.mutation_entry = tk.Entry(self)
        self.mutation_entry.pack(pady=5)
        self.crossover_label = tk.Label(self, text="Crossover Rate:")
        self.crossover_label.pack(pady=5)
        self.crossover_entry = tk.Entry(self)
        self.crossover_entry.pack(pady=5)
        self.start_button = tk.Button(self, text="Start Optimisation", font=("Helvetica", 12), command=self.start_task)
        self.start_button.pack(pady=5)
        self.solution_label = tk.Label(self, text="Best Portfolio:", font=("Helvetica", 32))
        self.solution_label.pack(pady=5)

        result_frame = tk.Frame(self)
        result_frame.pack(pady=5, fill=tk.BOTH, expand=False)
        scrollbar = tk.Scrollbar(result_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text = tk.Text(result_frame, height=8, width=60, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.result_text.config(state='disabled')
        scrollbar.config(command=self.result_text.yview)

        canvas = FigureCanvasTkAgg(f_genetic_tree, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def start_optimisation(self):
        population = int(self.population_entry.get())
        generations = int(self.generations_entry.get())
        mutation = float(self.mutation_entry.get())
        crossover = float(self.crossover_entry.get())

        best_portfolio = genetic_tree_algorithm(population, generations, mutation, crossover, calculate_portfolio_sharpe_ratio)
        formatted_result = "\n".join([
            f"{str(asset)}: {weight:.5f} ({weight * 100:.2f}%)"
            for asset, weight in sorted(best_portfolio.items())
        ])
        self.result_text.config(state='normal')  
        self.result_text.delete("1.0", tk.END)   
        self.result_text.insert(tk.END, formatted_result)
        self.result_text.config(state='disabled') 

    def start_task(self):
        threading.Thread(target=self.start_optimisation).start()

class ImprovedTabuSearchPage(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.back_button = tk.Button(self, text="Back", font=("Helvetica", 12), command=lambda: master.showPage("Home"))
        self.back_button.pack(pady=5)
        self.label = tk.Label(self, text="Tabu Search Improved", font=("Helvetica", 16))
        self.label.pack(pady=5)
        self.portfolio_iterations_label = tk.Label(self, text="Portfolio Iterations:")
        self.portfolio_iterations_label.pack(pady=5)
        self.portfolio_iterations_entry = tk.Entry(self)
        self.portfolio_iterations_entry.pack(pady=5)
        self.tabu_iterations_label = tk.Label(self, text="Tabu Iterations:")
        self.tabu_iterations_label.pack(pady=5)
        self.tabu_iterations_entry = tk.Entry(self)
        self.tabu_iterations_entry.pack(pady=5)
        self.neighbors_label = tk.Label(self, text="Number of Neighbors:")
        self.neighbors_label.pack(pady=5)
        self.neighbors_entry = tk.Entry(self)
        self.neighbors_entry.pack(pady=5)
        self.tenure_label = tk.Label(self, text="Tabu Tenure:")
        self.tenure_label.pack(pady=5)
        self.tenure_entry = tk.Entry(self)
        self.tenure_entry.pack(pady=5)
        self.start_button = tk.Button(self, text="Start Optimisation", font=("Helvetica", 12), command=self.start_task)
        self.start_button.pack(pady=5)
        self.solution_label = tk.Label(self, text="Best Portfolio:", font=("Helvetica", 32))
        self.solution_label.pack(pady=5)

        result_frame = tk.Frame(self)
        result_frame.pack(pady=5, fill=tk.BOTH, expand=False)
        scrollbar = tk.Scrollbar(result_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text = tk.Text(result_frame, height=8, width=60, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.result_text.config(state='disabled')
        scrollbar.config(command=self.result_text.yview)

        canvas = FigureCanvasTkAgg(f_tabu_improved, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def start_optimisation(self):
        portfolio_iterations = int(self.portfolio_iterations_entry.get())
        iterations = int(self.tabu_iterations_entry.get())
        neighbors = int(self.neighbors_entry.get())
        tenure = int(self.tenure_entry.get())

        best_portfolio = tabu_search_improved(portfolio_iterations, iterations, neighbors, tenure, calculate_portfolio_sharpe_ratio)
        formatted_result = "\n".join([
            f"{str(asset)}: {weight:.5f} ({weight * 100:.2f}%)"
            for asset, weight in sorted(best_portfolio.items())
        ])
        self.result_text.config(state='normal')  
        self.result_text.delete("1.0", tk.END)   
        self.result_text.insert(tk.END, formatted_result)
        self.result_text.config(state='disabled') 
    
    def start_task(self):
        threading.Thread(target=self.start_optimisation).start()

class TabuSearchPage(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.back_button = tk.Button(self, text="Back", font=("Helvetica", 12), command=lambda: master.showPage("Home"))
        self.back_button.pack(pady=5)
        self.label = tk.Label(self, text="Tabu Search", font=("Helvetica", 16))
        self.label.pack(pady=5)
        self.tabu_iterations_label = tk.Label(self, text="Tabu Iterations:")
        self.tabu_iterations_label.pack(pady=5)
        self.tabu_iterations_entry = tk.Entry(self)
        self.tabu_iterations_entry.pack(pady=5)
        self.tabu_neighbors_label = tk.Label(self, text="Number of Neighbors:")
        self.tabu_neighbors_label.pack(pady=5)
        self.tabu_neighbors_entry = tk.Entry(self)
        self.tabu_neighbors_entry.pack(pady=5)
        self.tabu_tenure_label = tk.Label(self, text="Tabu Tenure:")
        self.tabu_tenure_label.pack(pady=5)
        self.tabu_tenure_entry = tk.Entry(self)
        self.tabu_tenure_entry.pack(pady=5)
        self.start_button = tk.Button(self, text="Start Optimisation", font=("Helvetica", 12), command=self.start_task)
        self.start_button.pack(pady=5)
        self.solution_label = tk.Label(self, text="Best Portfolio:", font=("Helvetica", 32))
        self.solution_label.pack(pady=5)

        result_frame = tk.Frame(self)
        result_frame.pack(pady=5, fill=tk.BOTH, expand=False)
        scrollbar = tk.Scrollbar(result_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text = tk.Text(result_frame, height=8, width=60, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.result_text.config(state='disabled')
        scrollbar.config(command=self.result_text.yview)

        canvas = FigureCanvasTkAgg(f_tabu, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def start_optimisation(self):
        tabu_iterations = int(self.tabu_iterations_entry.get())
        tabu_neighbors = int(self.tabu_neighbors_entry.get())
        tabu_tenure = int(self.tabu_tenure_entry.get())

        best_portfolio = tabu_search(tabu_iterations, tabu_neighbors, tabu_tenure, calculate_portfolio_sharpe_ratio)
        formatted_result = "\n".join([
            f"{str(asset)}: {weight:.5f} ({weight * 100:.2f}%)"
            for asset, weight in sorted(best_portfolio.items())
        ])
        self.result_text.config(state='normal')  
        self.result_text.delete("1.0", tk.END)   
        self.result_text.insert(tk.END, formatted_result)
        self.result_text.config(state='disabled')     
        
    def start_task(self):
        threading.Thread(target=self.start_optimisation).start()

class HillClimbingPage(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.back_button = tk.Button(self, text="Back", font=("Helvetica", 12), command=lambda: master.showPage("Home"))
        self.back_button.pack(pady=5)
        self.label = tk.Label(self, text="Hill Climbing", font=("Helvetica", 16))
        self.label.pack(pady=5)
        self.iterations_label = tk.Label(self, text="Iterations:")
        self.iterations_label.pack(pady=5)
        self.iterations_entry = tk.Entry(self)
        self.iterations_entry.pack(pady=5)
        self.neighbors_label = tk.Label(self, text="Number of Neighbors:")
        self.neighbors_label.pack(pady=5)
        self.neighbors_entry = tk.Entry(self)
        self.neighbors_entry.pack(pady=5)
        self.start_button = tk.Button(self, text="Start Optimisation", font=("Helvetica", 12), command=self.start_task)
        self.start_button.pack(pady=5)
        self.solution_label = tk.Label(self, text="Best Portfolio:", font=("Helvetica", 32))
        self.solution_label.pack(pady=5)

        result_frame = tk.Frame(self)
        result_frame.pack(pady=5, fill=tk.BOTH, expand=False)
        scrollbar = tk.Scrollbar(result_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text = tk.Text(result_frame, height=8, width=60, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.result_text.config(state='disabled')
        scrollbar.config(command=self.result_text.yview)

        canvas = FigureCanvasTkAgg(f_hill, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
    def start_optimisation(self):
        iterations = int(self.iterations_entry.get())
        neighbors = int(self.neighbors_entry.get())

        best_portfolio = hill_climbing(iterations, neighbors, calculate_portfolio_sharpe_ratio)
        formatted_result = "\n".join([
            f"{str(asset)}: {weight:.5f} ({weight * 100:.2f}%)"
            for asset, weight in sorted(best_portfolio.items())
        ])
        self.result_text.config(state='normal')  
        self.result_text.delete("1.0", tk.END)   
        self.result_text.insert(tk.END, formatted_result)
        self.result_text.config(state='disabled')  
    
    def start_task(self):
        threading.Thread(target=self.start_optimisation).start()

def main():
    app = Application()
    ani_simulated = animation.FuncAnimation(f_simulated_annealing, animate_simulated_annealing, interval=1000, cache_frame_data=False)
    ani_genetic = animation.FuncAnimation(f_genetic, animate_genetic, interval=1000, cache_frame_data=False)
    ani_genetic_tree = animation.FuncAnimation(f_genetic_tree, animate_genetic_tree, interval=1000, cache_frame_data=False)
    ani_improved_tabu = animation.FuncAnimation(f_tabu_improved, animate_tabu_improved, interval=1000, cache_frame_data=False)
    ani_tabu = animation.FuncAnimation(f_tabu, animate_tabu, interval=1000, cache_frame_data=False)
    ani_hill = animation.FuncAnimation(f_hill, animate_hill, interval=1000, cache_frame_data=False)
    app.mainloop()

if __name__ == "__main__":
    main()
