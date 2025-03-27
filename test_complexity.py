from set_cover_measure_conquer_v2 import minimum_set_cover_reduction_rules, graph_to_set_cover
import argparse
import os
from Graph import Graph
import time
import matplotlib.pyplot as plt
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description="Minimum Set Cover with Reduction Rules")
    
    parser.add_argument("-i", "--input", required=True, help="Path to the folder that contains graphs")
    parser.add_argument("-o", "--output", default="Execution_times.png", help="Path to the plot")
    
    return parser.parse_args()


def test_complexity(graph_folder_path, rules):
    """
    solver: the function that solves the set cover
    rules: list of index indicating the rules to be applied
    returns a list of execution times for each grap
    the graphs should be named in increasing order
    """

    folder_path = graph_folder_path
    execution_times = []
    for file_name in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path): 
            print(file_name)
            graph = Graph(file_path, sol_path=None)
            U, S, node_map = graph_to_set_cover(graph)

            time_before = time.time()
            solution = minimum_set_cover_reduction_rules(S, U, rules)
            time_after = time.time()
            print(f"length of solution {len(solution)}")
            dt = time_after - time_before
            execution_times.append(dt)
            print(f"time passed to solve {dt}")
            print("Minimum Set Cover:", solution)
        
    return execution_times

def plot_execution_time(execution_times, save_name = "execution time", lower_bound = 5, upper_bound = 40, rule=1):
    """
    execution times: a list containing execution times
    """
    x_values = np.linspace(lower_bound, upper_bound, len(execution_times))

    plt.plot(x_values, execution_times, label=f"rule {rule}")
    plt.title("Execution time over number of vertices")
    plt.ylabel("execution time (s)")
    plt.xlabel("number of vertices")
    plt.legend()
    plt.savefig(save_name, format='png')

def main():
    args = parse_arguments()

    rules = []
    for i in range(0, 8):
        if i > 0:
            rules.append(i)
        execution_times = test_complexity(args.input, rules)
        plot_execution_time(execution_times, save_name=args.output, lower_bound=5, upper_bound=80, rule=i)
        


if __name__ == "__main__":
    main()