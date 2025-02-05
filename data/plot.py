import matplotlib.pyplot as plt;
import os
data = "gist"
source = "../results"
output = "./plots"

# Read a single log file given its path
def read_log_file(file_path):
    recall_qps = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line by spaces into an array of strings
            parts = line.strip().split()
            
            # Convert the array of strings to a tuple of (int, float, float, int)
            if len(parts) == 4:
                try:
                    # Tuple of data in the form (recall %, time per query in microseconds)
                    data_tuple = (float(parts[0]), float(parts[1]))
                    recall_qps.append(data_tuple)
                except ValueError as e:
                    print(f"Error converting line to tuple: {line.strip()} - {e}")
                    print(f"In file: {file_path}")
    return recall_qps

# Read all log files in a directory
def read_log_files(directory):
    results = []
    for file in os.listdir(directory):
        if file.endswith(".log"):
            file_path = os.path.join(directory, file)
            algo_tuple = (os.path.splitext(file)[0], read_log_file(file_path))
            results.append(algo_tuple)
    # Results is an array of tuples (file_name, [(recall %, time per query in microseconds), ...])
    return results

def plot_recall_qps(file_path):
    # Results is an array of tuples (file_name, [(recall %, time per query in microseconds), ...])
    results = read_log_files(file_path)

    # Define a list of markers
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', '+']

    # Plot
    plt.figure(figsize=(8, 6))
    for i, algo in enumerate(results):
        # Extract the recall and qps values (convert us per query to qps)
        recall = [x[0] for x in algo[1]]
        qps = [1 / (x[1]*1e-6) for x in algo[1]]

        # Plot the values with different markers and labels
        plt.plot(recall, qps, marker=markers[i % len(markers)], label=algo[0])

    # Log scale for the y-axis
    plt.yscale('log')

    # Labels and title
    plt.xlabel('Recall (%)')
    plt.ylabel('QPS (Queries Per Second)')
    plt.title('Performance Plot')
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()

    # Show plot
    output_file = os.path.join(output, f"{data}_recall_qps.png")
    plt.savefig(output_file)

if __name__ == '__main__':
    plot_recall_qps(os.path.join(source, data))