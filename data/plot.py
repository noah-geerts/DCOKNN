import matplotlib.pyplot as plt
import os
import argparse
import glob

source = "../results"
output = "./plots"

# Read a single log file given its path
def read_log_file(file_path):
    recall_qps = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line by spaces into an array of strings
            parts = line.strip().split()
            
            try:
                # Tuple of data in the form (recall %, time per query in microseconds, nprobe, total dimensions sampled, 
                # distance time per query in microseconds, rotation time per query in microseconds, positive objects, negative objects)
                data_tuple = (float(parts[0]), float(parts[1]), int(parts[2]), int(parts[3]), 
                            float(parts[4]), float(parts[5]), int(parts[6]), int(parts[7]))
                recall_qps.append(data_tuple)
            except ValueError as e:
                print(f"Error converting line to tuple: {line.strip()} - {e}")
                print(f"In file: {file_path}")
    return recall_qps

# Read all log files in a directory
def read_log_files(directory, algorithm):
    results = []
    for file in os.listdir(directory):
        if file.endswith(".log") and algorithm in file:
            file_path = os.path.join(directory, file)
            algo_tuple = (os.path.splitext(file)[0], read_log_file(file_path))
            results.append(algo_tuple)
    # Results is an array of tuples (file_name, [(recall %, time per query in microseconds), ...])
    return results

def plot_recall_qps(file_path, algorithm, dataset):
    # Results is an array of tuples (file_name, [(recall %, time per query in microseconds), ...])
    results = read_log_files(file_path, algorithm)

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
    plt.title(f'{algorithm} Performance Plot')
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()

    # Show plot
    if not os.path.exists(output):
        os.makedirs(output)
    output_file = os.path.join(output, f"{dataset}_{algorithm}_recall_qps.png")
    plt.savefig(output_file)
    plt.close()

def plot_total_dimensions(file_path, algorithm, dataset):
    # Get results for each variant
    results = read_log_files(file_path, algorithm)
    
    # For each algorithm variant, find the run with highest recall
    algo_names = []
    total_dimensions = []
    recalls = []
    
    for algo_name, measurements in results:
        algo_names.append(algo_name)  # Keep full algorithm name
        
        # Find the measurement with highest recall
        max_recall_measurement = max(measurements, key=lambda x: x[0])  # x[0] is recall percentage
        
        # Get dimensions sampled for highest recall run
        total_dimensions.append(max_recall_measurement[3])  # x[3] is dimensions sampled
        recalls.append(max_recall_measurement[0])  # Store recall percentage
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(algo_names)), total_dimensions)
    
    # Add 20% padding to the top of the graph
    max_height = max(total_dimensions)
    plt.ylim(0, max_height * 1.2)
    
    # Customize the plot
    plt.title(f'Dimensions Sampled at Best Recall by {algorithm} Variants')
    plt.xlabel('Algorithm Variant')
    plt.ylabel('Dimensions Sampled')
    plt.xticks(range(len(algo_names)), algo_names, rotation=45, ha='right')
    
    # Add value labels on top of each bar with recall percentage
    for bar, recall in zip(bars, recalls):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}\n({recall:.1f}% recall)',
                ha='center', va='bottom')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    if not os.path.exists(output):
        os.makedirs(output)
    output_file = os.path.join(output, f"{dataset}_{algorithm}_best_recall_dimensions.png")
    plt.savefig(output_file)
    plt.close()

def plot_time_breakdown(file_path, algorithm, dataset):
    # Get results for each variant
    results = read_log_files(file_path, algorithm)
    
    # For each algorithm variant, find the run with highest recall
    algo_names = []
    distance_times = []
    rotation_times = []
    other_times = []
    total_times = []
    recalls = []
    
    for algo_name, measurements in results:
        algo_names.append(algo_name)
        
        # Find the measurement with highest recall
        max_recall_measurement = max(measurements, key=lambda x: x[0])
        
        # Extract times (all in microseconds)
        total_time = max_recall_measurement[1]  # time per query
        distance_time = max_recall_measurement[4]  # distance time per query
        rotation_time = max_recall_measurement[5]  # rotation time per query
        other_time = total_time - distance_time - rotation_time
        
        # Store times
        total_times.append(total_time)
        distance_times.append(distance_time)
        rotation_times.append(rotation_time)
        other_times.append(other_time)
        recalls.append(max_recall_measurement[0])
    
    # Create stacked bar plot
    plt.figure(figsize=(12, 6))
    
    # Create bars
    bar_width = 0.8
    bars_positions = range(len(algo_names))
    
    # Create the stacked bars using absolute times
    plt.bar(bars_positions, distance_times, bar_width, 
            label='Distance Computation', color='#2ecc71')
    plt.bar(bars_positions, rotation_times, bar_width, 
            bottom=distance_times, label='Rotation', color='#3498db')
    plt.bar(bars_positions, other_times, bar_width,
            bottom=[sum(x) for x in zip(distance_times, rotation_times)], 
            label='Other Operations', color='#e74c3c')
    
    # Customize the plot
    plt.title(f'Time Breakdown at Best Recall by {algorithm} Variants')
    plt.xlabel('Algorithm Variant')
    plt.ylabel('Time per Query (μs)')
    plt.xticks(bars_positions, algo_names, rotation=45, ha='right')
    
    # Add labels on the bars
    for i in range(len(algo_names)):
        total_height = total_times[i]
        
        # Add recall and total time at the top
        plt.text(i, total_height + (max(total_times) * 0.03), 
                f'({recalls[i]:.1f}% recall)\n{total_height:.1f}μs', 
                ha='center', va='bottom')
        
        # Add percentage labels in the middle of each segment
        # Distance time
        if (distance_times[i] / total_height * 100) > 5:
            plt.text(i, distance_times[i]/2, 
                    f'{(distance_times[i] / total_height * 100):.1f}%', 
                    ha='center', va='center')
        
        # Rotation time
        if (rotation_times[i] / total_height * 100) > 5:
            plt.text(i, distance_times[i] + rotation_times[i]/2,
                    f'{(rotation_times[i] / total_height * 100):.1f}%', 
                    ha='center', va='center')
        
        # Other time
        if (other_times[i] / total_height * 100) > 5:
            plt.text(i, distance_times[i] + rotation_times[i] + other_times[i]/2,
                    f'{(other_times[i] / total_height * 100):.1f}%', 
                    ha='center', va='center')
    
    # Add padding at the top for labels (20% padding)
    plt.ylim(0, max(total_times) * 1.2)
    
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # Save the plot
    if not os.path.exists(output):
        os.makedirs(output)
    output_file = os.path.join(output, f"{dataset}_{algorithm}_time_breakdown.png")
    plt.savefig(output_file)
    plt.close()

def read_index_time(file_path):
    """Read the total time from an index log file."""
    with open(file_path, 'r') as file:
        # Read the last line which contains the total time
        lines = file.readlines()
        if lines:
            return int(lines[-1].strip())
    return 0

def read_projection_time(file_path):
    """Read the projection time from a projection log file."""
    with open(file_path, 'r') as file:
        return int(file.readline().strip())

def plot_indexing_breakdown(file_path, algorithm, dataset):
    # Base paths
    index_results_dir = os.path.join('..', 'index_results', dataset)
    results_dir = os.path.join('..', 'index_results', dataset)
    
    # Initialize data structures
    variants = []
    index_times = []
    projection_times = []
    
    if algorithm == 'HNSW':
        # HNSW variants
        variants = ['HNSW', 'HNSW+/++', 'HNSW_PCA/APCA']
        
        # Base HNSW
        index_file = os.path.join(results_dir, 'HNSW_index_ef500_M16.log')
        index_times.append(read_index_time(index_file))
        projection_times.append(0)  # No projection for base
        
        # ADS HNSW
        index_file = os.path.join(results_dir, 'HNSW_ADS_index_ef500_M16.log')
        index_times.append(read_index_time(index_file))
        projection_file = os.path.join(index_results_dir, 'ADS_projection.log')
        projection_times.append(read_projection_time(projection_file))
        
        # APCA HNSW
        index_file = os.path.join(results_dir, 'HNSW_APCA_index_ef500_M16.log')
        index_times.append(read_index_time(index_file))
        projection_file = os.path.join(index_results_dir, 'PCA_projection.log')
        projection_times.append(read_projection_time(projection_file))
        
    else:  # IVF
        # IVF variants
        variants = ['IVF', 'IVF+/++', 'IVF_PCA/APCA']
        
        # Base IVF
        index_files = glob.glob(os.path.join(results_dir, 'IVF_index_*.log'))
        index_time = sum(read_index_time(f) for f in index_files)
        clustering_time = read_projection_time(os.path.join(index_results_dir, 'IVF_clustering.log'))
        index_times.append(index_time + clustering_time)
        projection_times.append(0)  # No projection for base
        
        # ADS IVF
        index_files = glob.glob(os.path.join(results_dir, 'IVF_ADS_index_*.log'))
        index_time = sum(read_index_time(f) for f in index_files)
        clustering_time = read_projection_time(os.path.join(index_results_dir, 'IVF_clustering.log'))
        index_times.append(index_time + clustering_time)
        projection_time = (read_projection_time(os.path.join(index_results_dir, 'ADS_projection.log')) +
                         read_projection_time(os.path.join(index_results_dir, 'IVF_ADS_clustering.log')))
        projection_times.append(projection_time)
        
        # APCA IVF
        index_files = glob.glob(os.path.join(results_dir, 'IVF_APCA_index_*.log'))
        index_time = sum(read_index_time(f) for f in index_files)
        clustering_time = read_projection_time(os.path.join(index_results_dir, 'IVF_clustering.log'))
        index_times.append(index_time + clustering_time)
        projection_time = (read_projection_time(os.path.join(index_results_dir, 'PCA_projection.log')) +
                         read_projection_time(os.path.join(index_results_dir, 'IVF_APCA_clustering.log')))
        projection_times.append(projection_time)
    
    # Create stacked bar plot
    plt.figure(figsize=(10, 6))
    
    # Create bars
    bar_width = 0.8
    bars_positions = range(len(variants))
    
    # Create the stacked bars
    plt.bar(bars_positions, index_times, bar_width, 
            label='Index Building', color='#2ecc71')
    plt.bar(bars_positions, projection_times, bar_width, 
            bottom=index_times, label='Projection', color='#3498db')
    
    # Customize the plot
    plt.title(f'Indexing Time Breakdown for {algorithm} Variants')
    plt.xlabel('Algorithm Variant')
    plt.ylabel('Time (μs)')
    plt.xticks(bars_positions, variants)
    
    # Add value labels on the bars
    for i in range(len(variants)):
        total_height = index_times[i] + projection_times[i]
        
        # Add total time at the top
        plt.text(i, total_height + (max([sum(x) for x in zip(index_times, projection_times)]) * 0.03), 
                f'{total_height:,}μs', 
                ha='center', va='bottom')
        
        # Add percentage labels in the middle of each segment
        if index_times[i] > 0:
            plt.text(i, index_times[i]/2, 
                    f'{(index_times[i] / total_height * 100):.1f}%', 
                    ha='center', va='center')
        
        if projection_times[i] > 0:
            plt.text(i, index_times[i] + projection_times[i]/2,
                    f'{(projection_times[i] / total_height * 100):.1f}%', 
                    ha='center', va='center')
    
    # Add padding at the top for labels (20% padding)
    plt.ylim(0, max([sum(x) for x in zip(index_times, projection_times)]) * 1.2)
    
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # Save the plot
    if not os.path.exists(output):
        os.makedirs(output)
    output_file = os.path.join(output, f"{dataset}_{algorithm}_indexing_breakdown.png")
    plt.savefig(output_file)
    plt.close()

def plot_object_breakdown(file_path, algorithm, dataset):
    # Get results for each variant
    results = read_log_files(file_path, algorithm)
    
    # For each algorithm variant, find the run with highest recall
    algo_names = []
    positive_objects = []
    negative_objects = []
    recalls = []
    
    for algo_name, measurements in results:
        # Skip base algorithm log files
        if algo_name == algorithm:
            continue
            
        algo_names.append(algo_name)
        
        # Find the measurement with highest recall
        max_recall_measurement = max(measurements, key=lambda x: x[0])
        
        # Extract object counts
        positive_objects.append(max_recall_measurement[6])  # positive objects
        negative_objects.append(max_recall_measurement[7])  # negative objects
        recalls.append(max_recall_measurement[0])
    
    # Create stacked bar plot
    plt.figure(figsize=(12, 6))
    
    # Create bars
    bar_width = 0.8
    bars_positions = range(len(algo_names))
    
    # Create the stacked bars
    plt.bar(bars_positions, positive_objects, bar_width, 
            label='Positive Objects', color='#2ecc71')
    plt.bar(bars_positions, negative_objects, bar_width, 
            bottom=positive_objects, label='Negative Objects', color='#e74c3c')
    
    # Customize the plot
    plt.title(f'Object Breakdown at Best Recall by {algorithm} Variants')
    plt.xlabel('Algorithm Variant')
    plt.ylabel('Number of Objects')
    plt.xticks(bars_positions, algo_names, rotation=45, ha='right')
    
    # Add value labels on the bars
    for i in range(len(algo_names)):
        total_objects = positive_objects[i] + negative_objects[i]
        
        # Add recall and total objects at the top
        plt.text(i, total_objects + (max([sum(x) for x in zip(positive_objects, negative_objects)]) * 0.03), 
                f'({recalls[i]:.1f}% recall)\n{total_objects:,} objects', 
                ha='center', va='bottom')
        
        # Add percentage labels in the middle of each segment
        if positive_objects[i] > 0:
            plt.text(i, positive_objects[i]/2, 
                    f'{(positive_objects[i] / total_objects * 100):.1f}%', 
                    ha='center', va='center')
        
        if negative_objects[i] > 0:
            plt.text(i, positive_objects[i] + negative_objects[i]/2,
                    f'{(negative_objects[i] / total_objects * 100):.1f}%', 
                    ha='center', va='center')
    
    # Add padding at the top for labels (20% padding)
    plt.ylim(0, max([sum(x) for x in zip(positive_objects, negative_objects)]) * 1.2)
    
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # Save the plot
    if not os.path.exists(output):
        os.makedirs(output)
    output_file = os.path.join(output, f"{dataset}_{algorithm}_object_breakdown.png")
    plt.savefig(output_file)
    plt.close()

# Dictionary mapping plot types to their corresponding functions
PLOT_FUNCTIONS = {
    'qps': plot_recall_qps,
    'total_dimensions': plot_total_dimensions,
    'time_breakdown': plot_time_breakdown,
    'indexing_breakdown': plot_indexing_breakdown,
    'object_breakdown': plot_object_breakdown,
    # Add more plot types here as they are implemented
    # 'memory': plot_memory_usage,
    # etc.
}

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate performance plots for ANN algorithms')
    parser.add_argument('algorithm', choices=['IVF', 'HNSW'], help='Algorithm to plot (IVF or HNSW)')
    parser.add_argument('plot_type', choices=list(PLOT_FUNCTIONS.keys()), help='Type of plot to generate')
    parser.add_argument('data', choices=['gist', 'glove1.2m'], help='Dataset to plot')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Get the appropriate plotting function
    plot_function = PLOT_FUNCTIONS[args.plot_type]
    
    # Call the plotting function with the appropriate arguments
    plot_function(os.path.join(source, args.data), args.algorithm, args.data)

if __name__ == '__main__':
    main()