import os
source = "../results"

def read_log_file(file_path):
    results = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line by spaces into an array of strings
            parts = line.strip().split()
            
            # Convert the array of strings to a tuple of (int, float, float, int)
            if len(parts) == 4:
                try:
                    # Tuple of data in the form (nprobes, recall %, time per query in microseconds, total number of dimensions used for all distance calculations)
                    data_tuple = (int(parts[0]), float(parts[1]), float(parts[2]), int(parts[3]))
                    results.append(data_tuple)
                except ValueError as e:
                    print(f"Error converting line to tuple: {line.strip()} - {e}")
                    print(f"In file: {file_path}")
    return results

def plot_ivf(file_path):
    results = read_log_file(file_path)
    print(results)

if __name__ == '__main__':
    log_file_path = os.path.join(source, "gist_IVF41_0.log")
    plot_ivf(log_file_path)