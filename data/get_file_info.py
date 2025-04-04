import numpy as np
import os
import argparse
import struct

def get_file_info(file_path):
    """
    Get information about a .fvecs or .ivecs file without reading all vectors.
    
    Args:
        file_path (str): Path to the .fvecs or .ivecs file
        
    Returns:
        dict: Dictionary containing file information
    """
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return None
    
    file_size = os.path.getsize(file_path)
    file_type = os.path.splitext(file_path)[1]
    
    if file_type not in ['.fvecs', '.ivecs']:
        print(f"Error: File must be .fvecs or .ivecs, got {file_type}")
        return None
    
    # Determine if it's a float or int file
    is_float = file_type == '.fvecs'
    
    # Read just the first vector's header to get dimensionality
    with open(file_path, 'rb') as f:
        # Read the first 4 bytes which contain the dimensionality
        dim_bytes = f.read(4)
        if len(dim_bytes) < 4:
            print(f"Error: File {file_path} is too small to be a valid vector file")
            return None
        
        # Convert bytes to integer (little-endian)
        dim = struct.unpack('<i', dim_bytes)[0]
        
        # Calculate vector size in bytes
        vector_size = 4 * (dim + 1)  # 4 bytes per float/int + 4 bytes for dimension
        
        # Calculate number of vectors
        num_vectors = file_size // vector_size
        
        # Verify the calculation
        if file_size % vector_size != 0:
            print(f"Warning: File size ({file_size} bytes) is not a multiple of vector size ({vector_size} bytes)")
            print(f"This might indicate a corrupted or incomplete file")
        
        # Read a sample of the first vector to verify format
        f.seek(4)  # Skip dimension header
        sample_bytes = f.read(min(16, 4 * dim))  # Read up to 16 bytes or the whole vector if smaller
        
        # Try to interpret as float or int
        try:
            if is_float:
                sample_values = struct.unpack('<' + 'f' * (len(sample_bytes) // 4), sample_bytes)
            else:
                sample_values = struct.unpack('<' + 'i' * (len(sample_bytes) // 4), sample_bytes)
            
            # Print first few values as a sample
            sample_str = ', '.join([f"{x:.4f}" if is_float else str(x) for x in sample_values[:4]])
            if len(sample_values) > 4:
                sample_str += ", ..."
        except struct.error:
            sample_str = "Unable to parse sample values"
    
    # Calculate file size in MB
    file_size_mb = file_size / (1024 * 1024)
    
    # Prepare result
    result = {
        "file_path": file_path,
        "file_type": file_type,
        "dimensionality": dim,
        "num_vectors": num_vectors,
        "file_size_bytes": file_size,
        "file_size_mb": file_size_mb,
        "vector_size_bytes": vector_size,
        "sample_values": sample_str
    }
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Get information about a .fvecs or .ivecs file')
    parser.add_argument('file_path', type=str, help='Path to the .fvecs or .ivecs file')
    
    args = parser.parse_args()
    
    info = get_file_info(args.file_path)
    
    if info:
        print(f"File: {info['file_path']}")
        print(f"Type: {info['file_type']}")
        print(f"Dimensionality: {info['dimensionality']}")
        print(f"Number of vectors: {info['num_vectors']}")
        print(f"File size: {info['file_size_mb']:.2f} MB")
        print(f"Vector size: {info['vector_size_bytes']} bytes")
        print(f"Sample values: {info['sample_values']}")

if __name__ == "__main__":
    main() 