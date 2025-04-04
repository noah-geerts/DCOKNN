# Run this from the script directory because it cd .. into the ADSampling directory
cd ..
g++ -g -o ./src/index_ivf ./src/index_ivf.cpp -I ./src/ -I ./src/Eigen/ -O3

data='word2vec'
dataset_N=1000000

# Paths to data and index (example ../data/gist)
data_path=./data/${data}
index_path=./data/${data}/ivf_indexes
result_path=./index_results/${data}
if [ ! -d ${result_path} ]; then
    mkdir -p ${result_path}
fi

# Make index directory if it doesn't exist
if [ ! -d ${index_path} ]; then
    mkdir -p ${index_path}
fi

# C_list must match the k_list in ivf.py
sqrt_N=$(echo "sqrt($dataset_N)" | bc)
# C_list=($(echo "$sqrt_N / 10" | bc) $(echo "$sqrt_N / 5" | bc) $(echo "$sqrt_N / 2" | bc) $sqrt_N $(echo "$sqrt_N * 2" | bc) $(echo "$sqrt_N * 5" | bc) $(echo "$sqrt_N * 10" | bc))
C_list=($(echo "$sqrt_N" | bc))

# Initialize run counter and total runs
current_run=0
total_runs=$(( ${#C_list[@]} * 3))

# Iterate over the values of C and each algorithm
for C in "${C_list[@]}"; do
    for algoIndex in {0..4}; do
        echo -e "\n"

        # 0 - IVF, 1 - IVF++, 2 - IVF+, 3 - IVF_PCA
        if [ $algoIndex == "1" ]; then # preprocessed vectors
            continue
        elif [ $algoIndex == "2" ]; then # preprocessed vectors
            echo "IVF+/++"
            data_file="${data_path}/O${data}_base.fvecs"   #Odataset_type.fvecs means randomly orthogonally rotated vectors
            centroid_file="${data_path}/O${data}_centroid_${C}.fvecs"
            result_file="${result_path}/IVF_ADS_index_C${C}.log"
        elif [ $algoIndex == "3" ]; then # PCA-space vectors
            echo "IVF_PCA/APCA"
            data_file="${data_path}/PCA${data}_base.fvecs"
            centroid_file="${data_path}/PCA${data}_centroid_${C}.fvecs"
            result_file="${result_path}/IVF_APCA_index_C${C}.log"
        elif [ $algoIndex == "4" ]; then # PCA-space vectors
            continue
        else # raw vectors
            echo "IVF"
            data_file="${data_path}/${data}_base.fvecs"
            centroid_file="${data_path}/${data}_centroid_${C}.fvecs" 
            result_file="${result_path}/IVF_index_C${C}.log"
        fi

        
        # Output which C value is being indexed and for which dataset
        echo "Indexing - ${data} with C: ${C}"
        current_run=$((current_run + 1))
        echo -e "Run ${current_run}/${total_runs} \n"
        index_file="${index_path}/${data}_ivf_${C}_${algoIndex}.index"
        pca_data_file="${data_path}/PCA${data}_base.fvecs" # Use D-dimensional PCA-space data vectors   
        pca_centroid_file="${data_path}/PCA${data}_centroid_${C}.fvecs" # Use PCA-space centroids
        
        # Run program
        ./src/index_ivf -d $data_file -c $centroid_file -i $index_file -a $algoIndex -p $pca_data_file -q $pca_centroid_file -r $result_path
    done
done