# Run this from the script directory because it cd .. into the ADSampling directory
cd ..
g++ -g -o ./src/index_ivf ./src/index_ivf.cpp -I ./src/ -I ./src/Eigen/ -O3

data='gist'
dataset_N=10000

# Paths to data and index (example ../data/gist)
data_path=./data/${data}
index_path=./data/${data}

# C_list must match the k_list in ivf.py
sqrt_N=$(echo "sqrt($dataset_N)" | bc)
# C_list=($(echo "$sqrt_N / 10" | bc) $(echo "$sqrt_N / 5" | bc) $(echo "$sqrt_N / 2" | bc) $sqrt_N $(echo "$sqrt_N * 2" | bc) $(echo "$sqrt_N * 5" | bc) $(echo "$sqrt_N * 10" | bc))
C_list=($(echo "$sqrt_N" | bc))

# Initialize run counter and total runs
current_run=0
total_runs=$(( ${#C_list[@]} * 3))

# Iterate over the values of C and each algorithm
for C in "${C_list[@]}"; do
    for algoIndex in {0..2}; do

        current_run=$((current_run + 1))
        echo -e "\n"

        # 0 - IVF, 1 - IVF++, 2 - IVF+
        if [ $algoIndex == "1" ]; then # preprocessed vectors
            echo "IVF++"
            data_file="${data_path}/O${data}_base.fvecs"   #Odataset_type.fvecs means randomly orthogonally rotated vectors
            centroid_file="${data_path}/O${data}_centroid_${C}.fvecs"
        elif [ $algoIndex == "2" ]; then # preprocessed vectors
            echo "IVF+"
            data_file="${data_path}/O${data}_base.fvecs"   #Odataset_type.fvecs means randomly orthogonally rotated vectors
            centroid_file="${data_path}/O${data}_centroid_${C}.fvecs"
        else # raw vectors
            echo "IVF"
            data_file="${data_path}/${data}_base.fvecs"
            centroid_file="${data_path}/${data}_centroid_${C}.fvecs" 
        fi

        
        # Output which C value is being indexed and for which dataset
        echo "Indexing - ${data} with C: ${C}"
        echo -e "Run ${current_run}/${total_runs} \n"
        index_file="${index_path}/${data}_ivf_${C}_${algoIndex}.index"
        
        # Run program
        ./src/index_ivf -d $data_file -c $centroid_file -i $index_file -a $algoIndex

    done
done