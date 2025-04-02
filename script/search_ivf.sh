# 0 - IVF, 1 - IVF++, 2 - IVF+
cd ..

g++ -g ./src/search_ivf.cpp -O3 -o ./src/search_ivf -I ./src/

K=100
data='gist'
dataset_N=10000
path=./data
result_path=./results/${data}

# C_list must match theC_list in index_ivf.sh
sqrt_N=$(echo "sqrt($dataset_N)" | bc)
#C_list=($(echo "$sqrt_N / 10" | bc) $(echo "$sqrt_N / 5" | bc) $(echo "$sqrt_N / 2" | bc) $sqrt_N $(echo "$sqrt_N * 2" | bc) $(echo "$sqrt_N * 5" | bc) $(echo "$sqrt_N * 10" | bc))
C_list=($(echo "$sqrt_N" | bc))

# Make result directory if it doesn't exist
if [ ! -d ${result_path} ]; then
    mkdir -p ${result_path}
fi

# Delete the IVF(+/++).log files if they exists (they will be appended to in each run)
if [ -f ${result_path}/IVF.log ]; then
    rm ${result_path}/IVF.log
fi
if [ -f ${result_path}/IVF+.log ]; then
    rm ${result_path}/IVF+.log
fi
if [ -f ${result_path}/IVF++.log ]; then
    rm ${result_path}/IVF++.log
fi
if [ -f ${result_path}/IVF_PCA.log ]; then
    rm ${result_path}/IVF_PCA.log
fi
if [ -f ${result_path}/IVF_APCA.log ]; then
    rm ${result_path}/IVF_APCA.log
fi

# Initialize run counter and total runs
current_run=0
total_runs=$(( ${#C_list[@]} * 4))\

# Iterate over the values of C and each algorithm
for C in "${C_list[@]}"; do
    for algoIndex in {0..4}; do
        
        current_run=$((current_run + 1))
        echo -e "\n"

        # Echo which algorithm is being run and set result file to that algorithm
        if [ $algoIndex == "1" ]
        then 
            echo "IVF++"
            res="${result_path}/IVF++.log"
            trans="${path}/${data}/O.fvecs"
            index="${path}/${data}/ivf_indexes/${data}_ivf_${C}_2.index"
        elif [ $algoIndex == "2" ]
        then 
            echo "IVF+"
            res="${result_path}/IVF+.log"
            trans="${path}/${data}/O.fvecs"
            index="${path}/${data}/ivf_indexes/${data}_ivf_${C}_2.index"
        elif [ $algoIndex == "3" ]
        then 
            echo "IVF_PCA"
            res="${result_path}/IVF_PCA.log"
            trans="${path}/${data}/PCA.fvecs"
            index="${path}/${data}/ivf_indexes/${data}_ivf_${C}_3.index"
        elif [ $algoIndex == "4" ]
        then 
            echo "IVF_APCA"
            res="${result_path}/IVF_APCA.log"
            trans="${path}/${data}/PCA.fvecs"
            index="${path}/${data}/ivf_indexes/${data}_ivf_${C}_3.index"
        else
            echo "IVF"
            res="${result_path}/IVF.log"
            trans="${path}/${data}/O.fvecs"
            index="${path}/${data}/ivf_indexes/${data}_ivf_${C}_0.index"
        fi

        # Output which C value is being benchmarked
        echo "Benchmarking with C: ${C} on ${data}"
        echo -e "Run ${current_run}/${total_runs} \n"

        # File paths for index, query vectors, groundtruth, transformation matrix, PCA mean, data vector magnitudes, and data vector dimension variances
        query="${path}/${data}/${data}_query.fvecs"
        gnd="${path}/${data}/${data}_groundtruth.ivecs"
        mean="${path}/${data}/PCA_mean.fvecs"
        variances="${path}/${data}/${data}_variances.fvecs"
        magnitudes="${path}/${data}/${data}_magnitudes.fvecs"


        # Run program
        echo "./src/search_ivf -d ${algoIndex} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -m ${mean} -s ${magnitudes} -v ${variances} -k ${K}"
        ./src/search_ivf -d ${algoIndex} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -m ${mean} -s ${magnitudes} -v ${variances} -k ${K}
    done
done
