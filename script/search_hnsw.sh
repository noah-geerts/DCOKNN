# 0 - HNSW, 1 - HNSW++, 2 - HNSW+
cd ..

g++ -g ./src/search_hnsw.cpp -O3 -o ./src/search_hnsw -I ./src/

data='word2vec' 
path=./data
data_path=./data/${data}
result_path=./results/${data}
index_path=${data_path}/hnsw_indexes

# Make result directory if it doesn't exist
if [ ! -d ${result_path} ]; then
    mkdir -p ${result_path}
fi

query="${path}/${data}/${data}_query.fvecs"
gnd="${path}/${data}/${data}_groundtruth.ivecs"

# Delete the HNSW(+/++/PCA/APCA).log files if they exists (they will be appended to)
if [ -f ${result_path}/HNSW.log ]; then
    rm ${result_path}/HNSW.log
fi
if [ -f ${result_path}/HNSW+.log ]; then
    rm ${result_path}/HNSW+.log
fi
if [ -f ${result_path}/HNSW++.log ]; then
    rm ${result_path}/HNSW++.log
fi
if [ -f ${result_path}/HNSW_PCA.log ]; then
    rm ${result_path}/HNSW_PCA.log
fi
if [ -f ${result_path}/HNSW_APCA.log ]; then
    rm ${result_path}/HNSW_APCA.log
fi

# Define the values for M and ef
M=16
efConstruction=500

# Initialize run counter
current_run=0
total_runs=5

# Iterate over the values of M and ef
for algoIndex in {0..4}; do
    trans="${path}/${data}/O.fvecs"
    current_run=$((current_run + 1))
    echo -e "\n"

    # Echo which algorithm is being run and set the index and result file
    if [ $algoIndex == "1" ]; then 
        echo "HNSW++"
        index="${index_path}/O${data}_ef${efConstruction}_M${M}.index"
        res="${result_path}/HNSW++.log"
    elif [ $algoIndex == "2" ]; then 
        echo "HNSW+"
        index="${index_path}/O${data}_ef${efConstruction}_M${M}.index"
        res="${result_path}/HNSW+.log"
    elif [ $algoIndex == "3" ]; then 
        echo "HNSW_PCA"
        index="${index_path}/PCA${data}_ef${efConstruction}_M${M}.index"
        res="${result_path}/HNSW_PCA.log"
        trans="${path}/${data}/PCA.fvecs"
    elif [ $algoIndex == "4" ]; then 
        echo "HNSW_APCA"
        index="${index_path}/PCA${data}_ef${efConstruction}_M${M}.index"
        res="${result_path}/HNSW_APCA.log"
        trans="${path}/${data}/PCA.fvecs"
    else
        echo "HNSW"
        index="${index_path}/${data}_ef${efConstruction}_M${M}.index"
        res="${result_path}/HNSW.log"    
    fi

    # Output which parameter pair is being benchmarked
    echo "Benchmarking with efConstruction: ${ef} and M: ${M} on ${data}"
    echo -e "Run ${current_run}/${total_runs} \n"
    mean="${path}/${data}/PCA_mean.fvecs"
    variances="${path}/${data}/${data}_variances.fvecs"
    magnitudes="${path}/${data}/${data}_magnitudes.fvecs"

    # Run program
    echo "./src/search_hnsw -d ${algoIndex} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -m ${mean}"
    ./src/search_hnsw -d ${algoIndex} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -m ${mean}  -s ${magnitudes} -v ${variances} 
done