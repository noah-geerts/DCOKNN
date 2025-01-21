# 0 - HNSW, 1 - HNSW++, 2 - HNSW+
cd ..

g++ -g ./src/search_hnsw.cpp -O3 -o ./src/search_hnsw -I ./src/

data='gist'
path=./data
result_path=./results/${data}

# Make result directory if it doesn't exist
if [ ! -d ${result_path} ]; then
    mkdir -p ${result_path}
fi

query="${path}/${data}/${data}_query.fvecs"
gnd="${path}/${data}/${data}_groundtruth.ivecs"
trans="${path}/${data}/O.fvecs"

# Delete the HNSW(+/++).log files if they exists (they will be appended to)
if [ -f ${result_path}/HNSW.log ]; then
    rm ${result_path}/HNSW.log
fi
if [ -f ${result_path}/HNSW+.log ]; then
    rm ${result_path}/HNSW+.log
fi
if [ -f ${result_path}/HNSW++.log ]; then
    rm ${result_path}/HNSW++.log
fi

# Define the values for M and ef
M_values=(12 18 24 30 36 48 60 72 84 96)
efConstruction_values=(500)

# Initialize run counter
current_run=0
total_runs=$(( ${#M_values[@]} * ${#efConstruction_values[@]} * 3 ))

# Iterate over the values of M and ef
for M in "${M_values[@]}"; do
  for efConstruction in "${efConstruction_values[@]}"; do
    for algoIndex in {0..2}; do

      current_run=$((current_run + 1))
      echo -e "\n"

      # Echo which algorithm is being run and set the index and result file
      if [ $algoIndex == "1" ]; then 
          echo "HNSW++"
          index="${path}/${data}/O${data}_ef${efConstruction}_M${M}.index"
          res="${result_path}/HNSW++.log"
      elif [ $algoIndex == "2" ]; then 
          echo "HNSW+"
          index="${path}/${data}/O${data}_ef${efConstruction}_M${M}.index"
          res="${result_path}/HNSW+.log"
      else
          echo "HNSW"
          index="${path}/${data}/${data}_ef${efConstruction}_M${M}.index"
          res="${result_path}/HNSW.log"    
      fi

      # Output which parameter pair is being benchmarked
      echo "Benchmarking with efConstruction: ${ef} and M: ${M} on ${data}"
      echo -e "Run ${current_run}/${total_runs} \n"

      ./src/search_hnsw -d ${algoIndex} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans}
    done
  done
done