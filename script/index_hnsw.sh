
cd ..

g++ -g -o ./src/index_hnsw ./src/index_hnsw.cpp -I ./src/ -O3

data='glove1.2m'
data_path=./data/${data}
index_path=./data/${data}

# Define the values for M and efConstruction
M_values=(12 18 24 30 36 48 60 72 84 96)
efConstruction_values=(500)

# Iterate over the values of M and efConstruction
for M in "${M_values[@]}"; do
  for efConstruction in "${efConstruction_values[@]}"; do

    # Output which index is being built
    index_count=$((index_count + 1))
    total_indexes=$(( ${#M_values[@]} * ${#efConstruction_values[@]} ))
    echo -e "\nIndexing - ${data} with M=${M} and efConstruction=${efConstruction}" 
    echo -e "Building index $index_count/$total_indexes\n"

    #Create hsnw index on base vectors
    data_file="${data_path}/${data}_base.fvecs"
    index_file="${index_path}/${data}_ef${efConstruction}_M${M}.index"
    ./src/index_hnsw -d $data_file -i $index_file -e $efConstruction -m $M

    #Create hnsw index on randomly rotated vectors
    data_file="${data_path}/O${data}_base.fvecs"
    index_file="${index_path}/O${data}_ef${efConstruction}_M${M}.index"
    ./src/index_hnsw -d $data_file -i $index_file -e $efConstruction -m $M
  done
done
