
cd ..

g++ -g -o ./src/index_hnsw ./src/index_hnsw.cpp -I ./src/ -O3

data='word2vec'
data_path=./data/${data}
index_path=${data_path}/hnsw_indexes
result_path=./index_results/${data}
if [ ! -d ${result_path} ]; then
    mkdir -p ${result_path}
fi

# Define the values for M and efConstruction
M=16
efConstruction=500

# Make index directory if it doesn't exist
if [ ! -d ${index_path} ]; then
    mkdir -p ${index_path}
fi

# Output which index is being built
echo -e "\nIndexing - ${data} with M=${M} and efConstruction=${efConstruction}" 

#Create hsnw index on base vectors
echo -e "\nIndexing base vectors"
data_file="${data_path}/${data}_base.fvecs"
index_file="${index_path}/${data}_ef${efConstruction}_M${M}.index"
result_file="${result_path}/HNSW_index_ef${efConstruction}_M${M}.log"
./src/index_hnsw -d $data_file -i $index_file -e $efConstruction -m $M -r $result_file

#Create hnsw index on randomly rotated vectors
echo -e "\nIndexing rotated vectors"
data_file="${data_path}/O${data}_base.fvecs"
index_file="${index_path}/O${data}_ef${efConstruction}_M${M}.index"
result_file="${result_path}/HNSW_ADS_index_ef${efConstruction}_M${M}.log"
./src/index_hnsw -d $data_file -i $index_file -e $efConstruction -m $M -r $result_file

#Create hnsw index on PCA vectors
echo -e "\nIndexing pca-space vectors"
data_file="${data_path}/PCA${data}_base.fvecs"
index_file="${index_path}/PCA${data}_ef${efConstruction}_M${M}.index"
result_file="${result_path}/HNSW_APCA_index_ef${efConstruction}_M${M}.log"
./src/index_hnsw -d $data_file -i $index_file -e $efConstruction -m $M -r $result_file
