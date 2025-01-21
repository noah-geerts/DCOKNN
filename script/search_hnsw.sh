# 0 - HNSW, 1 - HNSW++, 2 - HNSW+
cd ..

g++ -g3 ./src/search_hnsw.cpp -O3 -o ./src/search_hnsw -I ./src/

data='glove1.2m'

path=./data/
result_path=./results/${data}

# Make result directory if it doesn't exist
if [ ! -d ${result_path} ]; then
    mkdir -p ${result_path}
fi

ef=500
M=16

#R
for algoIndex in {0..2}
do
if [ $algoIndex == "1" ]
then 
    echo "HNSW++"
    index="${path}/${data}/O${data}_ef${ef}_M${M}.index"
elif [ $algoIndex == "2" ]
then 
    echo "HNSW+"
    index="${path}/${data}/O${data}_ef${ef}_M${M}.index"
else
    echo "HNSW"
    index="${path}/${data}/${data}_ef${ef}_M${M}.index"    
fi

res="${result_path}/HNSW_ef${ef}_M${M}_${algoIndex}.log"
query="${path}/${data}/${data}_query.fvecs"
gnd="${path}/${data}/${data}_groundtruth.ivecs"
trans="${path}/${data}/O.fvecs"

./src/search_hnsw -d ${algoIndex} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} 

done


