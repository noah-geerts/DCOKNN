# Run this from the script directory because it cd .. into the ADSampling directory
cd ..
g++ -g3 -o ./src/index_ivf ./src/index_ivf.cpp -I ./src/ -I ./src/Eigen/ -O3

C=41 # Number of centroids... must match k in ivf.py script
data='gist'

for adaptive in {0..2}
do
    echo "Indexing - ${data}"

    # Paths to data and index (example ../data/gist)
    data_path=./data/${data}
    index_path=./data/${data}

    if [ $adaptive == "0" ] # raw vectors 
    then
        data_file="${data_path}/${data}_base.fvecs"
        centroid_file="${data_path}/${data}_centroid_${C}.fvecs"
    else                    # preprocessed vectors                  
        data_file="${data_path}/O${data}_base.fvecs"   #Odataset_type.fvecs means randomly orthogonally rotated vectors
        centroid_file="${data_path}/O${data}_centroid_${C}.fvecs"
    fi

    # 0 - IVF, 1 - IVF++, 2 - IVF+
    index_file="${index_path}/${data}_ivf_${C}_${adaptive}.index"
    
    # Run program
    echo "./src/index_ivf -d $data_file -c $centroid_file -i $index_file -a $adaptive"
    ./src/index_ivf -d $data_file -c $centroid_file -i $index_file -a $adaptive

done