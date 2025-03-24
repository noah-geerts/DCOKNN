#include <iostream>
#include <fstream>
#include <queue>
#include <getopt.h>
#include <unordered_set>
#include <chrono>
#include "matrix.h"
#include "utils.h"
#include "hnswlib/hnswlib.h"

using namespace std;
using namespace hnswlib;

int main(int argc, char * argv[]) {

    const struct option longopts[] ={
        // General Parameter
        {"help",                        no_argument,       0, 'h'}, 

        // Index Parameter
        {"efConstruction",              required_argument, 0, 'e'}, 
        {"M",                           required_argument, 0, 'm'}, 

        // Indexing Path 
        {"data_path",                   required_argument, 0, 'd'},
        {"index_path",                  required_argument, 0, 'i'},
        {"result_file",                 required_argument, 0, 'r'},
    };

    int ind;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)

    char index_path[256] = "";
    char data_path[256] = "";
    char result_file[256] = "";
    size_t efConstruction = 0;
    size_t M = 0;

    while(iarg != -1){
        iarg = getopt_long(argc, argv, "e:d:i:m:r:", longopts, &ind);
        switch (iarg){
            case 'e': 
                if(optarg){
                    efConstruction = atoi(optarg);
                }
                break;
            case 'm': 
                if(optarg){
                    M = atoi(optarg);
                }
                break;
            case 'd':
                if(optarg){
                    strcpy(data_path, optarg);
                }
                break;
            case 'i':
                if(optarg){
                    strcpy(index_path, optarg);
                }
                break;
            case 'r':
                if(optarg){
                    strcpy(result_file, optarg);
                }
                break;
        }
    }
    
    // Start timing
    using namespace std::chrono;
    auto start = high_resolution_clock::now();

    //Matrix contains all the vectors for the specified data path
    Matrix<float> *X = new Matrix<float>(data_path);
    size_t D = X->d;
    size_t N = X->n;
    size_t report = 50000;

    //Here we construct the index and save it to index_path
    L2Space l2space(D);
    HierarchicalNSW<float>* appr_alg = new HierarchicalNSW<float> (&l2space, N, M, efConstruction);

    for(int i=0;i<N;i++){
        //Add point by passing in its pointer address and its index
        appr_alg->addPoint(X->data + i * D, i);
        if(i % report == 0){
            cerr << "Processing - " << i << " / " << N << endl;
        }
    }

    appr_alg->saveIndex(index_path);

    // Stop timing
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    freopen(result_file, "a", stdout);
    cout << duration.count() << endl;
    fclose(stdout);
    return 0;
}
