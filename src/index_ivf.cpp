#include <iostream>
#include <fstream>
#include <queue>
#include <getopt.h>
#include <unordered_set>

#include "matrix.h"
#include "utils.h"
#include "ivf/ivf.h"

using namespace std;

int main(int argc, char *argv[])
{

    std::cout << "test";
    // immutable array of type 'struct option' called longopts
    const struct option longopts[] = {
        // General Parameter
        {"help", no_argument, 0, 'h'},

        // Index Parameter
        {"adaptive", required_argument, 0, 'a'},

        // Indexing Path
        {"data_path", required_argument, 0, 'd'},
        {"centroid_path", required_argument, 0, 'c'},
        {"index_path", required_argument, 0, 'i'},
    };

    int ind;
    int iarg = 0;
    opterr = 1; // getopt error message (off: 0)

    char index_path[256] = "";
    char data_path[256] = "";
    char centroid_path[256] = "";
    char pca_data_path[256] = "";
    char pca_centroid_path[256] = "";

    size_t adaptive = 0;

    while (iarg != -1)
    {
        iarg = getopt_long(argc, argv, "a:d:c:i:p:q:", longopts, &ind);
        switch (iarg)
        {
        case 'a':
            if (optarg)
            {
                adaptive = atoi(optarg);
            }
            break;
        case 'd':
            if (optarg)
            {
                strcpy(data_path, optarg);
            }
            break;
        case 'c':
            if (optarg)
            {
                strcpy(centroid_path, optarg);
            }
            break;
        case 'i':
            if (optarg)
            {
                strcpy(index_path, optarg);
            }
            break;
        case 'p':
            if (optarg)
            {
                strcpy(pca_data_path, optarg);
            }
            break;
        case 'q':
            if (optarg)
            {
                strcpy(pca_centroid_path, optarg);
            }
            break;
        }
    }

    Matrix<float> X(data_path);
    Matrix<float> C(centroid_path);
    Matrix<float> X_pca(pca_data_path);
    Matrix<float> C_pca(pca_centroid_path);

    if (adaptive == 3)
    {
        IVF ivf(X, C, X_pca, C_pca);
        ivf.save(index_path);
    }
    else
    {
        IVF ivf(X, C, adaptive);
        ivf.save(index_path);
    }

    return 0;
}
