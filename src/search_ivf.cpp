#define EIGEN_DONT_PARALLELIZE
#define EIGEN_DONT_VECTORIZE
#define COUNT_DIMENSION
#define COUNT_DIST_TIME

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <ctime>
#include <cmath>
#include <matrix.h>
#include <utils.h>
#include <ivf/ivf.h>
#include <adsampling.h>
#include <getopt.h>

using namespace std;

const int MAXK = 100;

long double rotation_time_per_query = 0;

void test(const Matrix<float> &Q, const Matrix<unsigned> &G, const IVF &ivf, int k, int algoIndex, Matrix<float> &magnitudes, Matrix<float> &variances)
{
    using namespace std::chrono; // Import chrono functions for brevity

    float total_time = 0, search_time = 0;
    float base_nprobes = std::sqrt(ivf.C);
    std::vector<int> nprobes = {(int)(0.25*base_nprobes), (int)(0.5*base_nprobes), (int)base_nprobes, (int)2*base_nprobes, (int)4*base_nprobes, (int)8*base_nprobes};
    // if (ivf.C > 100)
    // {
    //     nprobes = {10, 12, 14, 16, 18, 20};
    // }
    // if (ivf.C > 1000)
    // {
    //     nprobes = {20, 30, 40, 50, 60, 70, 80, 90, 100};
    // }

    for (auto nprobe : nprobes)
    {
        total_time = 0;
        adsampling::clear();
        int correct = 0;
            std::cerr << nprobe << " ";
            for (int i = 0; i < Q.n; i++)
            {
                // Start timing
                auto start = high_resolution_clock::now();

                // Compute error variance and magnitude of query vector
                float q_mag = 0;
                float *err_sd = new float[adsampling::D];

                if (algoIndex == 3 || algoIndex == 4)
                {
                    // Calculate running sums from back to front
                    float err_var = 0;
                    for (int j = adsampling::D - 1; j >= 0; j--)
                    {
                        // Q magnitude calculation
                        float q_j = Q.data[i * Q.d + j];
                        q_mag += q_j * q_j;

                        // Error variance calculation
                        err_var += q_j * q_j * variances.data[j];
                        err_sd[j] = std::sqrt(4 * err_var);
                    }
                }

                // Perform the search
                ResultHeap KNNs = ivf.search(Q.data + i * Q.d, k, nprobe, algoIndex, magnitudes, err_sd, q_mag);

                // Stop timing
                auto end = high_resolution_clock::now();

                // Free err_sd array
                delete[] err_sd;

                // Calculate elapsed time in microseconds
                auto elapsed_time = duration_cast<microseconds>(end - start).count();
                total_time += elapsed_time;

                // Recall calculation
                while (!KNNs.empty())
                {
                    int id = KNNs.top().second;
                    KNNs.pop();
                    for (int j = 0; j < k; j++)
                    {
                        if (id == G.data[i * G.d + j])
                        {
                            correct++;
                        }
                    }
                }
            }
        

        float time_us_per_query = total_time / Q.n + rotation_time_per_query;
        float distance_time_us_per_query = adsampling::distance_time / Q.n;
        float recall = 1.0f * correct / (Q.n * k);

        // Print results
        std::cout << recall * 100.00
                  << " " << time_us_per_query
                  << " " << nprobe
                  << " " << adsampling::tot_dimension
                  << " " << distance_time_us_per_query
                  << " " << rotation_time_per_query
                  << std::endl;
    }
}

int main(int argc, char *argv[])
{

    std::cout << "Main function in search_ivf.cpp:" << std::endl;
    std::cout.flush();
    // Array of option
    const struct option longopts[] = {
        // General Parameter
        {"help", no_argument, 0, 'h'},

        // Query Parameter
        {"algoIndex", required_argument, 0, 'd'}, // 0, 1, 3, or 4 for IVF, IVF+, IVF++, IVF_PCA, IVF_APCA
        {"K", required_argument, 0, 'k'},
        {"epsilon0", required_argument, 0, 'e'},
        {"delta_d", required_argument, 0, 'p'},

        // Indexing Path
        {"dataset", required_argument, 0, 'n'},             // dataset name (ex. gist)
        {"index_path", required_argument, 0, 'i'},          // path of ivf index
        {"query_path", required_argument, 0, 'q'},          // path of query vectors
        {"groundtruth_path", required_argument, 0, 'g'},    // path of ground truth NN indexes
        {"result_path", required_argument, 0, 'r'},         // where to write result
        {"transformation_path", required_argument, 0, 't'}, // path of transformation matrix
    };

    int ind;
    int iarg = 0;
    opterr = 1; // getopt error message (off: 0)

    char index_path[256] = "";
    char query_path[256] = "";
    char groundtruth_path[256] = "";
    char result_path[256] = "";
    char dataset[256] = "";
    char transformation_path[256] = "";
    char mean_path[256] = "";
    char variances_path[256] = "";
    char magnitudes_path[256] = "";

    int algoIndex = 0;
    int subk = 0;

    while (iarg != -1)
    {
        iarg = getopt_long(argc, argv, "d:i:q:g:r:t:n:k:e:p:m:v:s:", longopts, &ind);
        switch (iarg)
        {
        case 'd':
            if (optarg)
                algoIndex = atoi(optarg);
            break;
        case 'k':
            if (optarg)
                subk = atoi(optarg);
            break;
        case 'e':
            if (optarg)
                adsampling::epsilon0 = atof(optarg);
            break;
        case 'p':
            if (optarg)
                adsampling::delta_d = atoi(optarg);
            break;
        case 'i':
            if (optarg)
                strcpy(index_path, optarg);
            break;
        case 'q':
            if (optarg)
                strcpy(query_path, optarg);
            break;
        case 'g':
            if (optarg)
                strcpy(groundtruth_path, optarg);
            break;
        case 'r':
            if (optarg)
                strcpy(result_path, optarg);
            break;
        case 't':
            if (optarg)
                strcpy(transformation_path, optarg);
            break;
        case 'n':
            if (optarg)
                strcpy(dataset, optarg);
            break;
        case 'm':
            if (optarg)
                strcpy(mean_path, optarg);
            break;
        case 'v':
            if (optarg)
                strcpy(variances_path, optarg);
            break;
        case 's':
            if (optarg)
                strcpy(magnitudes_path, optarg);
            break;
        }
    }

    Matrix<float> Q(query_path);
    Matrix<unsigned> G(groundtruth_path);
    Matrix<float> P(transformation_path);
    Matrix<float> mean(mean_path);
    Matrix<float> magnitudes(magnitudes_path);
    Matrix<float> variances(variances_path);

    freopen(result_path, "a", stdout);

    // run this only if algoIndex is 1 or 2 (so for IVF+= and IVF+)
    if (algoIndex == 1 || algoIndex == 2)
    {
        StopW stopw = StopW();
        Q = mul(Q, P);
        rotation_time_per_query = stopw.getElapsedTimeMicro() / Q.n;
        adsampling::D = Q.d;
    }
    else if (algoIndex == 3 || algoIndex == 4)
    {
        StopW stopw = StopW();
        Q.subtract_rowwise(mean);
        Q = mul(Q, P);
        rotation_time_per_query = stopw.getElapsedTimeMicro() / Q.n;
        adsampling::D = Q.d;
    }

    IVF ivf;
    ivf.load(index_path);
    // Call test with (query matrix, ground truth matrix, ivf object with index loaded, k)
    test(Q, G, ivf, subk, algoIndex, magnitudes, variances);
    return 0;
}
