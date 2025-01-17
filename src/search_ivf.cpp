#define EIGEN_DONT_PARALLELIZE
#define EIGEN_DONT_VECTORIZE
#define COUNT_DIMENSION
// #define COUNT_DIST_TIME

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

long double rotation_time = 0;

void test(const Matrix<float> &Q, const Matrix<unsigned> &G, const IVF &ivf, int k)
{
    using namespace std::chrono; // Import chrono functions for brevity

    float total_time = 0, search_time = 0;
    std::vector<int> nprobes = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20};

    for (auto nprobe : nprobes)
    {
        total_time = 0;
        adsampling::clear();
        int correct = 0;
        std::cerr << std::endl
                  << std::endl
                  << nprobe << std::endl
                  << std::endl;
        for (int i = 0; i < Q.n; i++)
        {
            // Start timing
            auto start = high_resolution_clock::now();

            // Perform the search
            ResultHeap KNNs = ivf.search(Q.data + i * Q.d, k, nprobe);

            // Stop timing
            auto end = high_resolution_clock::now();

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

        float time_us_per_query = total_time / Q.n + rotation_time;
        float recall = 1.0f * correct / (Q.n * k);

        // Print results
        std::cout << nprobe
                  << " " << recall * 100.00
                  << " " << time_us_per_query
                  << " " << adsampling::tot_dimension << std::endl;
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
        {"randomized", required_argument, 0, 'd'}, // 0, 1, or 2 for IVF, IVF+, IVF++
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

    int randomize = 0;
    int subk = 0;

    while (iarg != -1)
    {
        iarg = getopt_long(argc, argv, "d:i:q:g:r:t:n:k:e:p:", longopts, &ind);
        switch (iarg)
        {
        case 'd':
            if (optarg)
                randomize = atoi(optarg);
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
        }
    }

    Matrix<float> Q(query_path);
    Matrix<unsigned> G(groundtruth_path);
    Matrix<float> P(transformation_path);

    freopen(result_path, "a", stdout);
    if (randomize)
    {
        StopW stopw = StopW();
        Q = mul(Q, P);
        rotation_time = stopw.getElapsedTimeMicro() / Q.n;
        adsampling::D = Q.d;
    }

    IVF ivf;
    ivf.load(index_path);
    // Call test with (query matrix, ground truth matrix, ivf object with index loaded, k)
    test(Q, G, ivf, subk);
    return 0;
}
