/*
We highlight the function search which are closely related to our proposed algorithms.
We have included detailed comments in these functions.

We explain the important variables for the enhanced IVF as follows.
1. d - It represents the number of initial dimensions.
    * For IVF  , d = D.
    * For IVF+ , d = 0.
    * For IVF++, d = delta_d.
2. L1_data - The array to store the first d dimensions of a vector.
3. res_data - The array to store the remaining dimensions of a vector.

*/
#include <limits>
#include <queue>
#include <vector>
#include <algorithm>
#include <map>

#include "adsampling.h"
#include "matrix.h"
#include "utils.h"

class IVF
{
public:
    size_t N; // Number of data points
    size_t D; // Dimensionality
    size_t C; // Number of centroids
    size_t d; // The number of dimensions in sub vectors for ADSampling

    float *L1_data;   // Flattened first d dimensions of each vector
    float *res_data;  // Flattened remaining (D - d) dimensions of each vector
    float *centroids; // Flattened centroids

    size_t *start; // Start position of each cluster in the flattened array of all data points
    size_t *len;   // Length of each cluster in the flattened array of all data points
    size_t *id;    // Indexes of data points in the flattened array of all data points

    IVF();
    IVF(const Matrix<float> &X, const Matrix<float> &_centroids, int adaptive = 0);
    IVF(const Matrix<float> &X, const Matrix<float> &_centroids, const Matrix<float> &X_pca, const Matrix<float> &_centroids_pca);
    ~IVF();

    ResultHeap search(float *query, size_t k, size_t nprobe, int algoIndex, Matrix<float> &magnitudes, float err_sd, float q_mag, float distK = std::numeric_limits<float>::max()) const;
    void save(char *filename);
    void load(char *filename);
};

IVF::IVF()
{
    N = D = C = d = 0;
    start = len = id = NULL;
    L1_data = res_data = centroids = NULL;
}

IVF::IVF(const Matrix<float> &X, const Matrix<float> &_centroids, const Matrix<float> &X_pca, const Matrix<float> &_centroids_pca)
{
    // Initialize the number of data points (N), dimensionality (D), and number of centroids (C)
    N = X.n;
    D = X.d;
    C = _centroids.n; // C is the number of centroids

    // Ensure that the dimensionality is greater than 32
    assert(D > 32);

    // Allocate memory for start, len, and id arrays
    start = new size_t[C];
    len = new size_t[C];
    id = new size_t[N];

    // Temporary storage for assigning data points to clusters
    std::vector<size_t> *temp = new std::vector<size_t>[C]; // Creates an array of vectors of size C
                                                            // (vectors are dynamic arrays that can change in size)
                                                            // Each vector, j, will store the indexes of all data points that belong to cluster j

    // Assign each data point to the nearest centroid
    for (int i = 0; i < X.n; i++)
    {
        int belong = 0;
        float dist_min = X.dist(i, _centroids, 0); // Distance from the i'th data point in X
        // to the 0'th centroid in _centroids
        for (int j = 1; j < C; j++)
        {
            float dist = X.dist(i, _centroids, j);
            if (dist < dist_min)
            {
                dist_min = dist;
                belong = j;
            }
        }
        if (i % 50000 == 0)
        {
            std::cerr << "Processing - " << i << " / " << X.n << std::endl;
        }
        temp[belong].push_back(i); // Adds data point i's index to the vector of the cluster it belongs to
    }
    std::cerr << "Cluster Generated!" << std::endl;

    // Calculate the start positions and lengths of each cluster in a flattened array of all data points
    size_t sum = 0;
    for (int i = 0; i < C; i++)
    {
        len[i] = temp[i].size(); // Len[i] is the number of data points in cluster i
        start[i] = sum;          // Start[i] is the start position of cluster i in the flattened array
        sum += len[i];           // Sum up the number of data points to get the start of the next cluster
        for (int j = 0; j < len[i]; j++)
        {
            // i is the cluster index, j is the data point index WITHIN that cluster in temp
            id[start[i] + j] = temp[i][j]; // Put the index of each data point in the flattened array
        }
    }
    // Outcome: say for example we have 3 centroids and 10 data points. Centroid 1 has 3 data points, centroid 2 has 4 data points, and centroid 3 has 3 data points.
    // Then start = [0, 3, 7] and len = [3, 4, 3], meaning centroid 1 is the elements in id from indexes 0 to 2 inclusive, centroid 2 is 3 to 6 inclusive, and centroid 3 is from 7 to 9 inclusive
    // Meaning say in temp we have [[0, 10, 7], [2, 3, 4, 5], [1, 6, 8]], id will be: [0, 10, 7, 2, 3, 4, 5, 1, 6, 8]
    // Essentially we just flattened temp from an array of vectors to an array where we use start and len to access each centroid

    // Set the dimensionality
    d = 0;

    // Allocate memory for L1_data and res_data arrays
    L1_data = new float[N * d + 1];        // Flattened first d dimensions of each vector
    res_data = new float[N * (D - d) + 1]; // Flattened remaining (D - d) dimensions of each vector
    centroids = new float[C * D];          // Flattened centroids

    // Populate L1_data and res_data arrays with data points
    // The following code fills L1_data and res_data with the data points in the same order that they appear in id, rather than the order in the original data matrix X
    for (int i = 0; i < N; i++)
    {
        int x = id[i]; // Original dataset index of the i'th data point in id
        for (int j = 0; j < D; j++)
        {
            if (j < d)
                L1_data[i * d + j] = X_pca.data[x * D + j];
            else
                res_data[i * (D - d) + j - d] = X_pca.data[x * D + j];
        }
    }

    // Copy centroid data to the centroids array
    std::memcpy(centroids, _centroids_pca.data, C * D * sizeof(float)); // Sets centroids property to the flattened centroids of the input matrix _centroids

    // Clean up temporary storage
    delete[] temp;
}

IVF::IVF(const Matrix<float> &X, const Matrix<float> &_centroids, int adaptive)
{
    // Initialize the number of data points (N), dimensionality (D), and number of centroids (C)
    N = X.n;
    D = X.d;
    C = _centroids.n; // C is the number of centroids

    // Ensure that the dimensionality is greater than 32
    assert(D > 32);

    // Allocate memory for start, len, and id arrays
    start = new size_t[C];
    len = new size_t[C];
    id = new size_t[N];

    // Temporary storage for assigning data points to clusters
    std::vector<size_t> *temp = new std::vector<size_t>[C]; // Creates an array of vectors of size C
                                                            // (vectors are dynamic arrays that can change in size)
                                                            // Each vector, j, will store the indexes of all data points that belong to cluster j

    // Assign each data point to the nearest centroid
    for (int i = 0; i < X.n; i++)
    {
        int belong = 0;
        float dist_min = X.dist(i, _centroids, 0); // Distance from the i'th data point in X
        // to the 0'th centroid in _centroids
        for (int j = 1; j < C; j++)
        {
            float dist = X.dist(i, _centroids, j);
            if (dist < dist_min)
            {
                dist_min = dist;
                belong = j;
            }
        }
        if (i % 50000 == 0)
        {
            std::cerr << "Processing - " << i << " / " << X.n << std::endl;
        }
        temp[belong].push_back(i); // Adds data point i's index to the vector of the cluster it belongs to
    }
    std::cerr << "Cluster Generated!" << std::endl;

    // Calculate the start positions and lengths of each cluster in a flattened array of all data points
    size_t sum = 0;
    for (int i = 0; i < C; i++)
    {
        len[i] = temp[i].size(); // Len[i] is the number of data points in cluster i
        start[i] = sum;          // Start[i] is the start position of cluster i in the flattened array
        sum += len[i];           // Sum up the number of data points to get the start of the next cluster
        for (int j = 0; j < len[i]; j++)
        {
            // i is the cluster index, j is the data point index WITHIN that cluster in temp
            id[start[i] + j] = temp[i][j]; // Put the index of each data point in the flattened array
        }
    }
    // Outcome: say for example we have 3 centroids and 10 data points. Centroid 1 has 3 data points, centroid 2 has 4 data points, and centroid 3 has 3 data points.
    // Then start = [0, 3, 7] and len = [3, 4, 3], meaning centroid 1 is the elements in id from indexes 0 to 2 inclusive, centroid 2 is 3 to 6 inclusive, and centroid 3 is from 7 to 9 inclusive
    // Meaning say in temp we have [[0, 10, 7], [2, 3, 4, 5], [1, 6, 8]], id will be: [0, 10, 7, 2, 3, 4, 5, 1, 6, 8]
    // Essentially we just flattened temp from an array of vectors to an array where we use start and len to access each centroid

    // Set the dimensionality for different IVF variants
    if (adaptive == 1)
        d = 32; // IVF++ - optimize cache (d = 32 by default)
    else if (adaptive == 0)
        d = D; // IVF   - plain scan
    else
        d = 0; // IVF+  - plain ADSampling

    // Allocate memory for L1_data and res_data arrays
    L1_data = new float[N * d + 1];        // Flattened first d dimensions of each vector
    res_data = new float[N * (D - d) + 1]; // Flattened remaining (D - d) dimensions of each vector
    centroids = new float[C * D];          // Flattened centroids

    // Populate L1_data and res_data arrays with data points
    // The following code fills L1_data and res_data with the data points in the same order that they appear in id, rather than the order in the original data matrix X
    for (int i = 0; i < N; i++)
    {
        int x = id[i]; // Original dataset index of the i'th data point in id
        for (int j = 0; j < D; j++)
        {
            if (j < d)
                L1_data[i * d + j] = X.data[x * D + j];
            else
                res_data[i * (D - d) + j - d] = X.data[x * D + j];
        }
    }

    // Copy centroid data to the centroids array
    std::memcpy(centroids, _centroids.data, C * D * sizeof(float)); // Sets centroids property to the flattened centroids of the input matrix _centroids

    // Clean up temporary storage
    delete[] temp;
}

IVF::~IVF()
{
    if (id != NULL)
        delete[] id;
    if (len != NULL)
        delete[] len;
    if (start != NULL)
        delete[] start;
    if (L1_data != NULL)
        delete[] L1_data;
    if (res_data != NULL)
        delete[] res_data;
    if (centroids != NULL)
        delete[] centroids;
}

ResultHeap IVF::search(float *query, size_t k, size_t nprobe, int algoIndex, Matrix<float> &magnitudes, float err_sd, float q_mag, float distK) const
{
    // the default value of distK is +inf
    Result *centroid_dist = new Result[C]; // will hold tuples, or "Results"
    // of the form (distance from query to centroid, index of centroid)

    // Find out the closest N_{probe} centroids to the query vector.
    for (int i = 0; i < C; i++)
    {
#ifdef COUNT_DIST_TIME
        StopW stopw = StopW();
#endif
        // here query will be my pca transformed query vector, and centroid will be my pca transformed centroid
        centroid_dist[i].first = sqr_dist(query, centroids + i * D, D);
#ifdef COUNT_DIST_TIME
        adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
        centroid_dist[i].second = i;
    }

    // Find out the closest N_{probe} centroids to the query vector.
    // Sorts ascending from pointer centroid_dist to pointer centroid_dist + nprobe using
    // the elements from centroid_dist to centroid_dist + C

    // Aka sorts the array such that the first nprobe centroids are the closest to the
    // query vector in ascending order, and the rest are in an unspecified order
    std::partial_sort(centroid_dist, centroid_dist + nprobe, centroid_dist + C);
    // Find our total number of candidate points
    size_t ncan = 0;
    for (int i = 0; i < nprobe; i++)
    {
        ncan += len[centroid_dist[i].second]; // sum up size of each of the nprobe centroids
    }
    if (d == D)
        adsampling::tot_dimension += 1ll * ncan * D;

    // Scan a few initial dimensions and store the distances.
    // For IVF (i.e., apply FDScanning), it should be D.
    // For IVF+ (i.e., apply ADSampling without optimizing data layout), it should be 0.
    // For IVF++ (i.e., apply ADSampling with optimizing data layout), it should be delta_d (i.e., 32).
    float *dist = new float[ncan]; // Will store the d-dimension distance between the query
                                   // and each candidate point (or 0 if d = 0, in the case of IVF+)
    Result *candidates = new Result[ncan];
    int *obj = new int[ncan]; // Will store the data index of the candidate points

    int cur = 0;
    for (int i = 0; i < nprobe; i++) // For each centroid in nprobe closest
    {
        int cluster_id = centroid_dist[i].second;
        for (int j = 0; j < len[cluster_id]; j++) // For each point in the centroid
        {
            size_t can = start[cluster_id] + j; // can = data index of the point
#ifdef COUNT_DIST_TIME
            StopW stopw = StopW();
#endif
            float tmp_dist = sqr_dist(query, L1_data + can * d, d); // temporary distance between query's first d dimensions and the data point's first d dimensions
#ifdef COUNT_DIST_TIME
            adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
            if (d > 0)
                dist[cur] = tmp_dist;
            else
                dist[cur] = 0;
            obj[cur] = can;
            cur++;
        }
    }
    ResultHeap KNNs;

    // FDScanning
    if (algoIndex == 0)
    {
        for (int i = 0; i < ncan; i++)
        {
            // Candidates will be of the form (distance from query to candidate, original index of the point)
            candidates[i].first = dist[i];
            candidates[i].second = id[obj[i]];
        }
        std::partial_sort(candidates, candidates + k, candidates + ncan);

        for (int i = 0; i < k; i++)
        {
            KNNs.emplace(candidates[i].first, candidates[i].second);
        }
    }
    // ADSampling with and without cache level optimization
    else
    {
        auto cur_dist = dist;
        for (int i = 0; i < nprobe; i++)
        {
            int cluster_id = centroid_dist[i].second;
            for (int j = 0; j < len[cluster_id]; j++)
            {
                size_t can = start[cluster_id] + j;
                // can is the index of the start of the candidate vector in the flattened array of all data points
#ifdef COUNT_DIST_TIME
                StopW stopw = StopW();
#endif
                // distK is our threshold to be a candidate. It starts at +inf
                float tmp_dist;
                if (algoIndex == 1 || algoIndex == 2)
                    tmp_dist = adsampling::dist_comp(distK, res_data + can * (D - d), query + d, *cur_dist, d);
                else
                    tmp_dist = adsampling::dist_comp_pca(distK, res_data + can * (D - d), query + d, err_sd, magnitudes.data[id[can]] + q_mag);
                // tmp_dist is negative if the object is a negative object, otherwise it is the actual distance
                // hence we do not need to check if tmp_dist < k before placing it on the heap
#ifdef COUNT_DIST_TIME
                adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
                if (tmp_dist > 0)
                {
                    KNNs.emplace(tmp_dist, id[can]);
                    if (KNNs.size() > k)
                        KNNs.pop();
                }
                if (KNNs.size() == k && KNNs.top().first < distK)
                {
                    distK = KNNs.top().first;
                }
                cur_dist++;
            }
        }
    }

    delete[] centroid_dist;
    delete[] dist;
    delete[] candidates;
    delete[] obj;
    return KNNs;
}

void IVF::save(char *filename)
{
    std::ofstream output(filename, std::ios::binary);

    output.write((char *)&N, sizeof(size_t));
    output.write((char *)&D, sizeof(size_t));
    output.write((char *)&C, sizeof(size_t));
    output.write((char *)&d, sizeof(size_t));

    if (d > 0)
        output.write((char *)L1_data, N * d * sizeof(float));
    if (d < D)
        output.write((char *)res_data, N * (D - d) * sizeof(float));
    output.write((char *)centroids, C * D * sizeof(float));

    output.write((char *)start, C * sizeof(size_t));
    output.write((char *)len, C * sizeof(size_t));
    output.write((char *)id, N * sizeof(size_t));

    output.close();
}

void IVF::load(char *filename)
{
    std::ifstream input(filename, std::ios::binary);
    cerr << filename << endl;

    if (!input.is_open())
        throw std::runtime_error("Cannot open file");

    input.read((char *)&N, sizeof(size_t));
    input.read((char *)&D, sizeof(size_t));
    input.read((char *)&C, sizeof(size_t));
    input.read((char *)&d, sizeof(size_t));
    cerr << N << " " << D << " " << C << " " << d << endl;

    L1_data = new float[N * d + 10];
    res_data = new float[N * (D - d) + 10];
    centroids = new float[C * D];

    start = new size_t[C];
    len = new size_t[C];
    id = new size_t[N];

    if (d > 0)
        input.read((char *)L1_data, N * d * sizeof(float));
    if (d < D)
        input.read((char *)res_data, N * (D - d) * sizeof(float));
    input.read((char *)centroids, C * D * sizeof(float));

    input.read((char *)start, C * sizeof(size_t));
    input.read((char *)len, C * sizeof(size_t));
    input.read((char *)id, N * sizeof(size_t));

    input.close();
}
