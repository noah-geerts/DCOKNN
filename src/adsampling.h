#pragma once

/*
The file is the core of the ADSampling algorithm.
We have included detailed comments in the function dist_comp.
Note that in the whole algorithm we do not calculate the square root of the distances.
*/

#include <cmath>
#include <limits>
#include <queue>
#include <vector>
#include <iostream>

using namespace std;

namespace adsampling
{

    unsigned int D = 960;      // The dimensionality of the dataset.
    float epsilon0 = 2.1;      // epsilon0 - by default 2.1, recommended in [1.0,4.0], valid in in [0, +\infty)
    unsigned int delta_d = 32; // dimension sampling for every delta_d dimensions.

    long double distance_time = 0;
    unsigned long long tot_dimension = 0;
    unsigned long long tot_dist_calculation = 0;
    unsigned long long tot_full_dist = 0;

    // to count positive and negative objects
    unsigned long long positive_objects = 0;
    unsigned long long total_objects = 0;

    void clear()
    {
        distance_time = 0;
        tot_dimension = 0;
        tot_dist_calculation = 0;
        tot_full_dist = 0;
    }

    // The hypothesis testing checks whether \sqrt{D/d} dis' > (1 +  epsilon0 / \sqrt{d}) * r.
    // We equivalently check whether dis' > \sqrt{d/D} * (1 +  epsilon0 / \sqrt{d}) * r.
    inline float ratio(const int &D, const int &i)
    {
        if (i == D)
            return 1.0;
        return 1.0 * i / D * (1.0 + epsilon0 / std::sqrt(i)) * (1.0 + epsilon0 / std::sqrt(i));
    }

    /*
        float dist_comp(const float&, const void *, const void *, float, int) is a generic function for DCOs.

        When D, epsilon_0 and delta_d can be pre-determined, it is highly suggested to define them as constexpr and provide dataset-specific functions.
    */
    float dist_comp(const float &dis, const void *data, const void *query,
                    float res = 0, int i = 0)
    {
        // If the algorithm starts a non-zero dimensionality (i.e., the case of IVF++), we conduct the hypothesis testing immediately.
        if (i && res >= dis * ratio(D, i))
        {
#ifdef COUNT_DIMENSION // This gets defined in the search.cpp files
            tot_dimension += i;
#endif
            return -res * D / i;
        }
        float *q = (float *)query;
        float *d = (float *)data;

        while (i < D)
        {
            // It continues to sample additional delta_d dimensions.
            int check = std::min(delta_d, D - i); // only sample D - i dimensions if there aren't enough left for delta_d
            i += check;                           // add these dimensions to the total number of dimensions sampled, i
            for (int j = 1; j <= check; j++)
            {
                float t = *d - *q;
                d++;          // iterate to the next float in the data array
                q++;          // iterate to the next float in the query array
                res += t * t; // add square distance to res
                // (res is the sum of squared differences).
                // so sqrt(res) is the partial euclidean distance
            }
            // Hypothesis tesing
            if (res >= dis * ratio(D, i))
            {
#ifdef COUNT_DIMENSION
                tot_dimension += i;
#endif
                // If the null hypothesis is reject, we return the approximate distance.
                // We return -dis' to indicate that it's a negative object.
                return -res * D / i;
            }
        }
#ifdef COUNT_DIMENSION
        tot_dimension += D;
#endif
        // We return the exact distance when we have sampled all the dimensions.
        // The last iterration of the while loop will have i == D, so if we return
        // res here, we are returning the full distance as a positive object,
        // (it's distance is < k so we know it will go on the heap).
        return res;
    }

    float dist_comp_pca(const float &dis, const void *data, const void *query, const float *err_sd, const float &C1,
                        float m = 3)
    {
        float *q = (float *)query;
        float *d = (float *)data;
        float sigma = err_sd[delta_d];

        // Compute C2
        float C2 = 0;
        for (int i = 0; i < delta_d; i++)
        {
            float mult = *q * *d;
            q++;
            d++;
            C2 += mult;
        }
        C2 *= 2;

        // Reject point if it is likely very far from threshold given data distribution
        float partial_dis = C1 - C2;
        if (partial_dis - m * sigma >= dis)
        {
#ifdef COUNT_DIMENSION
            tot_dimension += delta_d;
#endif
            return -1 * partial_dis;
        }

        // Compute remaining term to return full distance
        float C3 = 0;
        for (int i = 0; i < D - delta_d; i++)
        {
            float mult = *q * *d;
            q++;
            d++;
            C3 += mult;
        }
        C3 *= 2;

#ifdef COUNT_DIMENSION
        tot_dimension += D;
#endif
        return C1 - C2 - C3;
    }

    float dist_comp_adaptive_pca(const float &dis, const void *data, const void *query, const float *err_sd, const float &C1,
                                 float m = 3)
    {
        float *q = (float *)query;
        float *d = (float *)data;

        int p = 0;
        float C2 = 0;
        while (p < D)
        {
            int new_p = std::min(D, p + delta_d);
            float temp = 0;
            for (int i = p; i < new_p; i++)
            {
                float mult = *q * *d;
                q++;
                d++;
                temp += mult;
            }
            C2 += 2 * temp;

            // Return full distance if p == D
            float partial_dis = C1 - C2;
            if (new_p == D)
            {
#ifdef COUNT_DIMENSION
                tot_dimension += new_p;
#endif
                return partial_dis;
            }

            // Reject point if it is likely very far from threshold given data distribution
            float sigma = err_sd[new_p];
            if (partial_dis - m * sigma >= dis)
            {
#ifdef COUNT_DIMENSION
                tot_dimension += new_p;
#endif
                return -1 * partial_dis;
            }

            // Iteratively sample more dimensions
            p = new_p;
        }

        // This should never be reached because the loop will eventuall reach p == D
        return 0;
    };
}

float sqr_dist(float *a, float *b, int D)
{
    float ret = 0;
    for (int i = 0; i != D; i++)
    {
        float tmp = (*a - *b);
        ret += tmp * tmp;
        a++;
        b++;
    }
    return ret;
}