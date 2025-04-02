#pragma once
#ifndef MATRIX_HPP_
#define MATRIX_HPP_
#include <cstddef>
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <cstring>
#include <assert.h>
#include <Eigen/Dense>
#include <unistd.h>
#include <filesystem>

template <typename T>
class Matrix
{
private:
public:
    T *data;
    size_t n;
    size_t d;

    Matrix();                     // Default
    Matrix(char *data_file_path); // IO
    Matrix(size_t n, size_t d);   // Matrix of size n * d

    // Deconstruction
    ~Matrix() { delete[] data; }

    Matrix &operator=(const Matrix &X)
    {
        delete[] data;
        n = X.n;
        d = X.d;
        data = new T[n * d];
        memcpy(data, X.data, sizeof(T) * n * d);
        return *this;
    }

    float dist(size_t a, const Matrix<T> &B, size_t b) const;
    void subtract_rowwise(const Matrix<T> &B);
    void print_matrix() const;
};

template <typename T>
Matrix<T>::Matrix()
{
    n = 0;
    d = 0;
    data = NULL;
}

template <typename T>
Matrix<T>::Matrix(char *data_file_path)
{
    n = 0;
    d = 0;
    data = NULL;

    // Print working directory for debugging
    char cwd[1024];
    /*std::cout << std::endl
              << "Matrix constructor:" << std::endl;
    if (getcwd(cwd, sizeof(cwd)) != NULL)
    {
        std::cout << "Current working directory: " << cwd << std::endl;
    }
    else
    {
        perror("getcwd() error");
    }*/

    std::filesystem::path file_path(data_file_path);
    // std::cout << "Data file path: " << file_path << std::endl;

    std::ifstream in(file_path, std::ios::binary);
    if (!in.is_open())
    {
        std::cout << file_path << std::endl;
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    printf("File opened\n");
    in.read((char *)&d, 4);

    std::cerr << "Dimensionality - " << d << std::endl;
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    n = (size_t)(fsize / (d + 1) / 4);
    data = new T[(size_t)n * (size_t)d];
    std::cerr << "Cardinality - " << n << std::endl;
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < n; i++)
    {
        if (i % 50000 == 0)
        {
            std::cerr << "Reading from " << data_file_path << " - " << ((float)i / n) * 100 << "% - " << i << " / " << n << std::endl;
        }
        in.seekg(4, std::ios::cur);
        in.read((char *)(data + i * d), d * 4);
    }
    in.close();
}

template <typename T>
Matrix<T>::Matrix(size_t _n, size_t _d)
{
    n = _n;
    d = _d;
    data = new T[n * d];
}

template <typename T>
Matrix<T> mul(const Matrix<T> &A, const Matrix<T> &B)
{

    Eigen::MatrixXf _A(A.n, A.d);
    Eigen::MatrixXf _B(B.n, B.d);
    Eigen::MatrixXf _C(A.n, B.d);

    for (int i = 0; i < A.n; i++)
        for (int j = 0; j < A.d; j++)
            _A(i, j) = A.data[i * A.d + j];

    for (int i = 0; i < B.n; i++)
        for (int j = 0; j < B.d; j++)
            _B(i, j) = B.data[i * B.d + j];

    _C = _A * _B;

    Matrix<T> result(A.n, B.d);

    for (int i = 0; i < A.n; i++)
        for (int j = 0; j < B.d; j++)
            result.data[i * B.d + j] = _C(i, j);

    return result;
}

// Returns distance between the a'th point in this matrix and the b'th point in matrix B
template <typename T>
float Matrix<T>::dist(size_t a, const Matrix<T> &B, size_t b) const
{
    float dist = 0;
    for (size_t i = 0; i < d; i++)
    {
        dist += (data[a * d + i] - B.data[b * d + i]) * (data[a * d + i] - B.data[b * d + i]);
    }
    return dist;
}

template <typename T>
void Matrix<T>::subtract_rowwise(const Matrix<T> &B)
{
    // Ensure B has the correct shape (1, d)
    assert(B.n == 1 && B.d == this->d && "The second matrix must be of shape (1, d)");

    // Subtract B from each row of this matrix
    for (size_t i = 0; i < this->n; ++i)
    {
        for (size_t j = 0; j < this->d; ++j)
        {
            this->data[i * this->d + j] -= B.data[j];
        }
    }
}

template <typename T>
void Matrix<T>::print_matrix() const
{
    // Print shape
    std::cout << "Matrix shape: (" << n << ", " << d << ")" << std::endl;

    // Print contents
    for (size_t i = 0; i < n; i++)
    {
        std::cout << "[ ";
        for (size_t j = 0; j < d; j++)
        {
            std::cout << data[i * d + j] << " ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << std::endl;
}

#endif
