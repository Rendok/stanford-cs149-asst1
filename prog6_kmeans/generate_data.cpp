#include <fstream>
#include <iostream>
#include <random>
#include <stdlib.h>

using namespace std;

int main() {
    // Parameters - 100x bigger dataset
    int M = 1000000;  // Number of data points (100x increase: 10,000 -> 1,000,000)
    int N = 100;      // Number of dimensions (100-D as requested)
    int K = 5;        // Number of clusters
    double epsilon = 0.1;

    // Allocate arrays
    double *data = new double[M * N];
    double *clusterCentroids = new double[K * N];
    int *clusterAssignments = new int[M];

    // Random number generator
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::normal_distribution<double> normal(0.0, 0.15);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    // Generate random cluster centers in 100-D space
    double *centers = new double[K * N];
    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n++) {
            centers[k * N + n] = uniform(gen);
        }
    }

    // Generate data points around cluster centers
    cout << "Generating " << M << " data points..." << endl;
    for (int m = 0; m < M; m++) {
        if (m % 100000 == 0) {
            cout << "  Progress: " << (m * 100 / M) << "%" << endl;
        }
        int cluster = m % K;  // Distribute points evenly across clusters
        for (int n = 0; n < N; n++) {
            data[m * N + n] = centers[cluster * N + n] + normal(gen);
            // Clamp to [0, 1] range
            if (data[m * N + n] < 0.0) data[m * N + n] = 0.0;
            if (data[m * N + n] > 1.0) data[m * N + n] = 1.0;
        }
    }

    // Initialize cluster centroids (slightly offset from true centers)
    cout << "Initializing cluster centroids..." << endl;
    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n++) {
            clusterCentroids[k * N + n] = centers[k * N + n] + (normal(gen) * 0.1);
            if (clusterCentroids[k * N + n] < 0.0) clusterCentroids[k * N + n] = 0.0;
            if (clusterCentroids[k * N + n] > 1.0) clusterCentroids[k * N + n] = 1.0;
        }
    }

    // Initialize cluster assignments (assign to nearest centroid)
    cout << "Computing initial cluster assignments..." << endl;
    for (int m = 0; m < M; m++) {
        if (m % 100000 == 0) {
            cout << "  Progress: " << (m * 100 / M) << "%" << endl;
        }
        double minDist = 1e30;
        int bestCluster = 0;
        for (int k = 0; k < K; k++) {
            double dist = 0.0;
            for (int n = 0; n < N; n++) {
                double diff = data[m * N + n] - clusterCentroids[k * N + n];
                dist += diff * diff;
            }
            if (dist < minDist) {
                minDist = dist;
                bestCluster = k;
            }
        }
        clusterAssignments[m] = bestCluster;
    }

    // Write binary file
    cout << "Writing data.dat..." << endl;
    ofstream dataFile("data.dat", ios::out | ios::binary);
    if (!dataFile) {
        cerr << "Error opening data.dat for writing!" << endl;
        return 1;
    }

    dataFile.write((char *)&M, sizeof(int));
    dataFile.write((char *)&N, sizeof(int));
    dataFile.write((char *)&K, sizeof(int));
    dataFile.write((char *)&epsilon, sizeof(double));
    dataFile.write((char *)data, sizeof(double) * M * N);
    dataFile.write((char *)clusterCentroids, sizeof(double) * K * N);
    dataFile.write((char *)clusterAssignments, sizeof(int) * M);
    dataFile.close();

    cout << "\nGenerated data.dat with:" << endl;
    cout << "  M (data points): " << M << endl;
    cout << "  N (dimensions): " << N << endl;
    cout << "  K (clusters): " << K << endl;
    cout << "  epsilon: " << epsilon << endl;
    cout << "  Total data size: " << (M * N * sizeof(double) / (1024.0 * 1024.0)) << " MB" << endl;

    delete[] data;
    delete[] clusterCentroids;
    delete[] clusterAssignments;
    delete[] centers;

    return 0;
}



