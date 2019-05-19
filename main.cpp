#include <cmath>
#include <iostream>
#include <fstream>
#include <chrono>
#include <omp.h>
#include <Windows.h>
#include <sstream>
#include <vector>

using namespace std;

void print(long* array, int length) {
    for (int i = 0; i < length; i++) {
        cout << array[i] << " ";
    }
    cout << endl;
}

void print(int* array, int length) {
    for (int i = 0; i < length; i++) {
        cout << array[i] << " ";
    }
    cout << endl;
}

int** initEmpty(int length) {
    int** matrix = new int* [length];
    for (int i = 0; i < length; i++) {
        matrix[i] = new int[length];
        for (int j = 0; j < length; j++) {
            matrix[i][j] = 0;
        }
    }
    return matrix;
}

vector<vector<int>> matrixToVector(int** matrix, int length) {
    vector<vector<int>> vec(length);

    for (int i = 0; i < length; i++) {
        vec[i].resize(length);
        for (int j = 0; j < length; j++) {
            vec[i][j] = matrix[i][j];
        }
    }
    return vec;
}

void generate(int length, int max) {
    srand(unsigned(time(0)));
    int** matrix = initEmpty(length);
    for (int i = 0; i < length; i++) {
        for (int j = i + 1; j < length; j++) {
            matrix[i][j] = rand() % max;
            matrix[j][i] = matrix[i][j];
        }
        matrix[i][i] = 0;
    }

    std::ofstream output;
    output.open("C:\\university\\multithreading\\dijxtra\\a.txt");

    output << length << endl;
    for (int i = 0; i < length; i++) {
        output << matrix[i][0];
        for (int j = 1; j < length; j++)
            output << " " << matrix[i][j];
        output << endl;
    }
    output.close();

    delete[] matrix;
}

int** init(ifstream &inputFile, int length) {
    int** matrix = new int* [length];
    for (int i = 0; i < length; i++) {
        matrix[i] = new int[length];
        for (int j = 0; j < length; j++) {
            inputFile >> matrix[i][j];
        }
    }
    return matrix;
}

long* calculate_best(int** matrix, int length, int start) {
    bool* visited = new bool[length];
    long* distances = new long[length];
    long min;
    int minNumber;

    for (int i = 0; i < length; i++) {
        distances[i] = MAXLONG;
        visited[i] = false;
    }

    distances[start] = 0;
    visited[start] = true;
    int cur = start;

    for (int i = 0; i < length - 1; i++) {
        min = MAXLONG;
        minNumber = -1;
        for (int j = 0; j < length; j++) {
            // If not visited but reachable, recount distances
            if (!visited[j] && matrix[cur][j] != MAXLONG) {
                long distance = distances[cur] + matrix[cur][j];
                distances[j] = distances[j] == MAXLONG ? distance : (
                        distances[j] > distance ? distance : distances[j]
                );
                // If this way is shorter than others, it's going to be next entry point
                if (min > distances[j]) {
                    min = distances[j];
                    minNumber = j;
                }
            }
        }
        cur = minNumber;

        if (cur == -1) break;
        visited[cur] = true;
    }

    return distances;
}

long* calculate(int** matrix, int length, int start) {
    bool* visited = new bool[length];
    long* distances = new long[length];
    long min;
    int minNumber;

    for (int i = 0; i < length; i++) {
        distances[i] = MAXLONG;
        visited[i] = false;
    }

    distances[start] = 0;
    visited[start] = true;
    int cur = start;
    int j;

    for (int i = 0; i < length - 1; i++) {
        min = MAXLONG;
        minNumber = -1;

        for (j = 0; j < length; j++) {
            cur = minNumber;
            if(!visited[j] && (cur  == -1 || distances[j] < distances[cur])) {
                min = distances[j];
                minNumber = j;
            }
        }

        if (min < MAXLONG) {
            cur = minNumber;
        } else {
            cur = start;
        }

        if (distances[cur] == MAXLONG)
            break;
        visited[cur] = true;

        for(j = 0; j < length; j++) {
            if (distances[cur] + matrix[cur][j] < distances[j]) {
                distances[j] = distances[cur] + matrix[cur][j];
            }
        }
    }

    delete[] visited;
    return distances;
}

long* calculate_omp(int** matrix, int length, int start) {
    bool* visited = new bool[length];
    long* distances = new long[length];
    long* mins = new long[omp_get_max_threads()];
    int* minNumbers = new int[omp_get_max_threads()];
    long min = MAXLONG;


    #pragma omp parallel for shared(distances, visited) schedule(dynamic)
    for (int i = 0; i < length; i++) {
        distances[i] = MAXLONG;
        visited[i] = false;
    }

    distances[start] = 0;
    visited[start] = true;
    int cur = start;
    int j;

    for (int i = 0; i < length - 1; i++) {
        for (j = 0; j < omp_get_max_threads(); j++) {
            mins[j] = MAXLONG;
            minNumbers[j] = -1;
        }
        #pragma omp parallel for shared(matrix, distances, visited, mins, minNumbers) firstprivate(length) private(cur, j) schedule(dynamic)
        for (j = 0; j < length; j++) {
            cur = minNumbers[omp_get_thread_num()];
            if(!visited[j] && (cur == -1 || distances[j] < distances[cur])) {
                mins[omp_get_thread_num()] = distances[j];
                minNumbers[omp_get_thread_num()] = j;
            }
        }

        min = MAXLONG;
        for (j = 0; j < omp_get_max_threads(); j++) {
            if (mins[j] < min) {
                min = mins[j];
                cur = minNumbers[j];
            }
        }

        if (distances[cur] == MAXLONG)
            break;
        visited[cur] = true;

        #pragma omp parallel for firstprivate(cur, length) private(j) shared(matrix, distances) schedule(dynamic)
        for(j = 0; j < length; j++) {
            if (distances[cur] + matrix[cur][j] < distances[j]) {
                distances[j] = distances[cur] + matrix[cur][j];
            }
        }
    }

    delete[] visited;
    return distances;
}

long* calculate_omp_sized(int** matrix, int length, int start) {
    bool* visited = new bool[length];
    long* distances = new long[length];
    long* mins = new long[omp_get_max_threads()];
    int* minNumbers = new int[omp_get_max_threads()];
    long min = MAXLONG;

    int size = length / omp_get_max_threads();
    if (size == 0) {
        size = length;
        omp_set_num_threads(length);
    }

    #pragma omp parallel for shared(distances, visited) schedule(dynamic)
    for (int i = 0; i < length; i++) {
        distances[i] = MAXLONG;
        visited[i] = false;
    }

    distances[start] = 0;
    visited[start] = true;
    int cur = start;
    int j;

    for (int i = 0; i < length - 1; i++) {
        for (j = 0; j < omp_get_max_threads(); j++) {
            mins[j] = MAXLONG;
            minNumbers[j] = -1;
        }
        #pragma omp parallel for shared(matrix, distances, visited, mins, minNumbers) firstprivate(length) private(cur, j) schedule(dynamic, size)
        for (j = 0; j < length; j++) {
            cur = minNumbers[omp_get_thread_num()];
            if(!visited[j] && (cur == -1 || distances[j] < distances[cur])) {
                mins[omp_get_thread_num()] = distances[j];
                minNumbers[omp_get_thread_num()] = j;
            }
        }

        min = MAXLONG;
        for (j = 0; j < omp_get_max_threads(); j++) {
            if (mins[j] < min) {
                min = mins[j];
                cur = minNumbers[j];
            }
        }

        if (distances[cur] == MAXLONG)
            break;
        visited[cur] = true;

        #pragma omp parallel for firstprivate(cur, length) private(j) shared(matrix, distances) schedule(dynamic, size)
        for(j = 0; j < length; j++) {
            if (distances[cur] + matrix[cur][j] < distances[j]) {
                distances[j] = distances[cur] + matrix[cur][j];
            }
        }
    }

    delete[] visited;
    return distances;
}

long* calculate_omp_advanced(int** matrix, int length, int start) {
    bool* visited = new bool[length];
    long* distances = new long[length];

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < length; i++) {
        distances[i] = MAXLONG;
        visited[i] = false;
    }

    distances[start] = 0;

    for (int i = 0; i < length; ++i) {
        int minNumber = -1;
        long min = MAXLONG;
        #pragma omp parallel
        {
            int localMinNumber = -1;
            long localMin = MAXLONG;

            #pragma omp for schedule(dynamic) nowait
            for (int j = 0; j < length; ++j) {
                if (!visited[j] && localMin > distances[j]) {
                    localMinNumber = j;
                    localMin = distances[j];
                }
            }

            #pragma omp critical
            {
                if (localMin < min) {
                    min = localMin;
                    minNumber = localMinNumber;
                }
            };
        }

        visited[minNumber] = true;

        #pragma omp parallel for schedule(dynamic)
        for(int j = 0; j < length; ++j) {
            if (!visited[j] && matrix[minNumber][j] < MAXLONG && matrix[minNumber][j] >= 0) {
                distances[j] = distances[j] < distances[minNumber] + matrix[minNumber][j] ? distances[j] : distances[minNumber] + matrix[minNumber][j];
            }
        }
    }

    delete[] visited;
    return distances;
}

long* calculate_omp_advanced_sized(int** matrix, int length, int start) {
    bool *visited = new bool[length];
    long *distances = new long[length];

    int size = length / omp_get_max_threads();
    if (size == 0) {
        size = length;
        omp_set_num_threads(length);
    }

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < length; i++) {
        distances[i] = MAXLONG;
        visited[i] = false;
    }

    distances[start] = 0;

    for (int i = 0; i < length; ++i) {
        int minNumber = -1;
        long min = MAXLONG;
        #pragma omp parallel
        {
            int localMinNumber = -1;
            long localMin = MAXLONG;

            #pragma omp for schedule(dynamic) nowait
            for (int j = 0; j < length; ++j) {
                if (!visited[j] && localMin > distances[j]) {
                    localMinNumber = j;
                    localMin = distances[j];
                }
            }

            #pragma omp critical
            {
                if (localMin < min) {
                    min = localMin;
                    minNumber = localMinNumber;
                }
            };
        }

        visited[minNumber] = true;

        #pragma omp parallel for schedule(dynamic)
        for (int j = 0; j < length; ++j) {
            if (!visited[j] && matrix[minNumber][j] < MAXLONG && matrix[minNumber][j] >= 0) {
                distances[j] = distances[j] < distances[minNumber] + matrix[minNumber][j] ? distances[j] :
                               distances[minNumber] + matrix[minNumber][j];
            }
        }
    }

    delete[] visited;
    return distances;
}

long* calculate_best_vec(vector<vector<int>> matrix, int length, int start) {
    bool* visited = new bool[length];
    long* distances = new long[length];
    long min;
    int minNumber;

    for (int i = 0; i < length; i++) {
        distances[i] = MAXLONG;
        visited[i] = false;
    }

    distances[start] = 0;
    visited[start] = true;
    int cur = start;

    for (int i = 0; i < length - 1; i++) {
        min = MAXLONG;
        minNumber = -1;
        for (int j = 0; j < length; j++) {
            // If not visited but reachable, recount distances
            if (!visited[j] && matrix[cur][j] != MAXLONG) {
                long distance = distances[cur] + matrix[cur][j];
                distances[j] = distances[j] == MAXLONG ? distance : (
                        distances[j] > distance ? distance : distances[j]
                );
                // If this way is shorter than others, it's going to be next entry point
                if (min > distances[j]) {
                    min = distances[j];
                    minNumber = j;
                }
            }
        }
        cur = minNumber;

        if (cur == -1) break;
        visited[cur] = true;
    }

    return distances;
}

long* calculate_vec(vector<vector<int>> matrix, int length, int start) {
    bool* visited = new bool[length];
    long* distances = new long[length];
    long min;
    int minNumber;

    for (int i = 0; i < length; i++) {
        distances[i] = MAXLONG;
        visited[i] = false;
    }

    distances[start] = 0;
    visited[start] = true;
    int cur = start;
    int j;

    for (int i = 0; i < length - 1; i++) {
        min = MAXLONG;
        minNumber = -1;

        for (j = 0; j < length; j++) {
            cur = minNumber;
            if(!visited[j] && (cur  == -1 || distances[j] < distances[cur])) {
                min = distances[j];
                minNumber = j;
            }
        }

        if (min < MAXLONG) {
            cur = minNumber;
        } else {
            cur = start;
        }

        if (distances[cur] == MAXLONG)
            break;
        visited[cur] = true;

        for(j = 0; j < length; j++) {
            if (distances[cur] + matrix[cur][j] < distances[j]) {
                distances[j] = distances[cur] + matrix[cur][j];
            }
        }
    }

    delete[] visited;
    return distances;
}

long* calculate_omp_vec(vector<vector<int>> matrix, int length, int start) {
    bool* visited = new bool[length];
    long* distances = new long[length];
    long* mins = new long[omp_get_max_threads()];
    int* minNumbers = new int[omp_get_max_threads()];
    long min = MAXLONG;


#pragma omp parallel for shared(distances, visited) schedule(dynamic)
    for (int i = 0; i < length; i++) {
        distances[i] = MAXLONG;
        visited[i] = false;
    }

    distances[start] = 0;
    visited[start] = true;
    int cur = start;
    int j;

    for (int i = 0; i < length - 1; i++) {
        for (j = 0; j < omp_get_max_threads(); j++) {
            mins[j] = MAXLONG;
            minNumbers[j] = -1;
        }
#pragma omp parallel for shared(matrix, distances, visited, mins, minNumbers) firstprivate(length) private(cur, j) schedule(dynamic)
        for (j = 0; j < length; j++) {
            cur = minNumbers[omp_get_thread_num()];
            if(!visited[j] && (cur == -1 || distances[j] < distances[cur])) {
                mins[omp_get_thread_num()] = distances[j];
                minNumbers[omp_get_thread_num()] = j;
            }
        }

        min = MAXLONG;
        for (j = 0; j < omp_get_max_threads(); j++) {
            if (mins[j] < min) {
                min = mins[j];
                cur = minNumbers[j];
            }
        }

        if (distances[cur] == MAXLONG)
            break;
        visited[cur] = true;

#pragma omp parallel for firstprivate(cur, length) private(j) shared(matrix, distances) schedule(dynamic)
        for(j = 0; j < length; j++) {
            if (distances[cur] + matrix[cur][j] < distances[j]) {
                distances[j] = distances[cur] + matrix[cur][j];
            }
        }
    }

    delete[] visited;
    return distances;
}

long* calculate_omp_sized_vec(vector<vector<int>> matrix, int length, int start) {
    bool* visited = new bool[length];
    long* distances = new long[length];
    long* mins = new long[omp_get_max_threads()];
    int* minNumbers = new int[omp_get_max_threads()];
    long min = MAXLONG;

    int size = length / omp_get_max_threads();
    if (size == 0) {
        size = length;
        omp_set_num_threads(length);
    }

#pragma omp parallel for shared(distances, visited) schedule(dynamic)
    for (int i = 0; i < length; i++) {
        distances[i] = MAXLONG;
        visited[i] = false;
    }

    distances[start] = 0;
    visited[start] = true;
    int cur = start;
    int j;

    for (int i = 0; i < length - 1; i++) {
        for (j = 0; j < omp_get_max_threads(); j++) {
            mins[j] = MAXLONG;
            minNumbers[j] = -1;
        }
#pragma omp parallel for shared(matrix, distances, visited, mins, minNumbers) firstprivate(length) private(cur, j) schedule(dynamic, size)
        for (j = 0; j < length; j++) {
            cur = minNumbers[omp_get_thread_num()];
            if(!visited[j] && (cur == -1 || distances[j] < distances[cur])) {
                mins[omp_get_thread_num()] = distances[j];
                minNumbers[omp_get_thread_num()] = j;
            }
        }

        min = MAXLONG;
        for (j = 0; j < omp_get_max_threads(); j++) {
            if (mins[j] < min) {
                min = mins[j];
                cur = minNumbers[j];
            }
        }

        if (distances[cur] == MAXLONG)
            break;
        visited[cur] = true;

#pragma omp parallel for firstprivate(cur, length) private(j) shared(matrix, distances) schedule(dynamic, size)
        for(j = 0; j < length; j++) {
            if (distances[cur] + matrix[cur][j] < distances[j]) {
                distances[j] = distances[cur] + matrix[cur][j];
            }
        }
    }

    delete[] visited;
    return distances;
}

long* calculate_omp_advanced_vec(vector<vector<int>> matrix, int length, int start) {
    bool* visited = new bool[length];
    long* distances = new long[length];

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < length; i++) {
        distances[i] = MAXLONG;
        visited[i] = false;
    }

    distances[start] = 0;

    for (int i = 0; i < length; ++i) {
        int minNumber = -1;
        long min = MAXLONG;
#pragma omp parallel
        {
            int localMinNumber = -1;
            long localMin = MAXLONG;

#pragma omp for schedule(dynamic) nowait
            for (int j = 0; j < length; ++j) {
                if (!visited[j] && localMin > distances[j]) {
                    localMinNumber = j;
                    localMin = distances[j];
                }
            }

#pragma omp critical
            {
                if (localMin < min) {
                    min = localMin;
                    minNumber = localMinNumber;
                }
            };
        }

        visited[minNumber] = true;

#pragma omp parallel for schedule(dynamic)
        for(int j = 0; j < length; ++j) {
            if (!visited[j] && matrix[minNumber][j] < MAXLONG && matrix[minNumber][j] >= 0) {
                distances[j] = distances[j] < distances[minNumber] + matrix[minNumber][j] ? distances[j] : distances[minNumber] + matrix[minNumber][j];
            }
        }
    }

    delete[] visited;
    return distances;
}

long* calculate_omp_advanced_sized_vec(vector<vector<int>> matrix, int length, int start) {
    bool *visited = new bool[length];
    long *distances = new long[length];

    int size = length / omp_get_max_threads();
    if (size == 0) {
        size = length;
        omp_set_num_threads(length);
    }

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < length; i++) {
        distances[i] = MAXLONG;
        visited[i] = false;
    }

    distances[start] = 0;

    for (int i = 0; i < length; ++i) {
        int minNumber = -1;
        long min = MAXLONG;
#pragma omp parallel
        {
            int localMinNumber = -1;
            long localMin = MAXLONG;

#pragma omp for schedule(dynamic) nowait
            for (int j = 0; j < length; ++j) {
                if (!visited[j] && localMin > distances[j]) {
                    localMinNumber = j;
                    localMin = distances[j];
                }
            }

#pragma omp critical
            {
                if (localMin < min) {
                    min = localMin;
                    minNumber = localMinNumber;
                }
            };
        }

        visited[minNumber] = true;

#pragma omp parallel for schedule(dynamic)
        for (int j = 0; j < length; ++j) {
            if (!visited[j] && matrix[minNumber][j] < MAXLONG && matrix[minNumber][j] >= 0) {
                distances[j] = distances[j] < distances[minNumber] + matrix[minNumber][j] ? distances[j] :
                               distances[minNumber] + matrix[minNumber][j];
            }
        }
    }

    delete[] visited;
    return distances;
}

int main() {

    std::ifstream inputFile;
    std::ofstream outputFile;

    int length;
    int** matrix;
    vector<vector<int>> vec;

    int dynamic = omp_get_dynamic();
    omp_set_dynamic(0);

//    generate(50, 100);
//    inputFile.open("C:\\university\\multithreading\\dijxtra\\a.txt");
//    inputFile >> length;
//    matrix = init(inputFile, length);
//    inputFile.close();
//    long* a = calculate_best(matrix, length, 0);
//    long *b = calculate_omp(matrix, length, 0);
//    long *c = calculate_omp_sized(matrix, length, 0);
//    long* d = calculate_omp_advanced(matrix, length, 0);
//    long* e = calculate_omp_advanced_sized(matrix, length, 0);
//    print(a, length);
//    print(b, length);
//    print(c, length);
//    print(d, length);
//    print(e, length);
//
//    omp_set_dynamic(dynamic);
//    return 0;

    // Cores are suffocating at 4. So lets run up to 8 threads
    int maxThreads = 8;
    // We run 3 times to get avg
    int launchTimes = 3;

    outputFile.open("C:\\university\\multithreading\\dijxtra\\res_matrix.txt");
    outputFile << "N,SEQ BEST,SEQ";
    for (int i = 0; i < maxThreads; i++) {
        outputFile << ",OMP " << i + 1;
    }
    for (int i = 0; i < maxThreads; i++) {
        outputFile << ",OMP SIZED" << i + 1;
    }
    for (int i = 0; i < maxThreads; i++) {
        outputFile << ",OMP ADVANCED" << i + 1;
    }
    for (int i = 0; i < maxThreads; i++) {
        outputFile << ",OMP ADVANCED SIZED" << i + 1;
    }
    outputFile << endl;

    // TODO change back to from 1
    for (int i = 13; i <= 20; i++) {
        // TEMP
        if (i == 13) {}
        // We can't afford more volume (18)
        else if (i == 15) break;
        // 1000 2000 3000 ... 9000
        else if (i < 10) generate(i * 1000, 10000);
        // 10000 20000 30000 ... 90000
        else if ((i > 10) && (i < 20))  generate((i % 10) * 10000, 10000);
        // 100000
        else if (i == 20) generate(i * 100000, 10000);

        inputFile.open("C:\\university\\multithreading\\dijxtra\\a.txt");
        inputFile >> length;
        matrix = init(inputFile, length);
        inputFile.close();

        // Avg cycle for matrix
        for (int j = 0; j < launchTimes; j++) {
            outputFile << length;

            auto start = chrono::high_resolution_clock::now();
            long *dijxtra_seq_best = calculate_best(matrix, length, 0);
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            std::cout << "SEQ BEST.      Length: " << length << ". Attempt: " << j + 1 << ". Threads: -. Time: " << duration.count() << std::endl;
            outputFile << "," << duration.count();

            delete[] dijxtra_seq_best;

            start = chrono::high_resolution_clock::now();
            long *dijxtra_seq = calculate(matrix, length, 0);
            stop = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            std::cout << "SEQ.           Length: " << length << ". Attempt: " << j + 1 << ". Threads: -. Time: " << duration.count() << std::endl;
            outputFile << "," << duration.count();

            delete[] dijxtra_seq;

            // Threads cycle
            for (int k = 0; k < maxThreads; k++) {
                omp_set_num_threads(k + 1);

                start = chrono::high_resolution_clock::now();
                long *dijxtra_omp = calculate_omp(matrix, length, 0);
                stop = chrono::high_resolution_clock::now();
                duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                std::cout << "OMP.           Length: " << length << ". Attempt: " << j + 1 << ". Threads: " << k + 1 << ". Time: " << duration.count() << std::endl;
                outputFile << "," << duration.count();

                delete[] dijxtra_omp;
            }

            for (int k = 0; k < maxThreads; k++) {
                omp_set_num_threads(k + 1);

                start = chrono::high_resolution_clock::now();
                long *dijxtra_omp_sized = calculate_omp_sized(matrix, length, 0);
                stop = chrono::high_resolution_clock::now();
                duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                std::cout << "OMP SIZED.     Length: " << length << ". Attempt: " << j + 1 << ". Threads: " << k + 1
                          << ". Time: " << duration.count() << std::endl;
                outputFile << "," << duration.count();

                delete[] dijxtra_omp_sized;
            }

            for (int k = 0; k < maxThreads; k++) {
                omp_set_num_threads(k + 1);

                start = chrono::high_resolution_clock::now();
                long *dijxtra_omp_adv = calculate_omp_advanced(matrix, length, 0);
                stop = chrono::high_resolution_clock::now();
                duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                std::cout << "OMP ADV.       Length: " << length << ". Attempt: " << j + 1 << ". Threads: " << k + 1
                          << ". Time: " << duration.count() << std::endl;
                outputFile << "," << duration.count();

                delete[] dijxtra_omp_adv;
            }

            for (int k = 0; k < maxThreads; k++) {
                omp_set_num_threads(k + 1);

                start = chrono::high_resolution_clock::now();
                long *dijxtra_omp_adv_sized = calculate_omp_advanced_sized(matrix, length, 0);
                stop = chrono::high_resolution_clock::now();
                duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                std::cout << "OMP ADV SIZED. Length: " << length << ". Attempt: " << j + 1 << ". Threads: " << k + 1
                          << ". Time: " << duration.count() << std::endl;
                outputFile << "," << duration.count();

                delete[] dijxtra_omp_adv_sized;
            }

            outputFile << endl;
        }

        delete[] matrix;
    }

    outputFile.close();

    omp_set_dynamic(dynamic);
    return 0;
}