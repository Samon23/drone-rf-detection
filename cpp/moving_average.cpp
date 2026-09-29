// Moving-average FIR filter: each output is the mean of the last N inputs.
// Usage: moving_average input.txt output.txt 8
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: moving_average <input.txt> <output.txt> <N>\n";
        return 1;
    }
    std::ifstream in(argv[1]);
    std::ofstream out(argv[2]);
    int N = std::stoi(argv[3]);

    std::vector<double> x;
    double value;
    while (in >> value) x.push_back(value);

    double running_sum = 0.0;
    for (size_t n = 0; n < x.size(); ++n) {
        running_sum += x[n];                                        // add the newest sample
        if (n >= static_cast<size_t>(N)) running_sum -= x[n - N];   // drop the oldest one
        if (n >= static_cast<size_t>(N - 1)) out << running_sum / N << "\n";
    }
    std::cout << "Filtered " << x.size() << " samples with N = " << N << "\n";
    return 0;
}
