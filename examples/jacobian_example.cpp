#include <chrono>

#include "algodiff/algodiff.hpp"

int main()
{
    std::vector<std::function<algodiff::forward::DualNumber(
        Eigen::VectorX<algodiff::forward::DualNumber>)>>
        f = {
            [](const Eigen::VectorX<algodiff::forward::DualNumber> &vector) {
                return vector[0];
            },
            [](const Eigen::VectorX<algodiff::forward::DualNumber> &vector) {
                return 5.0 * vector[2];
            },
            [](Eigen::VectorX<algodiff::forward::DualNumber> vector) {
                return 4.0 * vector[1] * vector[1] - 2.0 * vector[2];
            },
            [](Eigen::VectorX<algodiff::forward::DualNumber> vector) {
                return vector[2] * algodiff::forward::sin(vector[0]);
            },

        };

    Eigen::Vector3d input = {1.0, 2.0, 3.0};

    auto begin = std::chrono::steady_clock::now();

    auto jacobian = algodiff::forward::jacobian(f, input);
    auto end = std::chrono::steady_clock::now();

    std::cout << "algodiff::forward::jacobian output:\n";
    std::cout << jacobian;

    std::cout << "\n";
    std::cout << "Time taken = "
              << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                       begin)
                     .count()
              << "[µs]" << std::endl;

    std::cout << "\nground truth output:\n";
    Eigen::Matrix<double, 4, 3> ground_truth;
    ground_truth << 1, 0, 0, 0, 0, 5.0, 0, 8.0 * input[1], -2.0,
        input[2] * std::cos(input[0]), 0, std::sin(input[0]);
    std::cout << ground_truth;
    std::cout << "\n";
    return 0;
}
