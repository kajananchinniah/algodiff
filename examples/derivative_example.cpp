#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>

#include <algodiff/algodiff.hpp>

auto main() -> int
{
    // Let's say we wish to evaluate all functions at x = 0.25
    constexpr double x{0.25};

    // Precision to print
    constexpr int precision{15};

    // Define a function that maps an input DualNumber to a desired output
    // space In this example, we use f(x) = (1 + x) / (2 - x)^2 + 2*cos(-3x)
    auto f = [](const algodiff::forward::DualNumber &u)
        -> algodiff::forward::DualNumber {
        const auto numerator = 1.0 + u;
        const auto denominator = algodiff::forward::pow((2.0 - u), 2.0);
        const auto term1 = numerator / denominator;
        const auto term2 = 2.0 * algodiff::forward::cos(-3.0 * u);
        return term1 + term2;
    };

    auto begin = std::chrono::steady_clock::now();
    // Evaluate the derivative of f evaluated at 0.25 using the derivative
    // function
    const double ad_derivative = algodiff::forward::derivative(f, x);

    auto end = std::chrono::steady_clock::now();

    // Print out the results
    std::cout << "algodiff::forward::derivative of f at " << x << " = "
              << std::setprecision(precision) << ad_derivative << "\n";
    std::cout << "Time taken = "
              << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                       begin)
                     .count()
              << "[µs]" << std::endl;

    // Analytically, the expression we get when taking this derivative is:
    // f'(x) = (4 + x) / (2 - x)^3 + 6*sin(-3x)
    auto f_prime = [](double u) -> double {
        const auto numerator = 4.0 + u;
        const auto denominator = std::pow(2.0 - u, 3.0);
        const auto term1 = numerator / denominator;
        const auto term2 = 6.0 * std::sin(-3.0 * u);
        return term1 + term2;
    };

    begin = std::chrono::steady_clock::now();
    const double exact_derivative{f_prime(x)};
    end = std::chrono::steady_clock::now();
    std::cout << "Exact derivative of f at " << x << " = "
              << std::setprecision(precision) << exact_derivative << "\n";
    std::cout << "Time taken = "
              << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                       begin)
                     .count()
              << "[µs]" << std::endl;
    return 0;
}
