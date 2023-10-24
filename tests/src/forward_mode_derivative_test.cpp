/* This file is part of the algodiff project.
 * Copyright (c) 2023 kajananchinniah
 * SPDX-License-Identifier: MIT
 */
#include <cmath>
#include <functional>
#include <random>

#include "algodiff/forward_mode.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "algodiff/algodiff.hpp"
#include "algodiff/dual_number.hpp"
#include "algodiff/dual_number_eigen.hpp"
#include "algodiff/dual_number_ops.hpp"

TEST_CASE("Power Derivatives", "[One Dimensional Derivative]")
{
    SECTION("single polynomial")
    {
        auto f = [](algodiff::forward::DualNumber num)
            -> algodiff::forward::DualNumber {
            return algodiff::forward::pow(num, 3.0);
        };

        REQUIRE(Catch::Approx(algodiff::forward::derivative(f, 2.5)) == 18.75);
    }

    SECTION("add two polynomials")
    {
        auto f = [](algodiff::forward::DualNumber num)
            -> algodiff::forward::DualNumber {
            return algodiff::forward::pow(num, 3.0) +
                   algodiff::forward::pow(num, 4.0);
        };

        REQUIRE(Catch::Approx(algodiff::forward::derivative(f, 1.234)) ==
                12.084591616);
    }

    SECTION("multiply two polynomials")
    {
        auto f = [](algodiff::forward::DualNumber num)
            -> algodiff::forward::DualNumber {
            return algodiff::forward::pow(num, 3.0) *
                   algodiff::forward::pow(num, 4.0);
        };

        REQUIRE(Catch::Approx(algodiff::forward::derivative(f, 0.582)) ==
                0.272041993034);
    }

    SECTION("sqrt function")
    {
        auto f = [](algodiff::forward::DualNumber num)
            -> algodiff::forward::DualNumber {
            return algodiff::forward::sqrt(num);
        };

        REQUIRE(Catch::Approx(algodiff::forward::derivative(f, 10.1265)) ==
                0.157123201939);
    }

    SECTION("add rational powers")
    {
        auto f = [](algodiff::forward::DualNumber num)
            -> algodiff::forward::DualNumber {
            return algodiff::forward::pow(num, 2.0 / 3.0) +
                   algodiff::forward::pow(num, 2.0);
        };
        REQUIRE(Catch::Approx(algodiff::forward::derivative(f, 9876.653)) ==
                19753.3110311);
    }

    SECTION("rational function")
    {
        auto f = [](algodiff::forward::DualNumber num)
            -> algodiff::forward::DualNumber {
            const auto numerator = algodiff::forward::pow(num, 2.0 / 5.0) *
                                       algodiff::forward::pow(num - 1.0, 2.0) +
                                   algodiff::forward::pow(num + 2.0, 3.0);
            const auto denominator =
                algodiff::forward::pow(num, 3.0) +
                (9.0 / 8.0) * algodiff::forward::pow(num, 2.0) +
                algodiff::forward::pow(2.0 * num, 1.0) + 0.5;
            return numerator / denominator;
        };

        REQUIRE(Catch::Approx(algodiff::forward::derivative(f, 0.301)) ==
                -11.7647053055);
    }
}

TEST_CASE("Exponential Derivative", "[One Dimensional Derivative]")
{
    SECTION("exponential function")
    {
        auto f = [](algodiff::forward::DualNumber num)
            -> algodiff::forward::DualNumber {
            return algodiff::forward::exp(num / 2.0);
        };

        REQUIRE(Catch::Approx(algodiff::forward::derivative(f, 3.124)) ==
                2.38417420581);
    }

    SECTION("exponential function multiplied by rational function")
    {
        auto f = [](algodiff::forward::DualNumber num)
            -> algodiff::forward::DualNumber {
            return algodiff::forward::exp(num / 4.0) *
                   algodiff::forward::pow(num - 10.0, 2.0);
        };

        REQUIRE(Catch::Approx(algodiff::forward::derivative(f, 7.656)) ==
                -22.472);
    }

    SECTION("testing exp2")
    {
        auto f = [](algodiff::forward::DualNumber num)
            -> algodiff::forward::DualNumber {
            return algodiff::forward::exp2((num - 10.0) / 7.0);
        };

        REQUIRE(Catch::Approx(algodiff::forward::derivative(f, 31.0)) ==
                0.792168206354);
    }

    SECTION("dual number to the power dual number")
    {
        auto f = [](algodiff::forward::DualNumber num)
            -> algodiff::forward::DualNumber {
            return algodiff::forward::pow(num, num / 2.0);
        };

        REQUIRE(Catch::Approx(algodiff::forward::derivative(f, 4.123)) ==
                22.4095770692);
    }
}

TEST_CASE("Logarithm Derivative", "[One Dimensional Derivative]")
{
    SECTION("natural logarithm function")
    {
        auto f = [](algodiff::forward::DualNumber num)
            -> algodiff::forward::DualNumber {
            return algodiff::forward::log(num / (num + 1));
        };

        REQUIRE(Catch::Approx(algodiff::forward::derivative(f, 987.123)) ==
                0.0000010252215364);
    }

    SECTION("logarithm base 2 function")
    {
        auto f = [](algodiff::forward::DualNumber num)
            -> algodiff::forward::DualNumber {
            return algodiff::forward::log2(num) * algodiff::forward::log(num);
        };

        REQUIRE(Catch::Approx(algodiff::forward::derivative(f, 2.0)) == 1.0);
    }

    SECTION("logarithm base 10 function")
    {
        auto f = [](algodiff::forward::DualNumber num)
            -> algodiff::forward::DualNumber {
            return (1 + 5 * num) / algodiff::forward::log10(num);
        };

        REQUIRE(Catch::Approx(algodiff::forward::derivative(f, 104.5)) ==
                1.94267407766);
    }

    SECTION("logarithm base function")
    {
        auto f = [](algodiff::forward::DualNumber num)
            -> algodiff::forward::DualNumber {
            return 4 * algodiff::forward::log(2.0 * (num - 5.0), 3.0) -
                   algodiff::forward::log(num);
        };

        REQUIRE(Catch::Approx(algodiff::forward::derivative(f, 6)) ==
                3.47429023984);
    }
}

TEST_CASE("Trigonometric Derivative", "[One Dimensional Derivative]")
{
    SECTION("sine function")
    {
        auto f = [](algodiff::forward::DualNumber num)
            -> algodiff::forward::DualNumber {
            return algodiff::forward::sin(2 * num);
        };

        REQUIRE(Catch::Approx(algodiff::forward::derivative(f, M_PI / 2.0)) ==
                -2);
    }

    SECTION("cosine function")
    {
        auto f = [](algodiff::forward::DualNumber num)
            -> algodiff::forward::DualNumber {
            return algodiff::forward::cos(
                algodiff::forward::pow(num + 1, 1.0 / 2.0));
        };
        REQUIRE(Catch::Approx(algodiff::forward::derivative(f, M_PI)) ==
                -0.219680157239);
    }

    SECTION("tan function")
    {
        auto f = [](algodiff::forward::DualNumber num)
            -> algodiff::forward::DualNumber {
            return algodiff::forward::pow(
                algodiff::forward::tan(algodiff::forward::sin(num)), 2.0);
        };

        REQUIRE(Catch::Approx(algodiff::forward::derivative(f, 1.111)) ==
                2.84400564897);
    }
}

TEST_CASE("Inverse Trigonmetric Derivative", "[One Dimensional Derivative]")
{
    SECTION("inverse sine function")
    {
        auto f = [](algodiff::forward::DualNumber num)
            -> algodiff::forward::DualNumber {
            return algodiff::forward::exp(algodiff::forward::asin(num) *
                                          algodiff::forward::log(num));
        };

        REQUIRE(Catch::Approx(algodiff::forward::derivative(f, 0.99999)) ==
                1.5640792669);
    }

    SECTION("inverse cosine function")
    {
        auto f = [](algodiff::forward::DualNumber num)
            -> algodiff::forward::DualNumber {
            return algodiff::forward::acos(algodiff::forward::pow(num, 0.5));
        };
        REQUIRE(Catch::Approx(algodiff::forward::derivative(f, 0.5)) == -1);
    }

    SECTION("inverse tan function")
    {
        auto f = [](algodiff::forward::DualNumber num)
            -> algodiff::forward::DualNumber {
            return algodiff::forward::atan(algodiff::forward::exp(num));
        };

        REQUIRE(Catch::Approx(algodiff::forward::derivative(f, 6.0)) ==
                0.00247873694678);
    }
}

TEST_CASE("Hyperbolic Derivative", "[One Dimensional Derivative]")
{
    SECTION("hyperbolic sine function")
    {
        auto f = [](algodiff::forward::DualNumber num)
            -> algodiff::forward::DualNumber {
            return algodiff::forward::sinh(2 * num);
        };

        REQUIRE(Catch::Approx(algodiff::forward::derivative(f, M_PI / 2.0)) ==
                23.183906551);
    }

    SECTION("hyperbolic cosine function")
    {
        auto f = [](algodiff::forward::DualNumber num)
            -> algodiff::forward::DualNumber {
            return algodiff::forward::cosh(
                algodiff::forward::pow(num + 1, 1.0 / 2.0));
        };
        REQUIRE(Catch::Approx(algodiff::forward::derivative(f, M_PI)) ==
                0.924071354158);
    }

    SECTION("hyperbolic tan function")
    {
        auto f = [](algodiff::forward::DualNumber num)
            -> algodiff::forward::DualNumber {
            return algodiff::forward::pow(
                algodiff::forward::tanh(algodiff::forward::sin(num)), 2.0);
        };

        REQUIRE(Catch::Approx(algodiff::forward::derivative(f, 1.111)) ==
                0.310445817122);
    }
}

TEST_CASE("Inverse Hyperbolic Trigonmetric Derivative",
          "[One Dimensional Derivative]")
{
    SECTION("inverse hyperbolic sine function")
    {
        auto f = [](algodiff::forward::DualNumber num)
            -> algodiff::forward::DualNumber {
            return algodiff::forward::exp(algodiff::forward::asinh(num) *
                                          algodiff::forward::log(num));
        };

        REQUIRE(Catch::Approx(algodiff::forward::derivative(f, 0.99999)) ==
                0.88136049046);
    }

    SECTION("inverse hyperbolic cosine function")
    {
        auto f = [](algodiff::forward::DualNumber num)
            -> algodiff::forward::DualNumber {
            return algodiff::forward::acosh(algodiff::forward::pow(num, 0.5));
        };
        REQUIRE(Catch::Approx(algodiff::forward::derivative(f, 1.5)) ==
                0.57735026919);
    }

    SECTION("inverse hyperbolic tan function")
    {
        auto f = [](algodiff::forward::DualNumber num)
            -> algodiff::forward::DualNumber {
            return algodiff::forward::atanh(algodiff::forward::exp(num));
        };

        REQUIRE(Catch::Approx(algodiff::forward::derivative(f, -0.35)) ==
                1.39981628472);
    }
}

TEST_CASE("Non-Member Function Derivative", "[One Dimensional Derivative]")
{
    SECTION("Absolute function")
    {
        auto f = [](algodiff::forward::DualNumber num)
            -> algodiff::forward::DualNumber {
            return algodiff::forward::abs(algodiff::forward::sin(num) / 4.0);
        };

        REQUIRE(Catch::Approx(algodiff::forward::derivative(f, 2)) ==
                -0.104036709137);
    }

    SECTION("Inverse function")
    {
        auto f = [](algodiff::forward::DualNumber num)
            -> algodiff::forward::DualNumber {
            return algodiff::forward::inverse(2.0 * num);
        };

        REQUIRE(Catch::Approx(algodiff::forward::derivative(f, -2.0)) ==
                -0.125);
    }

    SECTION("Inverse function via division operator")
    {
        auto f = [](algodiff::forward::DualNumber num)
            -> algodiff::forward::DualNumber { return 1.0 / (2.0 * num); };

        REQUIRE(Catch::Approx(algodiff::forward::derivative(f, -2.0)) ==
                -0.125);
    }
}
