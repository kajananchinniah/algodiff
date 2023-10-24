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

TEST_CASE("Power Functions", "[One Dimensional Function]")
{
  SECTION("single polynomial")
  {
    auto f = [](algodiff::forward::DualNumber num)
        -> algodiff::forward::DualNumber
    { return algodiff::forward::pow(num, 3.0); };

    auto result = f(algodiff::forward::DualNumber {2.5, 1.0});
    REQUIRE(Catch::Approx(result.primal()) == 15.625);
  }

  SECTION("add two polynomials")
  {
    auto f = [](algodiff::forward::DualNumber num)
        -> algodiff::forward::DualNumber
    {
      return algodiff::forward::pow(num, 3.0)
          + algodiff::forward::pow(num, 4.0);
    };
    auto result = f(algodiff::forward::DualNumber {1.234, 1.0});
    REQUIRE(Catch::Approx(result.primal()) == 4.19786673954);
  }

  SECTION("multiply two polynomials")
  {
    auto f = [](algodiff::forward::DualNumber num)
        -> algodiff::forward::DualNumber
    {
      return algodiff::forward::pow(num, 3.0)
          * algodiff::forward::pow(num, 4.0);
    };
    auto result = f(algodiff::forward::DualNumber {0.582, 1.0});
    REQUIRE(Catch::Approx(result.primal()) == 0.0226183485637);
  }

  SECTION("sqrt function")
  {
    auto f = [](algodiff::forward::DualNumber num)
        -> algodiff::forward::DualNumber
    { return algodiff::forward::sqrt(num); };

    auto result = f(algodiff::forward::DualNumber {10.1265, 1.0});
    REQUIRE(Catch::Approx(result.primal()) == 3.18221620887);
  }

  SECTION("add rational powers")
  {
    auto f = [](algodiff::forward::DualNumber num)
        -> algodiff::forward::DualNumber
    {
      return algodiff::forward::pow(num, 2.0 / 3.0)
          + algodiff::forward::pow(num, 2.0);
    };

    auto result = f(algodiff::forward::DualNumber {9876.653, 1.0});
    REQUIRE(Catch::Approx(result.primal()) == 97548734.8166);
  }

  SECTION("rational function")
  {
    auto f = [](algodiff::forward::DualNumber num)
        -> algodiff::forward::DualNumber
    {
      const auto numerator = algodiff::forward::pow(num, 2.0 / 5.0)
              * algodiff::forward::pow(num - 1.0, 2.0)
          + algodiff::forward::pow(num + 2.0, 3.0);
      const auto denominator = algodiff::forward::pow(num, 3.0)
          + (9.0 / 8.0) * algodiff::forward::pow(num, 2.0)
          + algodiff::forward::pow(2.0 * num, 1.0) + 0.5;
      return numerator / denominator;
    };

    auto result = f(algodiff::forward::DualNumber {0.301, 1.0});
    REQUIRE(Catch::Approx(result.primal()) == 10.1406491621);
  }
}

TEST_CASE("Exponential Function", "[One Dimensional Function]")
{
  SECTION("exponential function")
  {
    auto f = [](algodiff::forward::DualNumber num)
        -> algodiff::forward::DualNumber
    { return algodiff::forward::exp(num / 2.0); };

    auto result = f(algodiff::forward::DualNumber {3.124, 1.0});
    REQUIRE(Catch::Approx(result.primal()) == 4.76834841162);
  }

  SECTION("exponential function multiplied by rational function")
  {
    auto f = [](algodiff::forward::DualNumber num)
        -> algodiff::forward::DualNumber
    {
      return algodiff::forward::exp(num / 4.0)
          * algodiff::forward::pow(num - 10.0, 2.0);
    };

    auto result = f(algodiff::forward::DualNumber {7.656, 1.0});
    REQUIRE(Catch::Approx(result.primal()) == 37.2524510786);
  }

  SECTION("testing exp2")
  {
    auto f = [](algodiff::forward::DualNumber num)
        -> algodiff::forward::DualNumber
    { return algodiff::forward::exp2((num - 10.0) / 7.0); };

    auto result = f(algodiff::forward::DualNumber {31.0, 1.0});
    REQUIRE(Catch::Approx(result.primal()) == 8.00);
  }

  SECTION("dual number to the power dual number")
  {
    auto f = [](algodiff::forward::DualNumber num)
        -> algodiff::forward::DualNumber
    { return algodiff::forward::pow(num, num / 2.0); };

    auto result = f(algodiff::forward::DualNumber {4.123, 1.0});
    REQUIRE(Catch::Approx(result.primal()) == 18.5465139145);
  }
}

TEST_CASE("Logarithm Function", "[One Dimensional Function]")
{
  SECTION("natural logarithm function")
  {
    auto f = [](algodiff::forward::DualNumber num)
        -> algodiff::forward::DualNumber
    { return algodiff::forward::log(num / (num + 1)); };

    auto result = f(algodiff::forward::DualNumber {987.123, 1.0});
    REQUIRE(Catch::Approx(result.primal()) == -0.00101253219643);
  }

  SECTION("logarithm base 2 function")
  {
    auto f = [](algodiff::forward::DualNumber num)
        -> algodiff::forward::DualNumber
    { return algodiff::forward::log2(num) * algodiff::forward::log(num); };

    auto result = f(algodiff::forward::DualNumber {2.0, 1.0});
    REQUIRE(Catch::Approx(result.primal()) == 0.69314718056);
  }

  SECTION("logarithm base 10 function")
  {
    auto f = [](algodiff::forward::DualNumber num)
        -> algodiff::forward::DualNumber
    { return (1 + 5 * num) / algodiff::forward::log10(num); };

    auto result = f(algodiff::forward::DualNumber {104.5, 1.0});
    REQUIRE(Catch::Approx(result.primal()) == 259.271842081);
  }

  SECTION("logarithm base function")
  {
    auto f = [](algodiff::forward::DualNumber num)
        -> algodiff::forward::DualNumber
    {
      return 4 * algodiff::forward::log(2.0 * (num - 5.0), 3.0)
          - algodiff::forward::log(num);
    };

    auto result = f(algodiff::forward::DualNumber {6.0, 1.0});
    REQUIRE(Catch::Approx(result.primal()) == 0.731959545058);
  }
}

TEST_CASE("Trigonometric Function", "[One Dimensional Function]")
{
  SECTION("sine function")
  {
    auto f = [](algodiff::forward::DualNumber num)
        -> algodiff::forward::DualNumber
    { return algodiff::forward::sin(2 * num); };

    auto result = f(algodiff::forward::DualNumber {M_PI / 2.0, 1.0});
    // Workaround to compare with zero
    REQUIRE(result.primal() < 1e-12);
    REQUIRE(result.primal() >= 0.00);
  }

  SECTION("cosine function")
  {
    auto f = [](algodiff::forward::DualNumber num)
        -> algodiff::forward::DualNumber
    {
      return algodiff::forward::cos(
          algodiff::forward::pow(num + 1, 1.0 / 2.0));
    };

    auto result = f(algodiff::forward::DualNumber {M_PI, 1.0});
    REQUIRE(Catch::Approx(result.primal()) == -0.447791655095);
  }

  SECTION("tan function")
  {
    auto f = [](algodiff::forward::DualNumber num)
        -> algodiff::forward::DualNumber
    {
      return algodiff::forward::pow(
          algodiff::forward::tan(algodiff::forward::sin(num)), 2.0);
    };

    auto result = f(algodiff::forward::DualNumber {1.111, 1.0});
    REQUIRE(Catch::Approx(result.primal()) == 1.5630607527);
  }
}

TEST_CASE("Inverse Trigonmetric Function", "[One Dimensional Function]")
{
  SECTION("inverse sine function")
  {
    auto f = [](algodiff::forward::DualNumber num)
        -> algodiff::forward::DualNumber
    {
      return algodiff::forward::exp(algodiff::forward::asin(num)
                                     * algodiff::forward::log(num));
    };

    auto result = f(algodiff::forward::DualNumber {0.99999, 1.0});
    REQUIRE(Catch::Approx(result.primal()) == 0.999984336802);
  }

  SECTION("inverse cosine function")
  {
    auto f = [](algodiff::forward::DualNumber num)
        -> algodiff::forward::DualNumber
    { return algodiff::forward::acos(algodiff::forward::pow(num, 0.5)); };

    auto result = f(algodiff::forward::DualNumber {0.5, 1.0});
    REQUIRE(Catch::Approx(result.primal()) == 0.785398163397);
  }

  SECTION("inverse tan function")
  {
    auto f = [](algodiff::forward::DualNumber num)
        -> algodiff::forward::DualNumber
    { return algodiff::forward::atan(algodiff::forward::exp(num)); };

    auto result = f(algodiff::forward::DualNumber {6.0, 1.0});
    REQUIRE(Catch::Approx(result.primal()) == 1.56831757969);
  }
}

TEST_CASE("Hyperbolic Function", "[One Dimensional Function]")
{
  SECTION("hyperbolic sine function")
  {
    auto f = [](algodiff::forward::DualNumber num)
        -> algodiff::forward::DualNumber
    { return algodiff::forward::sinh(2 * num); };

    auto result = f(algodiff::forward::DualNumber {M_PI / 2.0, 1.0});
    REQUIRE(Catch::Approx(result.primal()) == 11.5487393573);
  }

  SECTION("hyperbolic cosine function")
  {
    auto f = [](algodiff::forward::DualNumber num)
        -> algodiff::forward::DualNumber
    {
      return algodiff::forward::cosh(
          algodiff::forward::pow(num + 1, 1.0 / 2.0));
    };

    auto result = f(algodiff::forward::DualNumber {M_PI, 1.0});
    REQUIRE(Catch::Approx(result.primal()) == 3.8918060338);
  }

  SECTION("hyperbolic tan function")
  {
    auto f = [](algodiff::forward::DualNumber num)
        -> algodiff::forward::DualNumber
    {
      return algodiff::forward::pow(
          algodiff::forward::tanh(algodiff::forward::sin(num)), 2.0);
    };

    auto result = f(algodiff::forward::DualNumber {1.111, 1.0});
    REQUIRE(Catch::Approx(result.primal()) == 0.510388202167);
  }
}

TEST_CASE("Inverse Hyperbolic Trigonmetric Function",
          "[One Dimensional Function]")
{
  SECTION("inverse hyperbolic sine function")
  {
    auto f = [](algodiff::forward::DualNumber num)
        -> algodiff::forward::DualNumber
    {
      return algodiff::forward::exp(algodiff::forward::asinh(num)
                                     * algodiff::forward::log(num));
    };

    auto result = f(algodiff::forward::DualNumber {0.99999, 1.0});
    REQUIRE(Catch::Approx(result.primal()) == 0.99999118633);
  }

  SECTION("inverse hyperbolic cosine function")
  {
    auto f = [](algodiff::forward::DualNumber num)
        -> algodiff::forward::DualNumber
    { return algodiff::forward::acosh(algodiff::forward::pow(num, 0.5)); };

    auto result = f(algodiff::forward::DualNumber {1.5, 1.0});
    REQUIRE(Catch::Approx(result.primal()) == 0.658478948462);
  }

  SECTION("inverse hyperbolic tan function")
  {
    auto f = [](algodiff::forward::DualNumber num)
        -> algodiff::forward::DualNumber
    { return algodiff::forward::atanh(algodiff::forward::exp(num)); };

    auto result = f(algodiff::forward::DualNumber {-0.35, 1.0});
    REQUIRE(Catch::Approx(result.primal()) == 0.876552656823);
  }
}

TEST_CASE("Non-Member Function Function", "[One Dimensional Function]")
{
  SECTION("Absolute function")
  {
    auto f = [](algodiff::forward::DualNumber num)
        -> algodiff::forward::DualNumber
    { return algodiff::forward::abs(algodiff::forward::sin(num) / 4.0); };

    auto result = f(algodiff::forward::DualNumber {2.00, 1.0});
    REQUIRE(Catch::Approx(result.primal()) == 0.227324356706);
  }

  SECTION("Inverse function")
  {
    auto f = [](algodiff::forward::DualNumber num)
        -> algodiff::forward::DualNumber
    { return algodiff::forward::inverse(2.0 * num); };

    auto result = f(algodiff::forward::DualNumber {-2.00, 1.0});
    REQUIRE(Catch::Approx(result.primal()) == -0.25);
  }

  SECTION("Inverse function via division operator")
  {
    auto f = [](algodiff::forward::DualNumber num)
        -> algodiff::forward::DualNumber { return 1.0 / (2.0 * num); };

    auto result = f(algodiff::forward::DualNumber {-2.00, 1.0});
    REQUIRE(Catch::Approx(result.primal()) == -0.25);
  }
}
