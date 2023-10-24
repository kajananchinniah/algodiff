/* This file is part of the algodiff project.
 * Copyright (c) 2022 kajananchinniah
 * SPDX-License-Identifier: MIT
 */
#include <cmath>
#include <functional>
#include <iostream>
#include <random>

#include "algodiff/forward_mode.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "algodiff/algodiff.hpp"
#include "algodiff/dual_number.hpp"
#include "algodiff/dual_number_eigen.hpp"
#include "algodiff/dual_number_ops.hpp"

TEST_CASE("Gradient", "[Multidimensional Derivative]")
{
  constexpr std::array<double, 3> expected_output = {
      2.00, -12.5663706144, 2.58689388};
  constexpr std::array<double, 3> input_array = {M_PI, 0.5, 0.9286};
  SECTION("std::vector test")
  {
    auto f = [&](std::vector<algodiff::forward::DualNumber> vector)
    {
      return algodiff::forward::sin(vector[0] / vector[1])
          + algodiff::forward::pow(vector[2], 3.0);
    };

    std::vector<double> input(input_array.begin(), input_array.end());

    auto gradient = algodiff::forward::gradient(f, input);

    REQUIRE(expected_output.size() == gradient.size());
    for (size_t i = 0; i < gradient.size(); ++i) {
      REQUIRE(Catch::Approx(gradient[i]) == expected_output.at(i));
    }
  }

  SECTION("Eigen VectorXd case")
  {
    auto f = [](const Eigen::VectorX<algodiff::forward::DualNumber>& vector)
    {
      return algodiff::forward::sin(vector[0] / vector[1])
          + algodiff::forward::pow(vector[2], 3.0);
    };

    Eigen::VectorXd input(input_array.size());
    for (size_t i = 0; i < input_array.size(); ++i) {
      input[static_cast<int>(i)] = input_array.at(i);
    }
    auto gradient = algodiff::forward::gradient(f, input);
    REQUIRE(gradient.size() == static_cast<int>(expected_output.size()));
    for (int i = 0; i < gradient.size(); ++i) {
      REQUIRE(Catch::Approx(gradient[i])
              == expected_output.at(static_cast<size_t>(i)));
    }
  }

  SECTION("Eigen Vector with specified input size")
  {
    constexpr size_t input_size {3};
    auto f =
        [](Eigen::Vector<algodiff::forward::DualNumber, input_size> vector)
    {
      return algodiff::forward::sin(vector[0] / vector[1])
          + algodiff::forward::pow(vector[2], 3.0);
    };

    Eigen::Vector<double, input_size> input = {
        input_array.at(0), input_array.at(1), input_array.at(2)};
    auto gradient = algodiff::forward::gradient(f, input);
    REQUIRE(static_cast<size_t>(gradient.size()) == expected_output.size());
    for (int i = 0; i < gradient.size(); ++i) {
      REQUIRE(Catch::Approx(gradient[i])
              == expected_output.at(static_cast<size_t>(i)));
    }
  }
}

TEST_CASE("Jacobian", "[Multidimensional Derivative]")
{
  constexpr std::array<double, 2> input_array = {1.25, M_PI / 3};
  const std::vector<std::vector<double>> expected_output = {
      {2.61799387799, 1.5625},
      {5, 0.5},
      {0.877299517946, -0.548312198716},
  };

  SECTION("std::vector function and std::vector input")
  {
    std::vector<algodiff::forward::DualNumber_function> f = {
        [](const std::vector<algodiff::forward::DualNumber>& vector)
        { return vector[0] * vector[0] * vector[1]; },
        [](const std::vector<algodiff::forward::DualNumber>& vector)
        { return 5.0 * vector[0] + algodiff::forward::sin(vector[1]); },
        [](std::vector<algodiff::forward::DualNumber> vector) {
          return vector[0] * vector[0] * algodiff::forward::exp(-vector[1]);
        }};

    std::vector<double> input(input_array.begin(), input_array.end());

    auto jacobian = algodiff::forward::jacobian(f, input);

    REQUIRE(jacobian.size() == f.size());
    for (const auto& jacobian_row : jacobian) {
      REQUIRE(jacobian_row.size() == input.size());
    }

    for (size_t i = 0; i < expected_output.size(); ++i) {
      for (size_t j = 0; j < expected_output[i].size(); ++j) {
        REQUIRE(Catch::Approx(jacobian[i][j]) == expected_output[i][j]);
      }
    }
  }

  SECTION("std::vector function and Eigen VectorXd input")
  {
    std::vector<std::function<algodiff::forward::DualNumber(
        Eigen::VectorX<algodiff::forward::DualNumber>)>>
        f = {[](const Eigen::VectorX<algodiff::forward::DualNumber>& vector)
             { return vector[0] * vector[0] * vector[1]; },
             [](const Eigen::VectorX<algodiff::forward::DualNumber>& vector)
             { return 5.0 * vector[0] + algodiff::forward::sin(vector[1]); },
             [](Eigen::VectorX<algodiff::forward::DualNumber> vector) {
               return vector[0] * vector[0]
                   * algodiff::forward::exp(-vector[1]);
             }};
    Eigen::VectorXd input(input_array.size());
    for (size_t i = 0; i < input_array.size(); ++i) {
      input[static_cast<int>(i)] = input_array.at(i);
    }

    auto jacobian = algodiff::forward::jacobian(f, input);

    REQUIRE(static_cast<size_t>(jacobian.rows()) == f.size());
    REQUIRE(jacobian.cols() == input.size());

    for (size_t i = 0; i < expected_output.size(); ++i) {
      for (size_t j = 0; j < expected_output[i].size(); ++j) {
        REQUIRE(
            Catch::Approx(jacobian(static_cast<int>(i), static_cast<int>(j)))
            == expected_output[i][j]);
      }
    }
  }

  SECTION("Single function with Eigen VectorXd input")
  {
    auto f = [](const Eigen::Ref<
                 const Eigen::VectorX<algodiff::forward::DualNumber>>& vector)
    {
      Eigen::MatrixX<algodiff::forward::DualNumber> mat(3, 3);
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          mat(i, j) = algodiff::forward::DualNumber {0.0, 0.0};
        }
      }
      mat(0, 1) = vector[0] * vector[0];
      mat(1, 0) = algodiff::forward::DualNumber {5.0, 0.0};
      mat(1, 2) = algodiff::forward::sin(vector[1]);
      mat(2, 0) = vector[0] * algodiff::forward::exp(-vector[1]);
      auto new_input =
          Eigen::VectorX<algodiff::forward::DualNumber>(vector.size() + 1);
      for (int i = 0; i < vector.size(); ++i) {
        new_input[i] = vector[i];
      }

      new_input[new_input.size() - 1] =
          algodiff::forward::DualNumber {1.0, 0.0};

      return (mat * new_input).eval();
    };

    Eigen::VectorXd input(input_array.size());
    for (size_t i = 0; i < input_array.size(); ++i) {
      input[static_cast<int>(i)] = input_array.at(i);
    }

    auto jacobian = algodiff::forward::jacobian<3>(f, input);

    REQUIRE(static_cast<size_t>(jacobian.rows()) == 3);
    REQUIRE(jacobian.cols() == input.size());

    for (size_t i = 0; i < expected_output.size(); ++i) {
      for (size_t j = 0; j < expected_output[i].size(); ++j) {
        REQUIRE(
            Catch::Approx(jacobian(static_cast<int>(i), static_cast<int>(j)))
            == expected_output[i][j]);
      }
    }
  }

  SECTION("Single function with fixed Eigen Vector input")
  {
    constexpr size_t input_size = 2;
    auto f =
        [&](Eigen::Vector<algodiff::forward::DualNumber, input_size> vector)
    {
      Eigen::Matrix3<algodiff::forward::DualNumber> mat;
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          mat(i, j) = algodiff::forward::DualNumber {0.0, 0.0};
        }
      }
      mat(0, 1) = vector[0] * vector[0];
      mat(1, 0) = algodiff::forward::DualNumber {5.0, 0.0};
      mat(1, 2) = algodiff::forward::sin(vector[1]);
      mat(2, 0) = vector[0] * algodiff::forward::exp(-vector[1]);
      auto new_input =
          Eigen::Vector<algodiff::forward::DualNumber, input_size + 1> {};
      for (int i = 0; i < vector.size(); ++i) {
        new_input[i] = vector[i];
      }

      new_input[new_input.size() - 1] =
          algodiff::forward::DualNumber {1.0, 0.0};

      return (mat * new_input).eval();
    };

    Eigen::Vector2d input = {input_array.at(0), input_array.at(1)};

    auto jacobian = algodiff::forward::jacobian<3>(f, input);

    REQUIRE(static_cast<size_t>(jacobian.rows()) == 3);
    REQUIRE(jacobian.cols() == input.size());

    for (size_t i = 0; i < expected_output.size(); ++i) {
      for (size_t j = 0; j < expected_output[i].size(); ++j) {
        REQUIRE(
            Catch::Approx(jacobian(static_cast<int>(i), static_cast<int>(j)))
            == expected_output[i][j]);
      }
    }
  }
}
