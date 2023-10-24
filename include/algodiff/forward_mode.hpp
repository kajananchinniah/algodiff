/* This file is part of the algodiff project.
 * Copyright (c) 2022 kajananchinniah
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include <algorithm>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "dual_number.hpp"
#include "dual_number_eigen.hpp"

namespace algodiff::forward
{
namespace internal
{
template <typename T, typename U, int Size>
auto createVector(const Eigen::Matrix<U, Size, 1> &input)
{
    if (Size == Eigen::Dynamic) {
        return Eigen::Matrix<T, Size, 1>(input.size());
    }
    return Eigen::Matrix<T, Size, 1>();
}

} // namespace internal

/**
 * \brief Returns the resultant DualNumber when a function f is evaluated at u.
 * The primal component is the function evaluated at u and the dual component is
 * the function's derivative at u
 *
 * \tparam F Function Type that takes as input a single DualNumber and outputs
 * a DualNumber
 * \param f A single dimension function
 * \param u The point to evaluate f at
 * \return The DualNumber of f evaluated at u
 */
template <class F> auto evaluate(F &&f, double u) -> DualNumber
{
    return f(DualNumber(u, 1.0));
}

/**
 * \brief Returns the derivative of f evaluated at u.
 *
 * \tparam F Function Type that takes as input a single DualNumber and outputs
 * a DualNumber
 * \param f A single dimension function
 * \param u The point to evaluate the derivative of f at
 * \return The derivative of f computed at u
 */
template <class F> auto derivative(F &&f, double u) -> double
{
    return evaluate(std::forward<F>(f), u).dual();
}

/**
 * \brief Returns a vector of DualNumbers representing the function f evaluated
 * at u. The primal component is the function evaluated at u and the dual
 * component is the function's derivative at u
 *
 * \tparam F Function Type that takes as input a std::vector of DualNumbers and
 * outputs a DualNumber
 * \param f A function that maps u (in DualNumber representation) to the output
 * space
 * \param u A vector of inputs that f will be evaluated at
 * \return A vector of DualNumbers representing f evaluated at u
 */
template <class F>
auto evaluate(F &&f, const std::vector<double> &u) -> std::vector<DualNumber>
{
    std::vector<DualNumber> dual_numbers{};
    std::transform(u.cbegin(), u.cend(), std::back_inserter(dual_numbers),
                   [](double x) {
                       return DualNumber{x, 0.0};
                   });

    std::vector<DualNumber> evaluations{};
    std::for_each(dual_numbers.begin(), dual_numbers.end(),
                  [&](DualNumber &num) {
                      num.dual() = 1.0;
                      evaluations.push_back(f(dual_numbers));
                      num.dual() = 0.0;
                  });
    return evaluations;
}

/**
 * \brief Returns the gradient of f evaluated at u
 *
 * \tparam F Function Type that takes as input a std::vector of DualNumber and
 * outputs a DualNumber
 * \param f A function that maps u (in DualNumber representation) to the output
 * space
 * \param u A vector of inputs that f will be evaluated at
 * \return The gradient of f computed at u
 */
template <class F>
auto gradient(F &&f, const std::vector<double> &u) -> std::vector<double>
{
    const std::vector<DualNumber> evaluations{evaluate(std::forward<F>(f), u)};
    std::vector<double> grad{};
    std::transform(evaluations.cbegin(), evaluations.cend(),
                   std::back_inserter(grad),
                   [](const DualNumber &num) { return num.dual(); });
    return grad;
}

/**
 * \brief Returns a vector of DualNumbers representing the function f evaluated
 * at u. The primal component is the function evaluated at u and the dual
 * component is the function's derivative at u
 *
 * \tparam F Function Type that takes as input a Eigen::Matrix<DualNumber,
 * InputSize, 1> and outputs a DualNumber
 * \tparam InputSize The dimension of the input vector
 * \param f A function that maps u (in DualNumber representation) to the output
 * space
 * \param u A vector of inputs that f will be evaluated at
 * \return A DualNumber vector representing f evaluated at u
 */
template <class F, int InputSize>
auto evaluate(F &&f, const Eigen::Matrix<double, InputSize, 1> &u)
    -> Eigen::Matrix<DualNumber, InputSize, 1>
{
    Eigen::Matrix<DualNumber, InputSize, 1> dual_numbers{
        internal::createVector<DualNumber>(u)};
    std::transform(u.data(), u.data() + u.size(), dual_numbers.data(),
                   [&](double x) {
                       return DualNumber{x, 0.0};
                   });
    Eigen::Matrix<DualNumber, InputSize, 1> evaluations{
        internal::createVector<DualNumber>(u)};
    for (int i = 0; i < dual_numbers.size(); ++i) {
        dual_numbers[i].dual() = 1.0;
        evaluations[i] = f(dual_numbers);
        dual_numbers[i].dual() = 0.0;
    }

    return evaluations;
}

/**
 * \brief Returns the gradient of f evaluated at u
 *
 * \tparam F Function Type that takes as input a Eigen::Matrix<DualNumber,
 * InputSize, 1> and outputs a DualNumber
 * \tparam InputSize The dimension of the input vector
 * \param f A function that maps u (in dual number representation) to the output
 * space
 * \param u A vector of inputs that f will be evaluated at
 * \return A DualNumber vector representing the gradient of f evaluated at u
 */
template <class F, int InputSize>
auto gradient(F &&f, const Eigen::Matrix<double, InputSize, 1> &u)
{
    Eigen::Matrix<DualNumber, InputSize, 1> evaluations{
        evaluate(std::forward<F>(f), u)};
    Eigen::Matrix<double, InputSize, 1> grad{
        internal::createVector<double>(evaluations)};
    std::transform(evaluations.data(), evaluations.data() + evaluations.size(),
                   grad.data(),
                   [](const DualNumber &num) { return num.dual(); });
    return grad;
}

/**
 * \brief Returns the jacobian of f evaluated at u
 *
 * \tparam F Function Type that takes as input a std::vector<DualNumber> and
 * outputs a DualNumber
 * \param f A set of functions that map u (in dual number representation) to the
 * output space
 * \param u A vector of inputs that each element of f will be evaluated at
 * \return A matrix representing the jacobian of f at u
 */
template <class F>
auto jacobian(const std::vector<F> &f, const std::vector<double> &u)
{
    std::vector<std::vector<double>> jac{};
    std::transform(f.cbegin(), f.cend(), std::back_inserter(jac),
                   [&](const F &func) { return gradient(func, u); });
    return jac;
}

/**
 * \brief Returns the jacobian of f evaluated at u
 *
 * \tparam F Function Type that takes as input a Eigen::VectorX<DualNumber> and
 * outputs a DualNumber
 * \param f A set of functions that map u (in dual number representation) to the
 * output space
 * \param u A vector of inputs that each element of f will be evaluated at
 * \return A matrix representing the jacobian of f at u
 */
template <class F>
auto jacobian(const std::vector<F> &f, const Eigen::VectorXd &u)
    -> Eigen::MatrixXd
{
    Eigen::MatrixXd jacobian(f.size(), u.size());
    for (int i = 0; i < jacobian.rows(); ++i) {
        jacobian.row(i) = gradient(f[static_cast<size_t>(i)], u);
    }
    return jacobian;
}

// TODO(kajananchinniah): consolidate the functions into one

/**
 * \brief Returns the jacobian of f evaluated at u
 *
 * \tparam F Function Type that takes as input a Eigen::VectorX<DualNumber> and
 * outputs a vector of DualNumbers (type: Eigen::VectorX<DualNumber>)
 * \param f A multidimensional function that maps u (in dual number
 * representation) to the output space
 * \param u A vector of inputs that each element of f will be evaluated at
 * \return A matrix representing the jacobian of f at u
 */
template <int FunctionSize, class F>
auto jacobian(F &&f, const Eigen::VectorXd &u) -> Eigen::MatrixXd
{
    Eigen::VectorX<DualNumber> dual_numbers(u.size());
    for (int i = 0; i < u.size(); ++i) {
        dual_numbers[i] = DualNumber{u[i], 0.0};
    }

    Eigen::MatrixXd jac(FunctionSize, u.size());
    for (int i = 0; i < jac.cols(); ++i) {
        dual_numbers[i].dual() = 1.0;
        Eigen::VectorX<DualNumber> result{f(dual_numbers)};
        for (int j = 0; j < FunctionSize; ++j) {
            jac.col(i)[j] = result[j].dual();
        }
        dual_numbers[i].dual() = 0.0;
    }
    return jac;
}

/**
 * \brief Returns the jacobian of f evaluated at u
 *
 * \warning f MUST output a vector of size FunctionSize
 *
 * \tparam F Function Type that takes as input a Eigen::VectorX<DualNumber,
 * InputSize> and outputs a vector of DualNumbers (type:
 * Eigen::VectorX<DualNumber, FunctionSize>)
 * \param f A multidimensional function that maps u (in dual number
 * representation) to the output space
 * \param u A vector of inputs that each element of f will be evaluated at
 * \return A matrix representing the jacobian of f at u
 */
template <int FunctionSize, class F, int InputSize>
auto jacobian(F &&f, const Eigen::Vector<double, InputSize> &u)
{
    Eigen::Vector<DualNumber, InputSize> dual_numbers{};
    for (int i = 0; i < InputSize; ++i) {
        dual_numbers(i) = DualNumber{u[i], 0.0};
    }

    Eigen::Matrix<double, FunctionSize, InputSize> jac;
    for (int i = 0; i < InputSize; ++i) {
        dual_numbers[i].dual() = 1.0;
        auto result = f(dual_numbers);
        for (int j = 0; j < FunctionSize; ++j) {
            jac.col(i)[j] = result[j].dual();
        }

        dual_numbers[i].dual() = 0.0;
    }
    return jac;
}

/// Convenience type alias
using DualNumber_function = std::function<algodiff::forward::DualNumber(
    std::vector<algodiff::forward::DualNumber>)>;

} // namespace algodiff::forward
