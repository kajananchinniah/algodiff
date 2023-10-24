/* This file is part of the algodiff project.
 * Copyright (c) 2023 kajananchinniah
 * SPDX-License-Identifier: MIT
 */
/// \file dual_number_ops.hpp
/// \brief Implements operations that can be performed on dual numbers
#pragma once

#include "dual_number.hpp"

namespace algodiff::forward
{
// Non-member functions
/**
 * \brief Returns the primal component of a DualNumber
 *
 * \param num The DualNumber
 * \return The primal component of num
 */
constexpr auto primal(const DualNumber &num) -> double
{
    return num.primal();
}

/**
 * \brief Returns the primal component of a DualNumber. This function can be
 * useful with Eigen
 *
 * \param num The DualNumber
 * \return The primal component of num
 */
constexpr auto real(const DualNumber &num) -> double
{
    return primal(num);
}

/**
 * \brief Returns the dual component of a DualNumber
 *
 * \param num The DualNumber
 * \return The dual component of num
 */
constexpr auto dual(const DualNumber &num) -> double
{
    return num.dual();
}

/**
 * \brief Returns the dual component of a DualNumber. This function can be
 * useful with Eigen
 *
 * \param num The DualNumber
 * \return The dual component of num
 */
constexpr auto imag(const DualNumber &num) -> double
{
    return dual(num);
}

/**
 * \brief Returns the absolute value of a DualNumber
 *
 * \warning This is not the magnitude, but the absolute value of the primal
 *          component
 *
 * \param num The DualNumber
 * \return The absolute value of the DualNumber
 */
auto abs(const DualNumber &num) -> DualNumber;

/**
 * \brief Computes the inverse of a DualNumber
 *
 * \param num The DualNumber
 * \return The inverse of the DualNumber
 */
auto inverse(const DualNumber &num) -> DualNumber;

/**
 * \brief Returns the conjugate of a DualNumber
 *
 * \param num The DualNumber
 * \return The conjugate of the DualNumber
 */
constexpr auto conj(const DualNumber &num) -> DualNumber
{
    return DualNumber{num.primal(), -num.dual()};
}

/**
 * \brief Computes the norm of a DualNumber
 *
 * \note This is equivalent to multiplying the DualNumber by itself
 *
 * \param num The DualNumber
 * \return The norm of the DualNumber
 */
constexpr auto abs2(const DualNumber &num) -> DualNumber
{
    return num * num;
}

/**
 * \brief Computes the norm of a DualNumber
 *
 * \note This is equivalent to multiplying the DualNumber by itself
 *
 * \param num The DualNumber
 * \return The norm of the DualNumber
 */
constexpr auto norm(const DualNumber &num) -> DualNumber
{
    return num * num;
}

// Power functions
/**
 * \brief Computes a DualNumber raised to the power of a scalar exponent
 *
 * \param num The DualNumber
 * \param exponent The scalar exponent
 * \return The DualNumber raised to the exponent
 */
auto pow(const DualNumber &num, double exponent) -> DualNumber;

/**
 * \brief Computes a DualNumber raised to the power of another DualNumber
 *
 * \param num The DualNumber
 * \param exponent The exponent DualNumber
 * \return The DualNumber raised to the exponent DualNumber
 */
auto pow(const DualNumber &num, const DualNumber &exponent) -> DualNumber;

/**
 * \brief Computes the square root of a DualNumber
 *
 * \param num The DualNumber
 * \return The square root of the DualNumber
 */
auto sqrt(const DualNumber &num) -> DualNumber;

// Exponential functions
/**
 * \brief Compute e (euler's number) raised to the power of a
 * DualNumber
 *
 * \param num The DualNumber
 * \return The base-e exponential of num
 */
auto exp(const DualNumber &num) -> DualNumber;

/**
 * \brief Computes 2 raised to the power of a DualNumber
 *
 * \param num The DualNumber
 * \return The base-2 exponential of num
 */
auto exp2(const DualNumber &num) -> DualNumber;

// Logarithms
/**
 * \brief Computes the natural (base e) logarithm of a DualNumber
 *
 * \param num The DualNumber
 * \return The natural logarithm of num
 */
auto log(const DualNumber &num) -> DualNumber;

/**
 * \brief Computes the base 2 logarithm of a DualNumber
 *
 * \param num The DualNumber
 * \return The base 2 logarithm of num
 */
auto log2(const DualNumber &num) -> DualNumber;

/**
 * \brief Computes the base 10 logarithm of a DualNumber
 *
 * \param num The DualNumber
 * \return The base 10 logarithm of num
 */
auto log10(const DualNumber &num) -> DualNumber;

/**
 * \brief Computes the input base logarithm of a DualNumber
 *
 * \param num The DualNumber
 * \param base The base of the logarithm
 * \return The base base logarithm of num
 */
auto log(const DualNumber &num, double base) -> DualNumber;

// Trigonometric functions
/**
 * \brief Computes cosine of a DualNumber
 *
 * \param num The DualNumber
 * \return Cosine of the DualNumber
 */
auto cos(const DualNumber &num) -> DualNumber;

/**
 * \brief Computes sine of a DualNumber
 *
 * \param num The DualNumber
 * \return Sine of the DualNumber
 */
auto sin(const DualNumber &num) -> DualNumber;

/**
 * \brief Computes tangent of a DualNumber
 *
 * \param num The DualNumber
 * \return Tangent of the DualNumber
 */
auto tan(const DualNumber &num) -> DualNumber;

// Inverse trigonometric functions
/**
 * \brief Computes inverse cosine of a DualNumber
 *
 * \param num The DualNumber
 * \return Inverse cosine of the DualNumber
 */
auto acos(const DualNumber &num) -> DualNumber;

/**
 * \brief Computes inverse sine of a DualNumber
 *
 * \param num The DualNumber
 * \return Inverse sine of the DualNumber
 */
auto asin(const DualNumber &num) -> DualNumber;

/**
 * \brief Computes inverse tangent of a DualNumber
 *
 * \param num The DualNumber
 * \return Inverse tangent of the DualNumber
 */
auto atan(const DualNumber &num) -> DualNumber;

// Hyperbolic functions
/**
 * \brief Computes hyperbolic cosine of a DualNumber
 *
 * \param num The DualNumber
 * \return Hyperbolic cosine of the the DualNumber
 */
auto cosh(const DualNumber &num) -> DualNumber;

/**
 * \brief Computes hyperbolic sine of a DualNumber
 *
 * \param num The DualNumber
 * \return Hyperbolic sine of the the DualNumber
 */
auto sinh(const DualNumber &num) -> DualNumber;

/**
 * \brief Computes hyperbolic tangent of a DualNumber
 *
 * \param num The DualNumber
 * \return Hyperbolic tangent of the the DualNumber
 */
auto tanh(const DualNumber &num) -> DualNumber;

// Inverse hyperbolic functions
/**
 * \brief Computes inverse hyperbolic cosine of a DualNumber
 *
 * \param num The DualNumber
 * \return Inverse hyperbolic cosine of the DualNumber
 */
auto acosh(const DualNumber &num) -> DualNumber;

/**
 * \brief Computes inverse hyperbolic sine of a DualNumber
 *
 * \param num The DualNumber
 * \return Inverse hyperbolic sine of the DualNumber
 */
auto asinh(const DualNumber &num) -> DualNumber;

/**
 * \brief Computes inverse hyperbolic tangent of a DualNumber
 *
 * \param num The DualNumber
 * \return Inverse hyperbolic tangent of the DualNumber
 */
auto atanh(const DualNumber &num) -> DualNumber;

// Special case: this is just inverse; hence implemented here
/**
 * \brief Computes the inverse of a DualNumber multiplied by a scalar
 *
 * \param scalar The scalar
 * \param num The DualNumber
 * \return The inverse of the DualNumber multiplied by scalar
 */
inline auto operator/(const double scalar, DualNumber num)
{
    return scalar * inverse(num);
}

} // namespace algodiff::forward
