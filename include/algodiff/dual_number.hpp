/* This file is part of the algodiff project.
 * Copyright (c) 2023 kajananchinniah
 * SPDX-License-Identifier: MIT
 */
/// \file dual_number.hpp
/// \brief Contains the implementation of a dual number
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

namespace algodiff::forward
{
/**
 * A dual number class used to compute functions and derivative.
 *
 * See: https://en.wikipedia.org/wiki/Dual_Number for more details
 */
class DualNumber
{
public:
    /// The default constructor
    constexpr DualNumber() = default;

    /**
     *  \brief Creates a DualNumber with the specified primal component and
     *  zero dual component
     *
     *  \param primal The primal component
     */
    constexpr explicit DualNumber(double primal) : m_primal{primal}
    {
    }

    /**
     * \brief Creates a dual number with the specified primal component and
     * specified dual component
     *
     * \param primal The primal component
     * \param dual The dual component
     */
    constexpr DualNumber(double primal, double dual)
        : m_primal{primal}, m_dual{dual}
    {
    }

    /**
     * \brief Returns a mutable reference to the primal component
     *
     * \return The primal component
     */
    constexpr auto primal() -> double &
    {
        return m_primal;
    }

    /**
     * \brief Sets the primal component to value
     */
    constexpr auto primal(double value) -> void
    {
        m_primal = value;
    }

    /**
     * \brief Returns a copy of the primal component
     *
     * \return The primal component
     */
    constexpr auto primal() const -> double
    {
        return m_primal;
    }

    /**
     * \brief Returns a mutable reference to the dual component
     *
     * \return The dual component
     */
    constexpr auto dual() -> double &
    {
        return m_dual;
    }

    /**
     * \brief Sets the primal component to value
     */
    constexpr auto dual(double value) -> void
    {
        m_dual = value;
    }

    /**
     * \brief Returns a copy of the dual component
     *
     * \return The dual component
     */
    constexpr auto dual() const -> double
    {
        return m_dual;
    }

    /**
     * \brief Returns the negation of the DualNumber (e.g. of negative the
     * primal component and negative of the dual component)
     *
     * \return The negation of the DualNumber
     */
    constexpr auto operator-()
    {
        return DualNumber{-primal(), -dual()};
    }

    /**
     *  \brief Compares two DualNumbers for equality
     *
     *  \param other The other DualNumber
     *  \return true if the input DualNumber and other are equal, false
     * otherwise
     */
    constexpr auto operator==(const DualNumber &other) const -> bool
    {
        // TODO(kajananchinniah): use:
        // http://realtimecollisiondetection.net/blog/?p=89 for equality
        return this == &other || (std::abs(primal() - other.primal()) <
                                      std::numeric_limits<double>::epsilon() &&
                                  (std::abs(dual() - other.dual()) <
                                   std::numeric_limits<double>::epsilon()));
    }

    /**
     * \brief Compares two DualNumbers for inequality
     *
     * \param other The other DualNumber
     * \return true if the input DualNumber and other are unequal, false
     * otherwise
     */
    constexpr auto operator!=(const DualNumber &other) const -> bool
    {
        return !(*this == other);
    }

    /**
     * \brief Adds other to *this
     *
     * \param other A DualNumber
     * \return The sum of *this and other
     */
    constexpr auto operator+=(const DualNumber &other) -> DualNumber &
    {
        primal() += other.primal();
        dual() += other.dual();
        return *this;
    }

    /**
     * \brief Adds a scalar to *this
     *
     * \note The scalar is treated as a DualNumber with primal part equal to n
     * and dual part set to zero
     *
     * \param n A scalar value
     * \return The sum of *this with the scalar
     */
    constexpr auto operator+=(const double n) -> DualNumber &
    {
        primal() += n;
        return *this;
    }

    /**
     * \brief Subtracts other from *this
     *
     * \param other The subtrahend DualNumber
     * \return The difference of *this and other
     */
    constexpr auto operator-=(const DualNumber &other) -> DualNumber &
    {
        primal() -= other.primal();
        dual() -= other.dual();
        return *this;
    }

    /**
     * \brief Subtracts n from *this
     *
     * \note The scalar is treated as a DualNumber with primal part equal to n
     * and dual part set to zero
     *
     * \param n The subtrahend scalar
     * \return The difference of the DualNumber with the scalar
     */
    constexpr auto operator-=(const double n) -> DualNumber &
    {
        primal() -= n;
        return *this;
    }

    /**
     * \brief Multiples *this by other
     *
     * \param other A DualNumber
     * \return The product of the two DualNumbers
     */
    constexpr auto operator*=(const DualNumber &other) -> DualNumber &
    {
        const auto primal_comp{primal()};
        const auto dual_comp{dual()};
        primal() = primal_comp * other.primal();
        dual() = primal_comp * other.dual() + dual_comp * other.primal();
        return *this;
    }

    /**
     * \brief Multiples *this by scalar
     *
     * \param scalar The scalar
     * \return The product of the DualNumber and the scalar
     */
    constexpr auto operator*=(const double scalar) -> DualNumber &
    {
        primal() = scalar * primal();
        dual() = scalar * dual();
        return *this;
    }

    /**
     * \brief Divides *this by other
     *
     * \param other The divisor DualNumber
     * \return The quotient of the two DualNumbers
     */
    constexpr auto operator/=(const DualNumber &other) -> DualNumber &
    {
        const auto primal_comp{primal()};
        const auto dual_comp{dual()};
        primal() = primal_comp / other.primal();
        dual() = (dual_comp * other.primal() - primal_comp * other.dual()) /
                 (other.primal() * other.primal());
        return *this;
    }

    /**
     * \brief Divides *this by scalar
     *
     * \param scalar The scalar (divisor)
     * \return The quotient of the DualNumber with the scalar
     */
    constexpr auto operator/=(const double scalar) -> DualNumber &
    {
        primal() = primal() / scalar;
        dual() = dual() / scalar;
        return *this;
    }

private:
    /// The primal component
    double m_primal{0.0};

    /// The dual component
    double m_dual{0.0};
};

/**
 * \brief Adds left and right
 *
 * \param left A DualNumber
 * \param right The other DualNumber
 * \return The sum of the two DualNumbers
 */
constexpr inline auto operator+(DualNumber left, const DualNumber &right)
{
    left += right;
    return left;
}

/**
 * \brief Adds num with n
 *
 * \note The scalar is treated as a DualNumber with primal part equal to n
 * and dual part set to zero
 *
 * \param num The DualNumber
 * \param n The scalar
 * \return The sum of the DualNumber with the scalar
 */
constexpr inline auto operator+(DualNumber num, const double n)
{
    num += n;
    return num;
}

/**
 * \brief Adds num with n
 *
 * \note The scalar is treated as a DualNumber with primal part equal to n
 * and dual part set to zero
 *
 * \param n The scalar
 * \param num The DualNumber
 * \return The sum of the DualNumber with the scalar
 */
constexpr inline auto operator+(const double n, DualNumber num)
{
    num += n;
    return num;
}

/**
 * \brief Subtracts right from left
 *
 * \param left The minuend DualNumber
 * \param right The subtrahend DualNumber
 * \return The difference between the left and right DualNumbers
 */
constexpr inline auto operator-(DualNumber left, const DualNumber &right)
{
    left -= right;
    return left;
}

/**
 * \brief Returns the negation of num
 *
 * \param num A DualNumber
 * \return The negation of the DualNumber
 */
constexpr inline auto operator-(const DualNumber &num)
{
    return DualNumber{-num.primal(), -num.dual()};
}

/**
 * \brief Subtracts n from num
 *
 * \note The scalar is treated as a DualNumber with primal part equal to n
 * and dual part set to zero
 *
 * \param num The minuend DualNumber
 * \param n The scalar (subtrahend)
 * \return The difference between the DualNumber and the scalar
 */
constexpr inline auto operator-(DualNumber num, const double n)
{
    num -= n;
    return num;
}

/**
 * \brief Subtracts num from n
 *
 * \warning The resultant DualNumber has negative num's dual component
 *
 * \param n The scalar (minuend)
 * \param num The DualNumber (subtrahend)
 * \return The difference between the DualNumber and the scalar
 */
constexpr inline auto operator-(const double n, DualNumber num)
{
    num.primal() = n - num.primal();
    num.dual() = -num.dual();
    return num;
}

/**
 * \brief Multiplies left and right
 *
 * \param left A DualNumber
 * \param right The other DualNumber
 * \return The product between the left and right DualNumber
 */
constexpr inline auto operator*(DualNumber left, const DualNumber &right)
{
    left *= right;
    return left;
}

/**
 * \brief Multiplies scalar with num
 *
 * \param scalar The scalar
 * \param num The DualNumber
 * \return The product between the DualNumber and the scalar
 */
constexpr inline auto operator*(const double scalar, DualNumber num)
{
    num *= scalar;
    return num;
}

/**
 * \brief Multiplies num with scalar
 *
 * \param num The DualNumber
 * \param scalar The scalar
 * \return The product between the DualNumber and the scalar
 */
constexpr inline auto operator*(DualNumber num, const double scalar)
{
    num *= scalar;
    return num;
}

/**
 * \brief Divides left by right
 *
 * \param left The dividend DualNumber
 * \param right The divisor DualNumber
 * \return The quotient between the left and right DualNumber
 */
constexpr inline auto operator/(DualNumber left, const DualNumber &right)
{
    left /= right;
    return left;
}

/**
 * \brief Divides num by scalar
 *
 * \param num The dividend DualNumber
 * \param scalar The scalar (divisor)
 * \return The quotient between the DualNumber and the scalar
 */
constexpr inline auto operator/(DualNumber num, const double scalar)
{
    num /= scalar;
    return num;
}

} // namespace algodiff::forward
