/* This file is part of the algodiff project.
 * Copyright (c) 2023 kajananchinniah
 * SPDX-License-Identifier: MIT
 */
#include <cmath>

#include "algodiff/dual_number_ops.hpp"

#include "algodiff/dual_number.hpp"

namespace algodiff::forward
{
auto abs(const DualNumber &num) -> DualNumber
{
    return DualNumber{std::abs(num.primal()),
                      num.dual() * num.primal() / std::abs(num.primal())};
}

auto inverse(const DualNumber &num) -> DualNumber
{
    return pow(num, -1.0);
}

auto pow(const DualNumber &num, const double exponent) -> DualNumber
{
    return DualNumber{std::pow(num.primal(), exponent),
                      exponent * num.dual() *
                          std::pow(num.primal(), exponent - 1.0)};
}

auto pow(const DualNumber &num, const DualNumber &exponent) -> DualNumber
{
    return DualNumber{std::pow(num.primal(), exponent.primal()),
                      std::pow(num.primal(), exponent.primal()) *
                          (exponent.dual() * std::log(num.primal()) +
                           num.dual() * exponent.primal() / num.primal())};
}

auto sqrt(const DualNumber &num) -> DualNumber
{
    constexpr double exponent{0.5};
    return pow(num, exponent);
}

auto exp(const DualNumber &num) -> DualNumber
{
    return DualNumber{std::exp(num.primal()),
                      num.dual() * std::exp(num.primal())};
}

auto exp2(const DualNumber &num) -> DualNumber
{
    return exp(std::log(2.0) * num); // NOLINT
}

auto log(const DualNumber &num) -> DualNumber
{
    return DualNumber{std::log(num.primal()), num.dual() / num.primal()};
}

auto log2(const DualNumber &num) -> DualNumber
{
    return log(num) / std::log(2.0); // NOLINT
}

auto log10(const DualNumber &num) -> DualNumber
{
    return log(num) / std::log(10.0); // NOLINT
}

auto log(const DualNumber &num, const double base) -> DualNumber
{
    return log(num) / std::log(base);
}

auto sin(const DualNumber &num) -> DualNumber
{
    return DualNumber{std::sin(num.primal()),
                      std::cos(num.primal()) * num.dual()};
}

auto cos(const DualNumber &num) -> DualNumber
{
    return DualNumber{std::cos(num.primal()),
                      -std::sin(num.primal()) * num.dual()};
}

auto tan(const DualNumber &num) -> DualNumber
{
    const double cos_primal = std::cos(num.primal());
    return DualNumber{std::tan(num.primal()),
                      num.dual() / (cos_primal * cos_primal)};
}

auto asin(const DualNumber &num) -> DualNumber
{
    return DualNumber{std::asin(num.primal()),
                      num.dual() /
                          std::sqrt(1.0 - num.primal() * num.primal())};
}

auto acos(const DualNumber &num) -> DualNumber
{
    return DualNumber{std::acos(num.primal()),
                      -num.dual() /
                          std::sqrt(1.0 - num.primal() * num.primal())};
}

auto atan(const DualNumber &num) -> DualNumber
{
    return DualNumber{std::atan(num.primal()),
                      num.dual() / (1.0 + num.primal() * num.primal())};
}

auto sinh(const DualNumber &num) -> DualNumber
{
    return DualNumber{std::sinh(num.primal()),
                      std::cosh(num.primal()) * num.dual()};
}

auto cosh(const DualNumber &num) -> DualNumber
{
    return DualNumber{std::cosh(num.primal()),
                      std::sinh(num.primal()) * num.dual()};
}

auto tanh(const DualNumber &num) -> DualNumber
{
    const double cosh_primal = std::cosh(num.primal());
    return DualNumber{std::tanh(num.primal()),
                      num.dual() / (cosh_primal * cosh_primal)};
}

auto asinh(const DualNumber &num) -> DualNumber
{
    return DualNumber{std::asinh(num.primal()),
                      num.dual() /
                          (std::sqrt(num.primal() * num.primal() + 1.0))};
}

auto acosh(const DualNumber &num) -> DualNumber
{
    return DualNumber{std::acosh(num.primal()),
                      num.dual() /
                          (std::sqrt(num.primal() * num.primal() - 1.0))};
}

auto atanh(const DualNumber &num) -> DualNumber
{
    return DualNumber{std::atanh(num.primal()),
                      num.dual() / (1.0 - num.primal() * num.primal())};
}

} // namespace algodiff::forward
