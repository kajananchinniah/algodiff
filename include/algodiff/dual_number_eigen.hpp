/* This file is part of the algodiff project.
 * Copyright (c) 2022 kajananchinniah
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <Eigen/Core>

#include "dual_number.hpp"
#include "dual_number_ops.hpp"

namespace Eigen
{
template <>
struct NumTraits<algodiff::forward::DualNumber> : NumTraits<double> {
    typedef algodiff::forward::DualNumber Real;       // NOLINT
    typedef algodiff::forward::DualNumber NonInteger; // NOLINT
    typedef algodiff::forward::DualNumber Nested;     // NOLINT

    enum {
        IsComplex = 0,             // NOLINT
        IsInteger = 0,             // NOLINT
        IsSigned = 1,              // NOLINT
        RequireInitialization = 1, // NOLINT
        ReadCost = 1,              // NOLINT
        AddCost = 3,               // NOLINT
        MulCost = 3,               // NOLINT
    };
};

template <typename BinaryOp>
struct ScalarBinaryOpTraits<algodiff::forward::DualNumber, double, BinaryOp> {
    typedef algodiff::forward::DualNumber ReturnType; // NOLINT
};

template <typename BinaryOp>
struct ScalarBinaryOpTraits<double, algodiff::forward::DualNumber, BinaryOp> {
    typedef algodiff::forward::DualNumber ReturnType; // NOLINT
};

} // namespace Eigen
