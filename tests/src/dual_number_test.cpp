/* This file is part of the algodiff project.
 * Copyright (c) 2022 kajananchinniah
 * SPDX-License-Identifier: MIT
 */
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

#include "algodiff/dual_number.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "algodiff/dual_number_ops.hpp"

namespace
{
class random_number_generator
{
public:
    random_number_generator(double min, double max)
        : m_rd{}, m_rng{m_rd()}, m_distribution{min, max}
    {
    }
    auto operator()() -> double
    {
        auto ret{m_distribution(m_rng)};
        // Exclude generation 0
        // TODO: this is not a good method
        while (std::abs(ret) < std::numeric_limits<double>::epsilon()) {
            ret = m_distribution(m_rng);
        }
        return ret;
    }

private:
    std::random_device m_rd;
    std::mt19937 m_rng;
    std::uniform_real_distribution<> m_distribution;
};

} // namespace

TEST_CASE("Test DualNumber operations", "[DualNumber]")
{
    random_number_generator rng{-10.0, 10.0};

    SECTION("setting dual number")
    {
        const auto primal{rng()};
        const auto dual{rng()};

        algodiff::forward::DualNumber a{primal, dual};

        REQUIRE(a.primal() == Catch::Approx(primal));
        REQUIRE(a.dual() == Catch::Approx(dual));

        a = algodiff::forward::DualNumber{primal};
        REQUIRE(a.primal() == Catch::Approx(primal));
        REQUIRE(a.dual() == Catch::Approx(0.0));

        a = algodiff::forward::DualNumber{};
        REQUIRE(a.primal() == Catch::Approx(0.0));
        REQUIRE(a.dual() == Catch::Approx(0.0));

        a.primal(primal);
        a.dual(dual);
        REQUIRE(a.primal() == Catch::Approx(primal));
        REQUIRE(a.dual() == Catch::Approx(dual));
    }

    SECTION("negation of dual number")
    {
        algodiff::forward::DualNumber a{rng(), rng()};

        const auto neg_a = -a;

        REQUIRE(neg_a.primal() == Catch::Approx(-a.primal()));
        REQUIRE(neg_a.dual() == Catch::Approx(-a.dual()));
    }

    SECTION("equality and inequality of dual number")
    {
        algodiff::forward::DualNumber a{rng(), rng()};
        algodiff::forward::DualNumber a_copy{a};

        std::vector<algodiff::forward::DualNumber> unequal_cases = {
            algodiff::forward::DualNumber{-11.0, -100.0},
            algodiff::forward::DualNumber{a.primal(), -100.0},
            algodiff::forward::DualNumber{-11.0, a.dual()}};

        // Test self equality
        REQUIRE((a == a) == true);
        REQUIRE((a != a) == false);

        // Test equality with a copy
        REQUIRE((a == a_copy) == true);
        REQUIRE((a != a_copy) == false);

        // Test unequal cases
        std::for_each(unequal_cases.begin(), unequal_cases.end(),
                      [&](algodiff::forward::DualNumber test_case) {
                          REQUIRE((a == test_case) == false);
                          REQUIRE((a != test_case) == true);
                      });
    }

    SECTION("mathematical assignment operators")
    {
        // Constants for the two dual numbers
        const double a_primal{rng()};
        const double a_dual{rng()};
        const double b_primal{rng()};
        const double b_dual{rng()};
        const double scalar{rng()};
        const algodiff::forward::DualNumber b{b_primal, b_dual};

        // Test addition assignment
        algodiff::forward::DualNumber a{a_primal, a_dual};
        a += b;
        REQUIRE(a.primal() == Catch::Approx((a_primal + b_primal)));
        REQUIRE(a.dual() == Catch::Approx((a_dual + b_dual)));

        // Reset the numbers and test subtraction
        a = algodiff::forward::DualNumber{a_primal, a_dual};
        a -= b;
        REQUIRE(a.primal() == Catch::Approx((a_primal - b_primal)));
        REQUIRE(a.dual() == Catch::Approx((a_dual - b_dual)));

        // Reset the numbers and test scalar multiplication
        a = algodiff::forward::DualNumber{a_primal, a_dual};
        a *= scalar;
        REQUIRE(a.primal() == Catch::Approx(scalar * a_primal));
        REQUIRE(a.dual() == Catch::Approx(scalar * a_dual));

        // Reset the numbers and test scalar division
        a = algodiff::forward::DualNumber{a_primal, a_dual};
        a /= scalar;
        REQUIRE(a.primal() == Catch::Approx(a_primal / scalar));
        REQUIRE(a.dual() == Catch::Approx(a_dual / scalar));

        // Reset the numbers and test dual number multiplication
        a = algodiff::forward::DualNumber{a_primal, a_dual};
        a *= b;
        REQUIRE(a.primal() == Catch::Approx(a_primal * b_primal));
        REQUIRE(a.dual() ==
                Catch::Approx(a_primal * b_dual + a_dual * b_primal));

        // Reset the numbers and test dual number division
        a = algodiff::forward::DualNumber{a_primal, a_dual};
        a /= b;
        REQUIRE(a.primal() == Catch::Approx(a_primal / b_primal));
        REQUIRE(a.dual() ==
                Catch::Approx((a_dual * b_primal - a_primal * b_dual) /
                              (b_primal * b_primal)));
    }

    SECTION("mathematical operators")
    {
        const algodiff::forward::DualNumber a{rng(), rng()};
        const algodiff::forward::DualNumber b{rng(), rng()};
        const double scalar{rng()};

        algodiff::forward::DualNumber c = a + b;
        REQUIRE(c.primal() == Catch::Approx(a.primal() + b.primal()));
        REQUIRE(c.dual() == Catch::Approx(a.dual() + b.dual()));

        c = a - b;
        REQUIRE(c.primal() == Catch::Approx(a.primal() - b.primal()));
        REQUIRE(c.dual() == Catch::Approx(a.dual() - b.dual()));

        c = scalar * a;
        REQUIRE(c.primal() == Catch::Approx(a.primal() * scalar));
        REQUIRE(c.dual() == Catch::Approx(a.dual() * scalar));

        c = a * scalar;
        REQUIRE(c.primal() == Catch::Approx(a.primal() * scalar));
        REQUIRE(c.dual() == Catch::Approx(a.dual() * scalar));

        c = scalar - a;
        REQUIRE(c.primal() == Catch::Approx(scalar - a.primal()));
        REQUIRE(c.dual() == Catch::Approx(-a.dual()));

        c = a / scalar;
        REQUIRE(c.primal() == Catch::Approx(a.primal() / scalar));
        REQUIRE(c.dual() == Catch::Approx(a.dual() / scalar));

        c = a * b;
        REQUIRE(c.primal() == Catch::Approx(a.primal() * b.primal()));
        REQUIRE(c.dual() ==
                Catch::Approx(a.primal() * b.dual() + a.dual() * b.primal()));

        c = a / b;
        REQUIRE(c.primal() == Catch::Approx(a.primal() / b.primal()));
        REQUIRE(c.dual() ==
                Catch::Approx((a.dual() * b.primal() - a.primal() * b.dual()) /
                              (b.primal() * b.primal())));
    }
}

TEST_CASE("Non member functions", "[DualNumber]")
{
    random_number_generator rng{-10.0, 10.0};
    const algodiff::forward::DualNumber a{rng(), rng()};
    REQUIRE(a.primal() == Catch::Approx(algodiff::forward::primal(a)));
    REQUIRE(a.dual() == Catch::Approx(algodiff::forward::dual(a)));
    REQUIRE(a.primal() == Catch::Approx(algodiff::forward::real(a)));
    REQUIRE(a.dual() == Catch::Approx(algodiff::forward::imag(a)));

    const algodiff::forward::DualNumber a_conj{algodiff::forward::conj(a)};
    REQUIRE(a.primal() == Catch::Approx(a_conj.primal()));
    REQUIRE(-a.dual() == Catch::Approx(a_conj.dual()));

    const auto norm_a = algodiff::forward::norm(a);
    const auto abs2_a = algodiff::forward::abs2(a);
    const auto aa = a * a;
    REQUIRE(aa.primal() == Catch::Approx(norm_a.primal()));
    REQUIRE(aa.primal() == Catch::Approx(abs2_a.primal()));
    REQUIRE(aa.dual() == Catch::Approx(norm_a.dual()));
    REQUIRE(aa.dual() == Catch::Approx(abs2_a.dual()));
}
