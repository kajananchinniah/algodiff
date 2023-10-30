#include "algodiff/dual_number.hpp"
#include "pybind11/operators.h"
#include "pybind11/pybind11.h"

#include "algodiff/algodiff.hpp"

namespace py = pybind11;

int add(int a, int b)
{
    return a + b;
}

PYBIND11_MODULE(algodiff_py, m)
{
    m.doc() = "algodiff module";

    // DualNumber class
    py::class_<algodiff::forward::DualNumber>(m, "DualNumber")
        .def(py::init<>())
        .def(py::init<double>())
        .def(py::init<double, double>())
        .def("primal",
             py::overload_cast<>(&algodiff::forward::DualNumber::primal))
        .def("primal",
             py::overload_cast<double>(&algodiff::forward::DualNumber::primal))
        .def("primal_const",
             py::overload_cast<>(&algodiff::forward::DualNumber::primal,
                                 py::const_))
        .def("dual", py::overload_cast<>(&algodiff::forward::DualNumber::dual))
        .def("dual",
             py::overload_cast<double>(&algodiff::forward::DualNumber::dual))
        .def("dual_const",
             py::overload_cast<>(&algodiff::forward::DualNumber::dual,
                                 py::const_))
        .def(-py::self)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self += py::self)
        .def(py::self += double())
        .def(py::self -= py::self)
        .def(py::self -= double())
        .def(py::self *= py::self)
        .def(py::self *= double())
        .def(py::self /= py::self)
        .def(py::self /= double())
        .def(py::self + py::self)
        .def(py::self + double())
        .def(double() + py::self)
        .def(py::self - py::self)
        .def(py::self - double())
        .def(double() - py::self)
        .def(py::self * py::self)
        .def(double() * py::self)
        .def(py::self * double())
        .def(py::self / py::self)
        .def(py::self / double())
        .def(double() / py::self);

    // DualNumber operations
    m.def("primal", &algodiff::forward::primal,
          "Returns the primal component of a DualNumber");
    m.def("real", &algodiff::forward::real,
          "Returns the primal component of a DualNumber");
    m.def("dual", &algodiff::forward::dual,
          "Returns the dual component of a DualNumber");
    m.def("imag", &algodiff::forward::imag,
          "Returns the dual component of a DualNumber");
    m.def("abs", &algodiff::forward::abs,
          "Returns the absolute value of the primal component");
    m.def("inverse", &algodiff::forward::inverse,
          "Returns the inverse of a DualNumber");
    m.def("conj", &algodiff::forward::conj,
          "Returns the conjugate of a DualNumber");
    m.def("abs2", &algodiff::forward::abs2, "Returns the norm of a DualNumber");
    m.def("norm", &algodiff::forward::norm, "Returns the norm of a DualNumber");
    m.def("pow",
          py::overload_cast<const algodiff::forward::DualNumber &, double>(
              &algodiff::forward::pow),
          "Returns a DualNumber raised to the power of a scalar exponent");
    m.def("pow",
          py::overload_cast<const algodiff::forward::DualNumber &,
                            const algodiff::forward::DualNumber &>(
              &algodiff::forward::pow),
          "Returns a DualNumber raised to the power of a another DualNumber");
    m.def("sqrt", &algodiff::forward::sqrt,
          "Returns the square root of a DualNumber");
    m.def("exp", &algodiff::forward::exp,
          "Computes e (euler's number) raised to the power of a DualNumber");
    m.def("exp2", &algodiff::forward::exp2,
          "Computes 2 raised to the power of a DualNumber");
    m.def("log",
          py::overload_cast<const algodiff::forward::DualNumber &>(
              &algodiff::forward::log),
          "Returns the natural (base e) logarithm of a DualNumber");
    m.def("log2", &algodiff::forward::log2,
          "Computes the base 2 logarithm of a DualNumber");
    m.def("log10", &algodiff::forward::log10,
          "Computes the base 10 logarithm of a DualNumber");
    m.def("log",
          py::overload_cast<const algodiff::forward::DualNumber &, double>(
              &algodiff::forward::log),
          "Compute the input base logarithm of a DualNumber");
    m.def("cos", &algodiff::forward::cos, "Computes cosine of a DualNumber");
    m.def("sin", &algodiff::forward::sin, "Computes sine of a DualNumber");
    m.def("tan", &algodiff::forward::tan, "Computes tangent of a DualNumber");
    m.def("acos", &algodiff::forward::acos,
          "Computes inverse cosine of a DualNumber");
    m.def("asin", &algodiff::forward::asin,
          "Computes inverse sine of a DualNumber");
    m.def("atan", &algodiff::forward::atan,
          "Computes inverse tangent of a DualNumber");
    m.def("cosh", &algodiff::forward::cosh,
          "Computes hyperbolic cosine of a DualNumber");
    m.def("sinh", &algodiff::forward::sinh,
          "Computes hyperbolic sine of a DualNumber");
    m.def("tanh", &algodiff::forward::tanh,
          "Computes hyperbolic tangent of a DualNumber");
    m.def("acosh", &algodiff::forward::acosh,
          "Computes inverse hyperbolic cosine of a DualNumber");
    m.def("asinh", &algodiff::forward::asinh,
          "Computes inverse hyperbolic sine of a DualNumber");
    m.def("atanh", &algodiff::forward::atanh,
          "Computes inverse hyperbolic tangent of a DualNumber");
}
