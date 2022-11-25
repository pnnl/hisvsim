#ifndef TYPES_H__
#define TYPES_H__

#include <complex>

namespace SvSim {

constexpr unsigned MAX_CONTROL_QUBITS = 5;
constexpr unsigned MAX_TARGET_QUBITS = 5;
constexpr double threshold = 1e-10;

template <typename T1, typename T2>
bool approx_equal(T1 a, T2 b) {
  return std::sqrt(std::abs(pow(a - b, 2))) <= threshold;
}

namespace Constants {

constexpr double pi = 3.14159265358979323846;
constexpr double pi_by_2 = 1.57079632679489661923;
constexpr double inv_pi = 0.318309886183790671538;
constexpr double sqrt_2 = 1.41421356237309504880;
constexpr double inv_sqrt_2 = 0.707106781186547524401;
constexpr double e = 2.71828182845904523536;
};  // namespace Constants

using Type = std::complex<double>;
using Qubit = int32_t;
using Slot = int32_t;
using Value = int32_t;

};  // namespace SvSim

#define NOT_IMPLEMENTED() assert(0)
#define UNREACHABLE() assert(0)

#define EXPECTS(cond, msg) if(!(cond)) { std::cerr<<"ERROR Message: "<<msg<<"\n"; assert(cond); }
#define ALIGNMENT 64

#ifndef AVX512
  #define STRIDE 1
#else
  #define STRIDE 2
#endif

#endif  // TYPES_H__
