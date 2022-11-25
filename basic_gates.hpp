#ifndef BASIC_GATES_HPP__
#define BASIC_GATES_HPP__

#include <cassert>
#include <complex>

#include "types.hpp"
#include "immintrin.h"

namespace SvSim {

enum class GateType {
  invalid,
  ALLOC,
  FREE,
  I,
  X,
  Y,
  Z,
  H,
  S,
  T,
  RX,
  RY,
  RZ,
  RI,
  R1,
  RXFrac,
  RYFrac,
  RZFrac,
  RIFrac,
  R1Frac,
  SWAP,
  C1,
  C2,
  U1,
  U2,
  U3,
  RXX,
  RYY,
  RZZ,
  TDG,
  SDG,
  // Control
  // RCCX,
  // CCX,
  // C3X,
  // C4X,
  // CSWAP,
  // CU1,
  // CU2,
  // CU3,
  // C3XSQRTX,
  // RC3X,
  // CX,
  // CY,
  // CZ,
  // CRX,
  // CRY,
  // CRZ,
  //	RC3X Relative-phase 3-controlled X
  //	RXX	2-qubit XX rotation
  //	RZZ	2-qubit ZZ rotation
  //	RCCX	Relative-phase CXX
  // TDG	conjugate of sqrt(S)
  // SDG	conjugate of sqrt(Z)
};

enum class AGType{
  invalid,
  NOPARAM,
  ONEPARAM,
  TWOPARAM,
  THREEPARAM,
  C1,
  C2,
  FRAC,
};

namespace BasicGates {

#ifdef AVX512
  //const __m512d avx_h_timer1 = _mm512_set4_pd(-1.0,-1.0,1.0,1.0);
  //const __m512d avx_h_timer2 = _mm512_set4_pd(Constants::inv_sqrt_2,Constants::inv_sqrt_2,Constants::inv_sqrt_2,Constants::inv_sqrt_2);
  const __m512d avx_y_timer = _mm512_set4_pd(1.0,-1.0,-1.0,1.0);
  const __m512d avx_z_timer = _mm512_set4_pd(-1.0,-1.0,1.0,1.0);
  const __m512d avx_s_timer = _mm512_set4_pd(1.0,-1.0,1.0,1.0);
  const __m512d avx_t_timer = _mm512_set4_pd(1.0,-1.0,0,0);
  const __m512d avx_t_timer2 = _mm512_set4_pd(Constants::inv_sqrt_2,Constants::inv_sqrt_2,Constants::inv_sqrt_2,Constants::inv_sqrt_2);
  const __m512d avx_tdg_timer = _mm512_set4_pd(-1.0,1.0,0.0,0.0);
  const __m512d avx_sdg_timer = _mm512_set4_pd(-1.0,1.0,1.0,1.0);
  const __m512d avx_tdg_timer_2 = _mm512_set4_pd(Constants::inv_sqrt_2,Constants::inv_sqrt_2,1.0,1.0);
#endif
/*struct AbstractGate{
  virtual void operator()(Type* base, size_t num_slots, const Slot slots[]) const = 0;
};*/

//struct I : public AbstractGate{
struct I {
  I() = default;
  constexpr GateType gate_type() const { return GateType::I; }
  void operator()(Type* base, size_t num_slots, const Slot slots[]) const {
    assert(num_slots == 1);
    // no-op
  }
};

//struct X : public AbstractGate {
  struct X {
  X() = default;
  constexpr GateType gate_type() const { return GateType::X; }
  void operator()(Type* base, size_t num_slots, const Slot slots[]) const {
    assert(num_slots == 1);
    #ifndef AVX512
      std::swap(base[0ul << slots[0]], base[1ul << slots[0]]);
    #else
      // avx512 implementation 
      int64_t pos_offset = 1ul << slots[0];
      if (pos_offset == 1){
        __m512d avx_x_0 = _mm512_load_pd((void const*)base);
        __m512d avx_x_dst_0 = _mm512_permutex_pd(avx_x_0,78);
        _mm512_store_pd((void *)base,avx_x_dst_0);
      }
      else{
        int64_t __attribute__((aligned(64))) i_index_ar[8] = {0,1,0+2*pos_offset,1+2*pos_offset,2,3,2+2*pos_offset,3+2*pos_offset};
        __m512i i_index = _mm512_load_epi64(i_index_ar);
        __m512d avx_x_t = _mm512_i64gather_pd(i_index, (void const *)base, sizeof(double));
        __m512d avx_x_dst = _mm512_permutex_pd(avx_x_t,78);
        _mm512_i64scatter_pd((void *)base, i_index, avx_x_dst, sizeof(double));
      }
    #endif
  }
};

//struct Y : public AbstractGate {
  struct Y {
  Y() = default;
  constexpr GateType gate_type() const { return GateType::Y; }
  void operator()(Type* base, size_t num_slots, const Slot slots[]) const {
    assert(num_slots == 1);
    #ifndef AVX512
      using namespace std::complex_literals;
      auto v0 = base[0ul << slots[0]];
      auto v1 = base[1ul << slots[0]];
      base[0ul << slots[0]] = -1.0i * v1;
      base[1ul << slots[0]] = 1.0i * v0;
    #else
      int64_t pos_offset = 1ul << slots[0];
      if (pos_offset == 1){
        __m512d avx_y_0 = _mm512_load_pd((void const*)base);
         //std::cout <<avx_x_0[0] << avx_x_0[1] <<  avx_x_0[2] <<  avx_x_0[3]
         //<<avx_x_0[4] << avx_x_0[5] <<  avx_x_0[6] <<  avx_x_0[7] << std::endl;
         __m512d avx_y_dst_0 = _mm512_permutex_pd(avx_y_0,27);
         //std::cout <<avx_x_dst_0[0] << avx_x_dst_0[1] <<  avx_x_dst_0[2] <<  avx_x_dst_0[3]
         //<<avx_x_dst_0[4] << avx_x_dst_0[5] <<  avx_x_dst_0[6] <<  avx_x_dst_0[7] << std::endl;
         //      __m512d avx_y_timer = _mm512_set4_pd(1.0,-1.0,-1.0,1.0);
        __m512d avx_y_dst_mul = _mm512_mul_pd(avx_y_dst_0,avx_y_timer);
        _mm512_store_pd((void *)base,avx_y_dst_mul);
      }
      else{
        int64_t __attribute__((aligned(64))) i_index_ar[8] = {0,1,0+2*pos_offset,1+2*pos_offset,2,3,2+2*pos_offset,3+2*pos_offset};
        __m512i i_index = _mm512_load_epi64(i_index_ar);
        __m512d avx_y_t = _mm512_i64gather_pd(i_index, (void const *)base,sizeof(double));
        __m512d avx_y_dst_0 = _mm512_permutex_pd(avx_y_t,27);
       // __m512d avx_y_timer = _mm512_set4_pd(1.0,-1.0,-1.0,1.0);
        __m512d avx_y_dst_mul = _mm512_mul_pd(avx_y_dst_0,avx_y_timer);
        _mm512_i64scatter_pd((void *)base, i_index, avx_y_dst_mul, sizeof(double));
      }
    #endif
    }
};

struct H {
  //H() = default;
  H(){
    #ifdef AVX512
      avx_h_timer1 = _mm512_set4_pd(-1.0,-1.0,1.0,1.0);
      avx_h_timer2 = _mm512_set4_pd(Constants::inv_sqrt_2,Constants::inv_sqrt_2,Constants::inv_sqrt_2,Constants::inv_sqrt_2);
    #endif
  }
  constexpr GateType gate_type() const { return GateType::H; }  
  inline __attribute__((always_inline)) void operator()(Type* base, size_t num_slots, const Slot slots[]) const{
    assert(num_slots == 1);
    using namespace std::complex_literals;
    #ifndef AVX512
      auto v0 = base[0ul << slots[0]];
      auto v1 = base[1ul << slots[0]];
      base[0ul << slots[0]] = Constants::inv_sqrt_2 * (v0 + v1);
      base[1ul << slots[0]] = Constants::inv_sqrt_2 * (v0 - v1);
    #else
      int64_t pos_offset = 1ul << slots[0];
      if (pos_offset == 1){
       // __m512d avx_h_0 = _mm512_loadu_pd((void const*)base);
         __m512d avx_h_0 = _mm512_load_pd((void const*)base);
    
         __m512d avx_h_1 = _mm512_permutex_pd(avx_h_0,78);   
        __m512d avx_h_tmp1 = _mm512_mul_pd(avx_h_0,avx_h_timer1);
        __m512d avx_h_tmp2 = _mm512_add_pd(avx_h_1,avx_h_tmp1);
        __m512d avx_h_tmp3 = _mm512_mul_pd(avx_h_tmp2,avx_h_timer2);
        _mm512_store_pd((void *)base,avx_h_tmp3);
      }
    else{
        int64_t __attribute__((aligned(64))) i_index_ar[8] = {0,1,0+2*pos_offset,1+2*pos_offset,2,3,2+2*pos_offset,3+2*pos_offset};
        __m512i i_index = _mm512_load_epi64(i_index_ar);
        __m512d avx_h_t = _mm512_i64gather_pd(i_index, (void const *)base,sizeof(double));
      //   std::cout <<"in h 1 case "<<avx_h_t[0] << " " << avx_h_t[1] <<" "<<  avx_h_t[2]<<" " <<  avx_h_t[3]<<" "
    //    <<avx_h_t[4] <<" "<< avx_h_t[5] <<" "<<  avx_h_t[6]<<" " <<  avx_h_t[7] << std::endl;
        __m512d avx_h_1 = _mm512_permutex_pd(avx_h_t,78);
       // __m512d avx_h_timer1 = _mm512_set4_pd(-1.0,-1.0,1.0,1.0);
        __m512d avx_h_tmp1 = _mm512_mul_pd(avx_h_t,avx_h_timer1);
         __m512d avx_h_tmp2 = _mm512_add_pd(avx_h_1,avx_h_tmp1);
        //__m512d avx_h_timer2 = _mm512_set4_pd(Constants::inv_sqrt_2,Constants::inv_sqrt_2,Constants::inv_sqrt_2,Constants::inv_sqrt_2);
        __m512d avx_h_tmp3 = _mm512_mul_pd(avx_h_tmp2,avx_h_timer2);
        _mm512_i64scatter_pd((void *)base, i_index, avx_h_tmp3, sizeof(double));
      }
    #endif
  }
  #ifdef AVX512
    __m512d avx_h_timer1; 
    __m512d avx_h_timer2;
  #endif
};

struct Z {
  Z() = default;
  constexpr GateType gate_type() const { return GateType::Z; }
  void operator()(Type* base, size_t num_slots, const Slot slots[]) const {
    assert(num_slots == 1);
    #ifndef AVX512
      base[1ul << slots[0]] *= -1.0;
    #else
       int64_t pos_offset = 1ul << slots[0];
      if (pos_offset == 1){
        __m512d avx_z_0 = _mm512_load_pd((void const*)base);
         //std::cout <<avx_x_0[0] << avx_x_0[1] <<  avx_x_0[2] <<  avx_x_0[3]
         //<<avx_x_0[4] << avx_x_0[5] <<  avx_x_0[6] <<  avx_x_0[7] << std::endl;
         //std::cout <<avx_x_dst_0[0] << avx_x_dst_0[1] <<  avx_x_dst_0[2] <<  avx_x_dst_0[3]
         //<<avx_x_dst_0[4] << avx_x_dst_0[5] <<  avx_x_dst_0[6] <<  avx_x_dst_0[7] << std::endl;
        //__m512d avx_z_timer = _mm512_set4_pd(-1.0,-1.0,1.0,1.0);
        __m512d avx_z_dst_mul = _mm512_mul_pd(avx_z_0,avx_z_timer);
        _mm512_store_pd((void *)base,avx_z_dst_mul);
      }
      else{
        int64_t __attribute__((aligned(64))) i_index_ar[8] = {0,1,0+2*pos_offset,1+2*pos_offset,2,3,2+2*pos_offset,3+2*pos_offset};
        __m512i i_index = _mm512_load_epi64(i_index_ar);
        __m512d avx_z_t = _mm512_i64gather_pd(i_index, (void const *)base,sizeof(double));
        //__m512d avx_z_timer = _mm512_set4_pd(-1.0,-1.0,1.0,1.0);
        __m512d avx_z_dst_mul = _mm512_mul_pd(avx_z_t,avx_z_timer);
        _mm512_i64scatter_pd((void *)base, i_index, avx_z_dst_mul, sizeof(double));
      }
    #endif
  }
};

struct S {
  S() = default;
  constexpr GateType gate_type() const { return GateType::S; }
  void operator()(Type* base, size_t num_slots, const Slot slots[]) const {
    assert(num_slots == 1);
    using namespace std::complex_literals;
     #ifndef AVX512
      base[1ul << slots[0]] *= 1.0i;
    #else
       int64_t pos_offset = 1ul << slots[0];
      if (pos_offset == 1){
        __m512d avx_s_0 = _mm512_load_pd((void const*)base);
        __m512d avx_s_dst_0 = _mm512_permutex_pd(avx_s_0,180);
         //std::cout <<avx_x_0[0] << avx_x_0[1] <<  avx_x_0[2] <<  avx_x_0[3]
         //<<avx_x_0[4] << avx_x_0[5] <<  avx_x_0[6] <<  avx_x_0[7] << std::endl;
         //std::cout <<avx_x_dst_0[0] << avx_x_dst_0[1] <<  avx_x_dst_0[2] <<  avx_x_dst_0[3]
         //<<avx_x_dst_0[4] << avx_x_dst_0[5] <<  avx_x_dst_0[6] <<  avx_x_dst_0[7] << std::endl;
        //__m512d avx_s_timer = _mm512_set4_pd(1.0,-1.0,1.0,1.0);
        __m512d avx_s_dst_mul = _mm512_mul_pd(avx_s_dst_0,avx_s_timer);
        _mm512_store_pd((void *)base,avx_s_dst_mul);
      }
      else{
        int64_t __attribute__((aligned(64))) i_index_ar[8] = {0,1,0+2*pos_offset,1+2*pos_offset,2,3,2+2*pos_offset,3+2*pos_offset};
        __m512i i_index = _mm512_load_epi64(i_index_ar);
        __m512d avx_s_t = _mm512_i64gather_pd(i_index, (void const *)base,sizeof(double));
        __m512d avx_s_dst_0 = _mm512_permutex_pd(avx_s_t,180);
       // __m512d avx_s_timer = _mm512_set4_pd(1.0,-1.0,1.0,1.0);
        __m512d avx_s_dst_mul = _mm512_mul_pd(avx_s_dst_0,avx_s_timer);
        _mm512_i64scatter_pd((void *)base, i_index, avx_s_dst_mul, sizeof(double));
      }
    #endif
    
  }
};

struct T {
  T() = default;
  constexpr GateType gate_type() const { return GateType::T; }
  void operator()(Type* base, size_t num_slots, const Slot slots[]) const {
    assert(num_slots == 1);
    using namespace std::complex_literals;
     #ifndef AVX512
      base[1ul << slots[0]] *= std::sqrt(1.0i);
    #else
       int64_t pos_offset = 1ul << slots[0];
      if (pos_offset == 1){
        __m512d avx_t_0 = _mm512_load_pd((void const*)base);
        __m512d avx_t_dst_0 = _mm512_permutex_pd(avx_t_0,180);
         //std::cout <<avx_x_0[0] << avx_x_0[1] <<  avx_x_0[2] <<  avx_x_0[3]
         //<<avx_x_0[4] << avx_x_0[5] <<  avx_x_0[6] <<  avx_x_0[7] << std::endl;
         //std::cout <<avx_x_dst_0[0] << avx_x_dst_0[1] <<  avx_x_dst_0[2] <<  avx_x_dst_0[3]
         //<<avx_x_dst_0[4] << avx_x_dst_0[5] <<  avx_x_dst_0[6] <<  avx_x_dst_0[7] << std::endl;
        //__m512d avx_t_timer = _mm512_set4_pd(1.0,-1.0,0,0);
        __m512d avx_t_dst_mul = _mm512_mul_pd(avx_t_dst_0,avx_t_timer);
        __m512d avx_t_tmp1 = _mm512_add_pd(avx_t_0,avx_t_dst_mul);

         //std::cout <<avx_x_dst_0[0] << avx_x_dst_0[1] <<  avx_x_dst_0[2] <<  avx_x_dst_0[3]
         //<<avx_x_dst_0[4] << avx_x_dst_0[5] <<  avx_x_dst_0[6] <<  avx_x_dst_0[7] << std::endl;
        //__m512d avx_t_timer2 = _mm512_set4_pd(Constants::inv_sqrt_2,Constants::inv_sqrt_2,Constants::inv_sqrt_2,Constants::inv_sqrt_2);
        __m512d avx_t_tmp2 = _mm512_mul_pd(avx_t_tmp1,avx_t_timer2);
        _mm512_store_pd((void *)base,avx_t_tmp2);
      }
      else{
        int64_t __attribute__((aligned(64))) i_index_ar[8] = {0,1,0+2*pos_offset,1+2*pos_offset,2,3,2+2*pos_offset,3+2*pos_offset};
        __m512i i_index = _mm512_load_epi64(i_index_ar);
        __m512d avx_t_t = _mm512_i64gather_pd(i_index, (void const *)base,sizeof(double));
         __m512d avx_t_dst_0 = _mm512_permutex_pd(avx_t_t,180);
         //std::cout <<avx_x_0[0] << avx_x_0[1] <<  avx_x_0[2] <<  avx_x_0[3]
         //<<avx_x_0[4] << avx_x_0[5] <<  avx_x_0[6] <<  avx_x_0[7] << std::endl;
         //std::cout <<avx_x_dst_0[0] << avx_x_dst_0[1] <<  avx_x_dst_0[2] <<  avx_x_dst_0[3]
         //<<avx_x_dst_0[4] << avx_x_dst_0[5] <<  avx_x_dst_0[6] <<  avx_x_dst_0[7] << std::endl;
       // __m512d avx_t_timer = _mm512_set4_pd(1.0,-1.0,0,0);
        __m512d avx_t_dst_mul = _mm512_mul_pd(avx_t_dst_0,avx_t_timer);
        __m512d avx_t_tmp1 = _mm512_add_pd(avx_t_t,avx_t_dst_mul);

         //std::cout <<avx_x_dst_0[0] << avx_x_dst_0[1] <<  avx_x_dst_0[2] <<  avx_x_dst_0[3]
         //<<avx_x_dst_0[4] << avx_x_dst_0[5] <<  avx_x_dst_0[6] <<  avx_x_dst_0[7] << std::endl;
       // __m512d avx_t_timer2 = _mm512_set4_pd(Constants::inv_sqrt_2,Constants::inv_sqrt_2,Constants::inv_sqrt_2,Constants::inv_sqrt_2);
        __m512d avx_t_tmp2 = _mm512_mul_pd(avx_t_tmp1,avx_t_timer2);
        _mm512_i64scatter_pd((void *)base, i_index, avx_t_tmp2, sizeof(double));
      }
    #endif
    
  }
};

struct RX {
  RX() = default;
  RX(Type theta) : theta_{theta} {
    using namespace std::complex_literals;
    cos_ = std::cos(theta_ / Type{2});
    sin_ = 1.0i * std::sin(theta_ / Type{2});
    #ifdef AVX512
      avx_rx_timer1_ = _mm512_set4_pd(cos_.real(),cos_.real(),cos_.real(),cos_.real());
      avx_rx_timer2_ = _mm512_set4_pd(-sin_.imag(),sin_.imag(),-sin_.imag(),sin_.imag());
    #endif
  }

  constexpr GateType gate_type() const { return GateType::RX; }
  void operator()(Type* base, size_t num_slots, const Slot slots[]) const {
    assert(num_slots == 1);
    Type ct_2 = cos_;
    Type ist_2 = sin_;
   
    #ifndef AVX512
      auto v0 = base[0ul << slots[0]];
      auto v1 = base[1ul << slots[0]];
      base[0ul << slots[0]] = ct_2 * v0 - ist_2 * v1;
      base[1ul << slots[0]] = -ist_2 * v0 + ct_2 * v1;
    #else
      int64_t pos_offset = 1ul << slots[0];
      if (pos_offset == 1){
        __m512d avx_rx_0 = _mm512_load_pd((void const*)base);
         __m512d avx_rx_1 = _mm512_permutex_pd(avx_rx_0,27);
       // __m512d avx_rx_timer1 = _mm512_set4_pd(ct_2.real(),ct_2.real(),ct_2.real(),ct_2.real());
        __m512d avx_rx_tmp1 = _mm512_mul_pd(avx_rx_0,avx_rx_timer1_);
       // __m512d avx_rx_timer2 = _mm512_set4_pd(-ist_2.imag(),ist_2.imag(),-ist_2.imag(),ist_2.imag());
        __m512d avx_rx_tmp2 = _mm512_mul_pd(avx_rx_1,avx_rx_timer2_);
        __m512d avx_rx_tmp3 = _mm512_add_pd(avx_rx_tmp1,avx_rx_tmp2);
        _mm512_store_pd((void *)base,avx_rx_tmp3);
      }
      else{
        int64_t __attribute__((aligned(64))) i_index_ar[8] = {0,1,0+2*pos_offset,1+2*pos_offset,2,3,2+2*pos_offset,3+2*pos_offset};
        __m512i i_index = _mm512_load_epi64(i_index_ar);
        __m512d avx_rx_t = _mm512_i64gather_pd(i_index, (void const *)base,sizeof(double));
        __m512d avx_rx_1 = _mm512_permutex_pd(avx_rx_t,27);
        //__m512d avx_rx_timer1 = _mm512_set4_pd(ct_2.real(),ct_2.real(),ct_2.real(),ct_2.real());
        __m512d avx_rx_tmp1 = _mm512_mul_pd(avx_rx_t,avx_rx_timer1_);
        //__m512d avx_rx_timer2 = _mm512_set4_pd(-ist_2.imag(),ist_2.imag(),-ist_2.imag(),ist_2.imag());
        __m512d avx_rx_tmp2 = _mm512_mul_pd(avx_rx_1,avx_rx_timer2_);
        __m512d avx_rx_tmp3 = _mm512_add_pd(avx_rx_tmp1,avx_rx_tmp2);
        _mm512_i64scatter_pd((void *)base, i_index, avx_rx_tmp3, sizeof(double));
      }
    #endif
  }
  Type theta_;
  Type cos_;
  Type sin_;
  #ifdef AVX512
    __m512d avx_rx_timer1_; 
    __m512d avx_rx_timer2_;
  #endif
};

struct RY {
  RY() = default;
  RY(Type theta) : theta_{theta} {
    using namespace std::complex_literals;
    cos_ = std::cos(theta_ / Type{2});
    sin_ = std::sin(theta_ / Type{2});
    #ifdef AVX512
      avx_ry_timer1_ = _mm512_set4_pd(cos_.real(),cos_.real(),cos_.real(),cos_.real());
      avx_ry_timer2_ = _mm512_set4_pd(sin_.real(),sin_.real(),-sin_.real(),-sin_.real());
    #endif
  }

  constexpr GateType gate_type() const { return GateType::RY; }
  void operator()(Type* base, size_t num_slots, const Slot slots[]) const {
    assert(num_slots == 1);
    
    Type ct_2 = cos_;
    Type st_2 = sin_;
    
    #ifndef AVX512
      auto v0 = base[0ul << slots[0]];
      auto v1 = base[1ul << slots[0]];
      base[0ul << slots[0]] = ct_2 * v0 - st_2 * v1;
      base[1ul << slots[0]] = st_2 * v0 + ct_2 * v1;
    #else
      int64_t pos_offset = 1ul << slots[0];
      if (pos_offset == 1){ 
        __m512d avx_ry_0 = _mm512_loadu_pd((void const*)base);
         __m512d avx_ry_1 = _mm512_permutex_pd(avx_ry_0,78);
        //__m512d avx_ry_timer1 = _mm512_set4_pd(ct_2.real(),ct_2.real(),ct_2.real(),ct_2.real());
        __m512d avx_ry_tmp1 = _mm512_mul_pd(avx_ry_0,avx_ry_timer1_);
        //__m512d avx_ry_timer2 = _mm512_set4_pd(st_2.real(),st_2.real(),-st_2.real(),-st_2.real());
        __m512d avx_ry_tmp2 = _mm512_mul_pd(avx_ry_1,avx_ry_timer2_);
        __m512d avx_ry_tmp3 = _mm512_add_pd(avx_ry_tmp1,avx_ry_tmp2);
        _mm512_storeu_pd((void *)base,avx_ry_tmp3);
      }
      else{
        int64_t __attribute__((aligned(64))) i_index_ar[8] = {0,1,0+2*pos_offset,1+2*pos_offset,2,3,2+2*pos_offset,3+2*pos_offset};
        __m512i i_index = _mm512_load_epi64(i_index_ar);
        __m512d avx_ry_t = _mm512_i64gather_pd(i_index, (void const *)base,sizeof(double));
        __m512d avx_ry_1 = _mm512_permutex_pd(avx_ry_t,78);
        //__m512d avx_ry_timer1 = _mm512_set4_pd(ct_2.real(),ct_2.real(),ct_2.real(),ct_2.real());
        __m512d avx_ry_tmp1 = _mm512_mul_pd(avx_ry_t,avx_ry_timer1_);
        //__m512d avx_ry_timer2 = _mm512_set4_pd(st_2.real(),st_2.real(),-st_2.real(),-st_2.real());
        __m512d avx_ry_tmp2 = _mm512_mul_pd(avx_ry_1,avx_ry_timer2_);
        __m512d avx_ry_tmp3 = _mm512_add_pd(avx_ry_tmp1,avx_ry_tmp2);
        _mm512_i64scatter_pd((void *)base, i_index, avx_ry_tmp3, sizeof(double));
      }
    #endif
  }
  Type theta_;
  Type cos_;
  Type sin_;
  #ifdef AVX512
    __m512d avx_ry_timer1_;
    __m512d avx_ry_timer2_;
  #endif
};

struct RI {
  RI() = default;
  RI(Type theta) : theta_{theta} {
    using namespace std::complex_literals;
    neg_exp_ = std::exp(-1.0i * theta_ / Type{2});
    #ifdef AVX512
      avx_ri_timer1_ = _mm512_set4_pd(neg_exp_.real(),neg_exp_.real(),neg_exp_.real(),neg_exp_.real());
      avx_ri_timer2_ = _mm512_set4_pd(neg_exp_.imag(),-neg_exp_.imag(),neg_exp_.imag(),-neg_exp_.imag());
    #endif
  }

  constexpr GateType gate_type() const { return GateType::RI; }
  void operator()(Type* base, size_t num_slots, const Slot slots[]) const {
    assert(num_slots == 1);
    

    #ifndef AVX512
      auto v0 = base[0ul << slots[0]];
      auto v1 = base[1ul << slots[0]];
      base[0ul << slots[0]] = v0 * neg_exp_;
      base[1ul << slots[0]] = v1 * neg_exp_;
    #else
      int64_t pos_offset = 1ul << slots[0];
      if (pos_offset == 1){ 
        __m512d avx_ri_0 = _mm512_loadu_pd((void const*)base);
         __m512d avx_ri_1 = _mm512_permutex_pd(avx_ri_0,177);
        //__m512d avx_ri_timer1 = _mm512_set4_pd(neg_exp_.real(),neg_exp_.real(),neg_exp_.real(),neg_exp_.real());
        __m512d avx_ri_tmp1 = _mm512_mul_pd(avx_ri_0,avx_ri_timer1_);
        //__m512d avx_ri_timer2 = _mm512_set4_pd(neg_exp_.imag(),-neg_exp_.imag(),neg_exp_.imag(),-neg_exp_.imag());
        __m512d avx_ri_tmp2 = _mm512_mul_pd(avx_ri_1,avx_ri_timer2_);
        __m512d avx_ri_tmp3 = _mm512_add_pd(avx_ri_tmp1,avx_ri_tmp2);
        _mm512_storeu_pd((void *)base,avx_ri_tmp3);
      }
      else{
        int64_t __attribute__((aligned(64))) i_index_ar[8] = {0,1,0+2*pos_offset,1+2*pos_offset,2,3,2+2*pos_offset,3+2*pos_offset};
        __m512i i_index = _mm512_load_epi64(i_index_ar);
        __m512d avx_ry_t = _mm512_i64gather_pd(i_index, (void const *)base,sizeof(double));
          __m512d avx_ri_1 = _mm512_permutex_pd(avx_ry_t,177);
        //__m512d avx_ri_timer1 = _mm512_set4_pd(neg_exp_.real(),neg_exp_.real(),neg_exp_.real(),neg_exp_.real());
        __m512d avx_ri_tmp1 = _mm512_mul_pd(avx_ry_t,avx_ri_timer1_);
        //__m512d avx_ri_timer2 = _mm512_set4_pd(neg_exp_.imag(),-neg_exp_.imag(),neg_exp_.imag(),-neg_exp_.imag());
        __m512d avx_ri_tmp2 = _mm512_mul_pd(avx_ri_1,avx_ri_timer2_);
        __m512d avx_ri_tmp3 = _mm512_add_pd(avx_ri_tmp1,avx_ri_tmp2);
        _mm512_i64scatter_pd((void *)base, i_index, avx_ri_tmp3, sizeof(double));
      }
    #endif
  }
  Type theta_;
  Type neg_exp_;
  #ifdef AVX512
   __m512d avx_ri_timer1_;
   __m512d avx_ri_timer2_;
  #endif
};

struct RZ {
  RZ() = default;
  RZ(Type theta) : theta_{theta} {
    using namespace std::complex_literals;
    neg_exp_ = std::exp(-1.0i * theta_ / Type{2});
    pos_exp_ = std::exp(+1.0i * theta_ / Type{2});
    #ifdef AVX512
      avx_rz_timer1_ = _mm512_set4_pd(pos_exp_.real(),pos_exp_.real(),neg_exp_.real(),neg_exp_.real());
      avx_rz_timer2_ = _mm512_set4_pd(pos_exp_.imag(),-pos_exp_.imag(),neg_exp_.imag(),-neg_exp_.imag());
    #endif
  }

  constexpr GateType gate_type() const { return GateType::RZ; }
  void operator()(Type* base, size_t num_slots, const Slot slots[]) const {
    assert(num_slots == 1);
    
   
    #ifndef AVX512
      auto v0 = base[0ul << slots[0]];
      auto v1 = base[1ul << slots[0]];
      // fix this 
      base[0ul << slots[0]] = v0 * neg_exp_;
      base[1ul << slots[0]] = v1 * pos_exp_;
    #else
      int64_t pos_offset = 1ul << slots[0];
      if (pos_offset == 1){ 
        __m512d avx_rz_0 = _mm512_loadu_pd((void const*)base);
         __m512d avx_rz_1 = _mm512_permutex_pd(avx_rz_0,177);
        //__m512d avx_rz_timer1 = _mm512_set4_pd(pos_exp_.real(),pos_exp_.real(),neg_exp_.real(),neg_exp_.real());
        __m512d avx_rz_tmp1 = _mm512_mul_pd(avx_rz_0,avx_rz_timer1_);
        //__m512d avx_rz_timer2 = _mm512_set4_pd(pos_exp_.imag(),-pos_exp_.imag(),neg_exp_.imag(),-neg_exp_.imag());
        __m512d avx_rz_tmp2 = _mm512_mul_pd(avx_rz_1,avx_rz_timer2_);
        __m512d avx_rz_tmp3 = _mm512_add_pd(avx_rz_tmp1,avx_rz_tmp2);
        _mm512_storeu_pd((void *)base,avx_rz_tmp3);
      }
      else{
        int64_t __attribute__((aligned(64))) i_index_ar[8] = {0,1,0+2*pos_offset,1+2*pos_offset,2,3,2+2*pos_offset,3+2*pos_offset};
        __m512i i_index = _mm512_load_epi64(i_index_ar);
        __m512d avx_rz_t = _mm512_i64gather_pd(i_index, (void const *)base,sizeof(double));
        __m512d avx_rz_1 = _mm512_permutex_pd(avx_rz_t,177);
        //__m512d avx_rz_timer1 = _mm512_set4_pd(pos_exp_.real(),pos_exp_.real(),neg_exp_.real(),neg_exp_.real());
        __m512d avx_rz_tmp1 = _mm512_mul_pd(avx_rz_t,avx_rz_timer1_);
        //__m512d avx_rz_timer2 = _mm512_set4_pd(pos_exp_.imag(),-pos_exp_.imag(),neg_exp_.imag(),-neg_exp_.imag());
        __m512d avx_rz_tmp2 = _mm512_mul_pd(avx_rz_1,avx_rz_timer2_);
        __m512d avx_rz_tmp3 = _mm512_add_pd(avx_rz_tmp1,avx_rz_tmp2);
        _mm512_i64scatter_pd((void *)base, i_index, avx_rz_tmp3, sizeof(double));
      }
    #endif
  }
  Type theta_;
  Type neg_exp_;
  Type pos_exp_;
  #ifdef AVX512
    __m512d avx_rz_timer1_;
    __m512d avx_rz_timer2_;
  #endif
};

using U1 = RZ;

struct RXX {
  RXX() = default;
  //RXX(Type theta) : theta_{theta} {}
  RXX(Type theta) : h1_{}, h2_{}, h3_{}, h4_{}, rz_{theta} {}
  constexpr GateType gate_type() const { return GateType::RXX; }
  void operator()(Type* base, size_t num_slots, const Slot slots[]) const {
    assert(num_slots == 2);
    auto v0 = base[(0ul << slots[0]) + (0ul << slots[1])];
    auto v1 = base[(0ul << slots[0]) + (1ul << slots[1])];
    auto v2 = base[(1ul << slots[0]) + (0ul << slots[1])];
    auto v3 = base[(1ul << slots[0]) + (1ul << slots[1])];
    int pos_qubit_0 = 1ul << slots[0];
    int pos_qubit_1 = 1ul << slots[1];
    #ifndef AVX512
      h1_(base,1,&slots[0]);
      h1_(base+(1ul << slots[1]),1,&slots[0]);
      h2_(base,1,&slots[1]);
      h2_(base+(1ul << slots[0]),1,&slots[1]);
      h2_(base,1,&slots[1]);
      std::swap(base[(1ul << slots[0]) + (0ul << slots[1])], base[(1ul << slots[0]) + (1ul << slots[1])]);
      rz_(base,1,&slots[1]);
      rz_(base+(1ul << slots[0]),1,&slots[1]);
      std::swap(base[(1ul << slots[0]) + (0ul << slots[1])], base[(1ul << slots[0]) + (1ul << slots[1])]);
      h3_(base,1,&slots[1]);
      h3_(base+(1ul << slots[0]),1,&slots[1]);
      h4_(base,1,&slots[0]);
      h4_(base+(1ul << slots[1]),1,&slots[0]);
    #else
      //std::cout << "first h" << std::endl;
      if ((slots[0] == 0 && slots[1] == 1) || (slots[1] == 0 && slots[0] ==1)){
        h1_(base,1,&slots[0]);
        h2_(base,1,&slots[1]);
      }
      else 
      {
        if (slots[0] == 0){
          h1_(base,1,&slots[0]);
          h1_(base+(1ul << slots[1]),1,&slots[0]);
          h2_(base,1,&slots[1]);
        }
        else if (slots[1] == 0){
          h1_(base,1,&slots[0]);
          h2_(base,1,&slots[1]);
          h2_(base+(1ul << slots[0]),1,&slots[1]);
        }
        else{
          h1_(base,1,&slots[0]);
          h1_(base+(1ul << slots[1]),1,&slots[0]);
          h2_(base,1,&slots[1]);
          h2_(base+(1ul << slots[0]),1,&slots[1]);
        }
      }

      if ((slots[0] == 0 && slots[1] == 1) || (slots[0] == 1 && slots[1] == 0)){
        int64_t __attribute__((aligned(64))) i_index_ar[8] = {2*pos_qubit_0,2*pos_qubit_0+1,2*(pos_qubit_0+pos_qubit_1),1+2*(pos_qubit_1+pos_qubit_0),(pos_qubit_0+4)*2,(pos_qubit_0+4)*2+1,(pos_qubit_1+pos_qubit_0+4)*2,(pos_qubit_1+pos_qubit_0+4)*2+1};
        __m512i i_index = _mm512_load_epi64(i_index_ar);
        __m512d avx_swap_t = _mm512_i64gather_pd(i_index, (void const *)base,sizeof(double));
   //      std::cout <<"in 0 case " <<avx_swap_t[0] << " " << avx_swap_t[1] <<" "<<  avx_swap_t[2]<<" " <<  avx_swap_t[3]<<" "
   //     <<avx_swap_t[4] <<" "<< avx_swap_t[5] <<" "<<  avx_swap_t[6]<<" " <<  avx_swap_t[7] << std::endl;
        __m512d avx_swap_1 = _mm512_permutex_pd(avx_swap_t,78);
         _mm512_i64scatter_pd((void *)base, i_index, avx_swap_1, sizeof(double));
      }
      else{
        if (std::abs(slots[0]-slots[1]) == 1){
          int64_t __attribute__((aligned(64))) i_index_ar[8] = {2*pos_qubit_0,2*pos_qubit_0+1,2*(pos_qubit_1+pos_qubit_0),1+2*(pos_qubit_1+pos_qubit_0),2+pos_qubit_0*2,pos_qubit_0*2+3,(pos_qubit_1 + pos_qubit_0)*2+2,(pos_qubit_1+pos_qubit_0)*2+3};
          __m512i i_index = _mm512_load_epi64(i_index_ar);
          __m512d avx_swap_t = _mm512_i64gather_pd(i_index, (void const *)base,sizeof(double));
          __m512d avx_swap_1 = _mm512_permutex_pd(avx_swap_t,78);
          _mm512_i64scatter_pd((void *)base, i_index, avx_swap_1, sizeof(double));
        }
        else{
          int64_t __attribute__((aligned(64))) i_index_ar[8] = {2*pos_qubit_0,2*pos_qubit_0+1,2*(pos_qubit_1+pos_qubit_0),1+2*(pos_qubit_1+pos_qubit_0),4+pos_qubit_0*2,pos_qubit_0*2+5,(pos_qubit_1+pos_qubit_0)*2+4,(pos_qubit_1 + pos_qubit_0)*2+5};
          __m512i i_index = _mm512_load_epi64(i_index_ar);
          __m512d avx_swap_t = _mm512_i64gather_pd(i_index, (void const *)base,sizeof(double));
          __m512d avx_swap_1 = _mm512_permutex_pd(avx_swap_t,78);
    //       std::cout <<"in 2 case " <<avx_swap_t[0] << " " << avx_swap_t[1] <<" "<<  avx_swap_t[2]<<" " <<  avx_swap_t[3]<<" "
    //    <<avx_swap_t[4] <<" "<< avx_swap_t[5] <<" "<<  avx_swap_t[6]<<" " <<  avx_swap_t[7] << std::endl;
          _mm512_i64scatter_pd((void *)base, i_index, avx_swap_1, sizeof(double));
        }
      }
      rz_(base,1,&slots[1]);
      if (slots[0] != 0 && slots[1] != 0)
        rz_(base+(1ul << slots[0]),1,&slots[1]);
     if ((slots[0] == 0 && slots[1] == 1) || (slots[0] == 1 && slots[1] == 0)){
        int64_t __attribute__((aligned(64))) i_index_ar[8] = {2*pos_qubit_0,2*pos_qubit_0+1,2*(pos_qubit_0+pos_qubit_1),1+2*(pos_qubit_1+pos_qubit_0),(pos_qubit_0+4)*2,(pos_qubit_0+4)*2+1,(pos_qubit_1+pos_qubit_0+4)*2,(pos_qubit_1+pos_qubit_0+4)*2+1};
        __m512i i_index = _mm512_load_epi64(i_index_ar);
        __m512d avx_swap_t = _mm512_i64gather_pd(i_index, (void const *)base,sizeof(double));
        __m512d avx_swap_1 = _mm512_permutex_pd(avx_swap_t,78);
         _mm512_i64scatter_pd((void *)base, i_index, avx_swap_1, sizeof(double));
      }
      else{
        if (std::abs(slots[0]-slots[1]) == 1){
          int64_t __attribute__((aligned(64))) i_index_ar[8] = {2*pos_qubit_0,2*pos_qubit_0+1,2*(pos_qubit_1+pos_qubit_0),1+2*(pos_qubit_1+pos_qubit_0),2+pos_qubit_0*2,pos_qubit_0*2+3,(pos_qubit_1 + pos_qubit_0)*2+2,(pos_qubit_1+pos_qubit_0)*2+3};
          __m512i i_index = _mm512_load_epi64(i_index_ar);
          __m512d avx_swap_t = _mm512_i64gather_pd(i_index, (void const *)base,sizeof(double));
          __m512d avx_swap_1 = _mm512_permutex_pd(avx_swap_t,78);
          _mm512_i64scatter_pd((void *)base, i_index, avx_swap_1, sizeof(double));
        }
        else{
          int64_t __attribute__((aligned(64))) i_index_ar[8] = {2*pos_qubit_0,2*pos_qubit_0+1,2*(pos_qubit_1+pos_qubit_0),1+2*(pos_qubit_1+pos_qubit_0),4+pos_qubit_0*2,pos_qubit_0*2+5,(pos_qubit_1+pos_qubit_0)*2+4,(pos_qubit_1 + pos_qubit_0)*2+5};
          __m512i i_index = _mm512_load_epi64(i_index_ar);
          __m512d avx_swap_t = _mm512_i64gather_pd(i_index, (void const *)base,sizeof(double));
          __m512d avx_swap_1 = _mm512_permutex_pd(avx_swap_t,78);
      //      std::cout <<"in 2 case " <<avx_swap_t[0] << " " << avx_swap_t[1] <<" "<<  avx_swap_t[2]<<" " <<  avx_swap_t[3]<<" "
     //   <<avx_swap_t[4] <<" "<< avx_swap_t[5] <<" "<<  avx_swap_t[6]<<" " <<  avx_swap_t[7] << std::endl;
          _mm512_i64scatter_pd((void *)base, i_index, avx_swap_1, sizeof(double));
        }
      }
       if ((slots[0] == 0 && slots[1] == 1) || (slots[1] == 0 && slots[0] ==1)){
        h3_(base,1,&slots[1]);
        h4_(base,1,&slots[0]);
      }
      else 
      {
        if (slots[0] == 0){
          h3_(base,1,&slots[1]);   
          h4_(base,1,&slots[0]);
          h4_(base+(1ul << slots[1]),1,&slots[0]);
        }
        else if (slots[1] == 0){
          h3_(base,1,&slots[1]);
          h3_(base+(1ul << slots[0]),1,&slots[1]);
          h4_(base,1,&slots[0]);
        }
        else{
          h3_(base,1,&slots[1]);
          h3_(base+(1ul << slots[0]),1,&slots[1]);
          h4_(base,1,&slots[0]);
          h4_(base+(1ul << slots[1]),1,&slots[0]);
        }
      }
    #endif

  }
  Type theta_;
  H h1_;
  H h2_;
  H h3_;
  H h4_;
  RZ rz_;
};


struct R1 {
  R1() = default;
  R1(Type theta) : rz_{theta}, ri_{-theta} {}

  constexpr GateType gate_type() const { return GateType::R1; }
  void operator()(Type* base, size_t num_slots, const Slot slots[]) const {
    assert(num_slots == 1);
    rz_(base, 1, slots);
    ri_(base, 1, slots);
  }
  RZ rz_;
  RI ri_;
};

struct RZFrac {
  RZFrac() = default;
  RZFrac(unsigned numerator, unsigned power)
      : rz_{-Constants::pi * numerator / std::pow(2, power - 1)} {}

  constexpr GateType gate_type() const { return GateType::RZFrac; }
  void operator()(Type* base, size_t num_slots, const Slot slots[]) const {
    assert(num_slots == 1);
    return rz_(base, 1, slots);
  }
  RZ rz_;
};

struct RXFrac {
  RXFrac() = default;
  RXFrac(unsigned numerator, unsigned power)
      : rx_{-Constants::pi * numerator / std::pow(2, power - 1)} {}

  constexpr GateType gate_type() const { return GateType::RXFrac; }
  void operator()(Type* base, size_t num_slots, const Slot slots[]) const {
    assert(num_slots == 1);
    return rx_(base, 1, slots);
  }
  RX rx_;
};

struct RYFrac {
  RYFrac() = default;
  RYFrac(unsigned numerator, unsigned power)
      : ry_{-Constants::pi * numerator / std::pow(2, power - 1)} {}

  constexpr GateType gate_type() const { return GateType::RYFrac; }
  void operator()(Type* base, size_t num_slots, const Slot slots[]) const {
    assert(num_slots == 1);
    return ry_(base, 1, slots);
  }
  RY ry_;
};

struct RIFrac {
  RIFrac() = default;
  RIFrac(unsigned numerator, unsigned power)
      : ri_{-Constants::pi * numerator / std::pow(2, power - 1)} {}

  constexpr GateType gate_type() const { return GateType::RIFrac; }
  void operator()(Type* base, size_t num_slots, const Slot slots[]) const {
    assert(num_slots == 1);
    return ri_(base, 1, slots);
  }
  RI ri_;
};

struct R1Frac {
  R1Frac() = default;
  R1Frac(unsigned numerator, unsigned power)
      : rzf_{-numerator, power + 1}, rif_{numerator, power + 1} {}

  constexpr GateType gate_type() const { return GateType::R1Frac; }
  void operator()(Type* base, size_t num_slots, const Slot slots[]) const {
    assert(num_slots == 1);
    rzf_(base, 1, slots);
    rif_(base, 1, slots);
  }
  RZFrac rzf_;
  RIFrac rif_;
};

struct SWAP {
  SWAP() = default;

  constexpr GateType gate_type() const { return GateType::SWAP; }
  void operator()(Type* base, size_t num_slots, const Slot slots[]) const {
    assert(num_slots == 2);
    #ifndef AVX512
      std::swap(base[(1ul << slots[0]) + (0ul << slots[1])],
                base[(0ul << slots[0]) + (1ul << slots[1])]);
      //std::cout << "swaping " << (1ul << slots[0]) << " and " << (1ul << slots[1]) << std::endl;
    #else
      int pos_qubit_0 = 1ul << slots[0];
      int pos_qubit_1 = 1ul << slots[1];
      if ((slots[0] == 0 && slots[1] == 1) || (slots[0] == 1 && slots[1] == 0)){
        int64_t __attribute__((aligned(64))) i_index_ar[8] = {2*pos_qubit_0,2*pos_qubit_0+1,2*pos_qubit_1,1+2*pos_qubit_1,(pos_qubit_0+4)*2,(pos_qubit_0+4)*2+1,(pos_qubit_1+4)*2,(pos_qubit_1+4)*2+1};
        __m512i i_index = _mm512_load_epi64(i_index_ar);
        __m512d avx_swap_t = _mm512_i64gather_pd(i_index, (void const *)base,sizeof(double));
        __m512d avx_swap_1 = _mm512_permutex_pd(avx_swap_t,78);
         _mm512_i64scatter_pd((void *)base, i_index, avx_swap_1, sizeof(double));
      }
      else{
        if (std::abs(slots[0]-slots[1]) == 1){
          int64_t __attribute__((aligned(64))) i_index_ar[8] = {2*pos_qubit_0,2*pos_qubit_0+1,2*pos_qubit_1,1+2*pos_qubit_1,2+pos_qubit_0*2,pos_qubit_0*2+3,pos_qubit_1*2+2,pos_qubit_1*2+3};
          __m512i i_index = _mm512_load_epi64(i_index_ar);
          __m512d avx_swap_t = _mm512_i64gather_pd(i_index, (void const *)base,sizeof(double));
          __m512d avx_swap_1 = _mm512_permutex_pd(avx_swap_t,78);
          _mm512_i64scatter_pd((void *)base, i_index, avx_swap_1, sizeof(double));
        }
        else{
          int64_t __attribute__((aligned(64))) i_index_ar[8] = {2*pos_qubit_0,2*pos_qubit_0+1,2*pos_qubit_1,1+2*pos_qubit_1,4+pos_qubit_0*2,pos_qubit_0*2+5,pos_qubit_1*2+4,pos_qubit_1*2+5};
          __m512i i_index = _mm512_load_epi64(i_index_ar);
          __m512d avx_swap_t = _mm512_i64gather_pd(i_index, (void const *)base,sizeof(double));
          __m512d avx_swap_1 = _mm512_permutex_pd(avx_swap_t,78);
          _mm512_i64scatter_pd((void *)base, i_index, avx_swap_1, sizeof(double));
        }
      }
    #endif
  }
};

struct C1 {
  C1() = default;

  C1(Type c00, Type c01, Type c10, Type c11)
      : c00_{c00}, c01_{c01}, c10_{c10}, c11_{c11} {
        #ifdef AVX512
          avx_c1_timer1_ = _mm512_set4_pd(c01_.real(),c01_.real(),c00_.real(),c00_.real());
          avx_c1_timer2_ = _mm512_set4_pd(c01_.imag(),-c01_.imag(),c00_.imag(),-c00_.imag());
          avx_c1_timer3_ = _mm512_set4_pd(c11_.real(),c11_.real(),c10_.real(),c10_.real());
          avx_c1_timer4_ = _mm512_set4_pd(c11_.imag(),-c11_.imag(),c10_.imag(),-c10_.imag());
        #endif
      }

  constexpr GateType gate_type() const { return GateType::C1; }
  void operator()(Type* base, size_t num_slots, const Slot slots[]) const {
    assert(num_slots == 1);
  
    #ifndef AVX512
      auto v0 = base[0ul << slots[0]];
      auto v1 = base[1ul << slots[0]];
      base[0ul << slots[0]] = c00_ * v0 + c01_ * v1;
      base[1ul << slots[0]] = c10_ * v0 + c11_ * v1;
    #else
      int64_t pos_offset = 1ul << slots[0];
      if (pos_offset == 1){ 
     //   std::cout << "get into here" << std::endl;
        __m512d avx_c1_0 = _mm512_load_pd((void const*)base);
        __m512d avx_c1_1 = _mm512_permutex_pd(avx_c1_0,177);
        //__m512d avx_c1_timer1 = _mm512_set4_pd(c01_.real(),c01_.real(),c00_.real(),c00_.real());
        __m512d avx_c1_tmp1 = _mm512_mul_pd(avx_c1_0,avx_c1_timer1_);
        //__m512d avx_c1_timer2 = _mm512_set4_pd(c01_.imag(),-c01_.imag(),c00_.imag(),-c00_.imag());
        __m512d avx_c1_tmp2 = _mm512_mul_pd(avx_c1_1,avx_c1_timer2_);
        __m512d avx_c1_tmp3 = _mm512_add_pd(avx_c1_tmp1,avx_c1_tmp2);
        __m128d avx_c1_128_a = _mm512_extractf64x2_pd(avx_c1_tmp3,0);
     //    std::cout << avx_c1_128_a[0] << " " << avx_c1_128_a[1] << std::endl;
        __m128d avx_c1_128_b = _mm512_extractf64x2_pd(avx_c1_tmp3,1);
//std::cout << avx_c1_128_b[0] << " " << avx_c1_128_b[1] << std::endl;
        __m128d avx_c1_128_c = _mm512_extractf64x2_pd(avx_c1_tmp3,2);
   //     std::cout << avx_c1_128_c[0] << " " << avx_c1_128_c[1] << std::endl;
        __m128d avx_c1_128_d = _mm512_extractf64x2_pd(avx_c1_tmp3,3);
    //    std::cout << avx_c1_128_d[0] << " " << avx_c1_128_d[1] << std::endl;
        __m128d avx_c1_128_0 = _mm_add_pd(avx_c1_128_a,avx_c1_128_b);
        __m128d avx_c1_128_2 = _mm_add_pd(avx_c1_128_c,avx_c1_128_d);
    //    std::cout << avx_c1_128_0[0] << " " << avx_c1_128_0[1] << std::endl;
     //   std::cout << avx_c1_128_2[0] << " " << avx_c1_128_2[1] << std::endl;
        __m512d avx_c1_2 = avx_c1_1;
        // __m512d avx_c1_timer3 = _mm512_set4_pd(c11_.real(),c11_.real(),c10_.real(),c10_.real());
        __m512d avx_c1_tmp4 = _mm512_mul_pd(avx_c1_0,avx_c1_timer3_);
       // __m512d avx_c1_timer4 = _mm512_set4_pd(c11_.imag(),-c11_.imag(),c10_.imag(),-c10_.imag());
        __m512d avx_c1_tmp5 = _mm512_mul_pd(avx_c1_2,avx_c1_timer4_);
        __m512d avx_c1_tmp6 = _mm512_add_pd(avx_c1_tmp4,avx_c1_tmp5);
        __m128d avx_c1_128_e = _mm512_extractf64x2_pd(avx_c1_tmp6,0);
        __m128d avx_c1_128_f = _mm512_extractf64x2_pd(avx_c1_tmp6,1);
        __m128d avx_c1_128_g = _mm512_extractf64x2_pd(avx_c1_tmp6,2);
        __m128d avx_c1_128_h = _mm512_extractf64x2_pd(avx_c1_tmp6,3);
        __m128d avx_c1_128_1 = _mm_add_pd(avx_c1_128_e,avx_c1_128_f);
        __m128d avx_c1_128_3 = _mm_add_pd(avx_c1_128_g,avx_c1_128_h);
     //   std::cout << avx_c1_128_1[0] << " " << avx_c1_128_1[1] << std::endl;
     //   std::cout << avx_c1_128_3[0] << " " << avx_c1_128_3[1] << std::endl;
        __m512d avx_c1_tmp7 = _mm512_setzero_pd(); 
        avx_c1_tmp7 = _mm512_mask_broadcast_f64x2(avx_c1_tmp7,3,avx_c1_128_0);
       //  std::cout << avx_c1_tmp7[0] << " " << avx_c1_tmp7[1] << std::endl;
        avx_c1_tmp7 = _mm512_mask_broadcast_f64x2(avx_c1_tmp7,12,avx_c1_128_1);
    //     std::cout << avx_c1_tmp7[2] << " " << avx_c1_tmp7[3] << std::endl;
        avx_c1_tmp7 = _mm512_mask_broadcast_f64x2(avx_c1_tmp7,48,avx_c1_128_2);
    //     std::cout << avx_c1_tmp7[4] << " " << avx_c1_tmp7[5] << std::endl;
        avx_c1_tmp7 = _mm512_mask_broadcast_f64x2(avx_c1_tmp7,192,avx_c1_128_3);
      //   std::cout << avx_c1_tmp7[6] << " " << avx_c1_tmp7[7] << std::endl;
        _mm512_store_pd((void *)base,avx_c1_tmp7);
      }
      else{
        
        int64_t __attribute__((aligned(64))) i_index_ar[8] = {0,1,0+2*pos_offset,1+2*pos_offset,2,3,2+2*pos_offset,3+2*pos_offset};
        __m512i i_index = _mm512_load_epi64(i_index_ar);
        __m512d avx_c1_t = _mm512_i64gather_pd(i_index, (void const *)base,sizeof(double));
        __m512d avx_c1_1 = _mm512_permutex_pd(avx_c1_t,177);
        //__m512d avx_c1_timer1 = _mm512_set4_pd(c01_.real(),c01_.real(),c00_.real(),c00_.real());
        __m512d avx_c1_tmp1 = _mm512_mul_pd(avx_c1_t,avx_c1_timer1_);
        //__m512d avx_c1_timer2 = _mm512_set4_pd(c01_.imag(),-c01_.imag(),c00_.imag(),-c00_.imag());
        __m512d avx_c1_tmp2 = _mm512_mul_pd(avx_c1_1,avx_c1_timer2_);
        __m512d avx_c1_tmp3 = _mm512_add_pd(avx_c1_tmp1,avx_c1_tmp2);
        __m128d avx_c1_128_a = _mm512_extractf64x2_pd(avx_c1_tmp3,0);
  //       std::cout << avx_c1_128_a[0] << " " << avx_c1_128_a[1] << std::endl;
        __m128d avx_c1_128_b = _mm512_extractf64x2_pd(avx_c1_tmp3,1);
   //     std::cout << avx_c1_128_b[0] << " " << avx_c1_128_b[1] << std::endl;
        __m128d avx_c1_128_c = _mm512_extractf64x2_pd(avx_c1_tmp3,2);
    //    std::cout << avx_c1_128_c[0] << " " << avx_c1_128_c[1] << std::endl;
        __m128d avx_c1_128_d = _mm512_extractf64x2_pd(avx_c1_tmp3,3);
    //    std::cout << avx_c1_128_d[0] << " " << avx_c1_128_d[1] << std::endl;
        __m128d avx_c1_128_0 = _mm_add_pd(avx_c1_128_a,avx_c1_128_b);
        __m128d avx_c1_128_2 = _mm_add_pd(avx_c1_128_c,avx_c1_128_d);
   //     std::cout << avx_c1_128_0[0] << " " << avx_c1_128_0[1] << std::endl;
   //     std::cout << avx_c1_128_2[0] << " " << avx_c1_128_2[1] << std::endl;
        __m512d avx_c1_2 = avx_c1_1;
        // __m512d avx_c1_timer3 = _mm512_set4_pd(c11_.real(),c11_.real(),c10_.real(),c10_.real());
        __m512d avx_c1_tmp4 = _mm512_mul_pd(avx_c1_t,avx_c1_timer3_);
        //__m512d avx_c1_timer4 = _mm512_set4_pd(c11_.imag(),-c11_.imag(),c10_.imag(),-c10_.imag());
        __m512d avx_c1_tmp5 = _mm512_mul_pd(avx_c1_2,avx_c1_timer4_);
        __m512d avx_c1_tmp6 = _mm512_add_pd(avx_c1_tmp4,avx_c1_tmp5);
        __m128d avx_c1_128_e = _mm512_extractf64x2_pd(avx_c1_tmp6,0);
        __m128d avx_c1_128_f = _mm512_extractf64x2_pd(avx_c1_tmp6,1);
        __m128d avx_c1_128_g = _mm512_extractf64x2_pd(avx_c1_tmp6,2);
        __m128d avx_c1_128_h = _mm512_extractf64x2_pd(avx_c1_tmp6,3);
        __m128d avx_c1_128_1 = _mm_add_pd(avx_c1_128_e,avx_c1_128_f);
        __m128d avx_c1_128_3 = _mm_add_pd(avx_c1_128_g,avx_c1_128_h);
     //   std::cout << avx_c1_128_1[0] << " " << avx_c1_128_1[1] << std::endl;
    //    std::cout << avx_c1_128_3[0] << " " << avx_c1_128_3[1] << std::endl;
        __m512d avx_c1_tmp7 = _mm512_setzero_pd(); 
        avx_c1_tmp7 = _mm512_mask_broadcast_f64x2(avx_c1_tmp7,3,avx_c1_128_0);
      //   std::cout << avx_c1_tmp7[0] << " " << avx_c1_tmp7[1] << std::endl;
        avx_c1_tmp7 = _mm512_mask_broadcast_f64x2(avx_c1_tmp7,12,avx_c1_128_1);
      //   std::cout << avx_c1_tmp7[2] << " " << avx_c1_tmp7[3] << std::endl;
        avx_c1_tmp7 = _mm512_mask_broadcast_f64x2(avx_c1_tmp7,48,avx_c1_128_2);
      //   std::cout << avx_c1_tmp7[4] << " " << avx_c1_tmp7[5] << std::endl;
        avx_c1_tmp7 = _mm512_mask_broadcast_f64x2(avx_c1_tmp7,192,avx_c1_128_3);
      //   std::cout << avx_c1_tmp7[6] << " " << avx_c1_tmp7[7] << std::endl;
        _mm512_i64scatter_pd((void *)base, i_index, avx_c1_tmp7, sizeof(double));
      }
    #endif
  }
  Type c00_, c01_, c10_, c11_;
  #ifdef AVX512
     __m512d avx_c1_timer1_;
     __m512d avx_c1_timer2_;
     __m512d avx_c1_timer3_;
     __m512d avx_c1_timer4_;
  #endif
};

struct C2 {
  C2() = default;

  C2(Type c00, Type c01, Type c02, Type c03, Type c10, Type c11, Type c12,
     Type c13, Type c20, Type c21, Type c22, Type c23, Type c30, Type c31,
     Type c32, Type c33)
      : c00_{c00},
        c01_{c00},
        c02_{c00},
        c03_{c00},
        c10_{c10},
        c11_{c10},
        c12_{c10},
        c13_{c10},
        c20_{c20},
        c21_{c20},
        c22_{c20},
        c23_{c20},
        c30_{c30},
        c31_{c30},
        c32_{c30},
        c33_{c30} {}

  constexpr GateType gate_type() const { return GateType::C2; }
  void operator()(Type* base, size_t num_slots, const Slot slots[]) const {
    assert(num_slots == 2);
    auto v0 = base[(0ul << slots[0]) + (0ul << slots[1])];
    auto v1 = base[(0ul << slots[0]) + (1ul << slots[1])];
    auto v2 = base[(1ul << slots[0]) + (0ul << slots[1])];
    auto v3 = base[(1ul << slots[0]) + (1ul << slots[1])];

    base[(0ul << slots[0]) + (0ul << slots[1])] =
        c00_ * v0 + c01_ * v1 + c02_ * v2 + c03_ * v3;
    base[(0ul << slots[0]) + (1ul << slots[1])] =
        c10_ * v0 + c11_ * v1 + c12_ * v2 + c13_ * v3;
    base[(1ul << slots[0]) + (0ul << slots[1])] =
        c20_ * v0 + c21_ * v1 + c22_ * v2 + c23_ * v3;
    base[(1ul << slots[0]) + (1ul << slots[1])] =
        c30_ * v0 + c31_ * v1 + c32_ * v2 + c33_ * v3;
  }

  Type c00_, c01_, c02_, c03_;
  Type c10_, c11_, c12_, c13_;
  Type c20_, c21_, c22_, c23_;
  Type c30_, c31_, c32_, c33_;
};

struct U2 {
  U2() = default;
  U2(Type phi, Type lambda) {
    using namespace std::complex_literals;
    /*Type e0 = Constants::inv_sqrt_2 * (std::cos((-phi - lambda) / 2.0) +
                                       1.0i * sin((-phi - lambda) / 2.0));
    Type e1 = -Constants::inv_sqrt_2 * (std::cos((-phi + lambda) / 2.0) +
                                        1.0i * sin((-phi + lambda) / 2.0));
    Type e2 = Constants::inv_sqrt_2 * (std::cos((phi - lambda) / 2.0) +
                                       1.0i * sin((phi - lambda) / 2.0));
    Type e3 = Constants::inv_sqrt_2 * (std::cos((phi + lambda) / 2.0) +
                                       1.0i * sin((phi + lambda) / 2.0));*/
    Type e0 = Constants::inv_sqrt_2 +0.0i;
    Type e1 = -Constants::inv_sqrt_2 * (std::cos(lambda)+1.0i*std::sin(lambda));
    Type e2 = Constants::inv_sqrt_2 * (std::cos(phi)+1.0i*std::sin(phi));
    Type e3 = Constants::inv_sqrt_2 * (std::cos(phi+lambda)+1.0i*std::sin(phi+lambda));
    c1_ = C1{e0, e1, e2, e3};
  }

  constexpr GateType gate_type() const { return GateType::U2; }
  void operator()(Type* base, size_t num_slots, const Slot slots[]) const {
    assert(num_slots == 1);
    c1_(base, 1, slots);
  }
  C1 c1_;
};

struct U3 {
  U3() = default;
  U3(Type theta, Type phi, Type lambda) {
    using namespace std::complex_literals;
    Type e0 = std::cos(theta / 2.0) * std::cos((-phi - lambda) / 2.0) +
              1.0i * std::cos(theta / 2.0) * std::sin((-phi - lambda) / 2.0);
    Type e1 = -std::sin(theta / 2.0) * std::cos((-phi + lambda) / 2.0) +
              1.0i * -std::sin(theta / 2.0) * std::sin((-phi + lambda) / 2.0);
    Type e2 = std::sin(theta / 2.0) * std::cos((phi - lambda) / 2.0) +
              1.0i * std::sin(theta / 2.0) * std::sin((phi - lambda) / 2.0);
    Type e3 = std::cos(theta / 2.0) * std::cos((phi + lambda) / 2.0) +
              1.0i * std::cos(theta / 2.0) * std::sin((phi + lambda) / 2.0);
    c1_ = C1{e0, e1, e2, e3};
  }

  constexpr GateType gate_type() const { return GateType::U3; }
  void operator()(Type* base, size_t num_slots, const Slot slots[]) const {
    assert(num_slots == 1);
    c1_(base, 1, slots);
  }
  C1 c1_;
};

// struct RC3X {
//   RC3X() = default;
//   RC3X(Type theta) : theta_{theta} {}
//   constexpr GateType gate_type() const { return GateType::RC3X; }
//   void operator()(Type* base, Slot slot0, Slot slot1, Slot slot2, Slot slot3)
//   {
//     auto v0 = base[(0ul << slots[0]) + (0ul << slots[1])];
//     auto v1 = base[(0ul << slots[0]) + (1ul << slots[1])];
//     auto v2 = base[(1ul << slots[0]) + (0ul << slots[1])];
//     auto v3 = base[(1ul << slots[0]) + (1ul << slots[1])];
//     NOT_IMPLEMENTED();
//   }
//   Type theta_;
// };
/*
struct RXX {
  RXX() = default;
  RXX(Type theta) : theta_{theta} {}
  constexpr GateType gate_type() const { return GateType::RXX; }
  void operator()(Type* base, size_t num_slots, const Slot slots[]) const {
    assert(num_slots == 2);
    auto v0 = base[(0ul << slots[0]) + (0ul << slots[1])];
    auto v1 = base[(0ul << slots[0]) + (1ul << slots[1])];
    auto v2 = base[(1ul << slots[0]) + (0ul << slots[1])];
    auto v3 = base[(1ul << slots[0]) + (1ul << slots[1])];
    NOT_IMPLEMENTED();
  }
  Type theta_;
};
*/

struct RYY {
  RYY() = default;
  //RYY(Type theta) : theta_{theta} {}
  RYY(Type theta) : rx1_{Constants::pi_by_2}, rx2_{Constants::pi_by_2}, rx3_{-Constants::pi_by_2}, rx4_{-Constants::pi_by_2}, rz_{theta} {}
  constexpr GateType gate_type() const { return GateType::RYY; }
  void operator()(Type* base, size_t num_slots, const Slot slots[]) const {
    assert(num_slots == 2);
    auto v0 = base[(0ul << slots[0]) + (0ul << slots[1])];
    auto v1 = base[(0ul << slots[0]) + (1ul << slots[1])];
    auto v2 = base[(1ul << slots[0]) + (0ul << slots[1])];
    auto v3 = base[(1ul << slots[0]) + (1ul << slots[1])];
    int pos_qubit_0 = 1ul << slots[0];
    int pos_qubit_1 = 1ul << slots[1];
    #ifndef AVX512
      rx1_(base,1,&slots[0]);
      rx1_(base+(1ul << slots[1]),1,&slots[0]);
      rx2_(base,1,&slots[1]);
      rx2_(base+(1ul << slots[0]),1,&slots[1]);
      //h2_(base,1,&slots[1]);    
      std::swap(base[(1ul << slots[0]) + (0ul << slots[1])], base[(1ul << slots[0]) + (1ul << slots[1])]);
      rz_(base,1,&slots[1]);
      rz_(base+(1ul << slots[0]),1,&slots[1]);
      std::swap(base[(1ul << slots[0]) + (0ul << slots[1])], base[(1ul << slots[0]) + (1ul << slots[1])]);
      rx3_(base,1,&slots[0]);
      rx3_(base+(1ul << slots[1]),1,&slots[0]);
      rx4_(base,1,&slots[1]);
      rx4_(base+(1ul << slots[0]),1,&slots[1]);
    #else
      //std::cout << "first h" << std::endl;
      if ((slots[0] == 0 && slots[1] == 1) || (slots[1] == 0 && slots[0] ==1)){
        rx1_(base,1,&slots[0]);
        rx2_(base,1,&slots[1]);
      }
      else 
      {
        if (slots[0] == 0){
          rx1_(base,1,&slots[0]);
          rx1_(base+(1ul << slots[1]),1,&slots[0]);
          rx2_(base,1,&slots[1]);
        }
        else if (slots[1] == 0){
          rx1_(base,1,&slots[0]);
          rx2_(base,1,&slots[1]);
          rx2_(base+(1ul << slots[0]),1,&slots[1]);
        }
        else{
          rx1_(base,1,&slots[0]);
          rx1_(base+(1ul << slots[1]),1,&slots[0]);
          rx2_(base,1,&slots[1]);
          rx2_(base+(1ul << slots[0]),1,&slots[1]);
        }
      }

      if ((slots[0] == 0 && slots[1] == 1) || (slots[0] == 1 && slots[1] == 0)){
        int64_t __attribute__((aligned(64))) i_index_ar[8] = {2*pos_qubit_0,2*pos_qubit_0+1,2*(pos_qubit_0+pos_qubit_1),1+2*(pos_qubit_1+pos_qubit_0),(pos_qubit_0+4)*2,(pos_qubit_0+4)*2+1,(pos_qubit_1+pos_qubit_0+4)*2,(pos_qubit_1+pos_qubit_0+4)*2+1};
        __m512i i_index = _mm512_load_epi64(i_index_ar);
        __m512d avx_swap_t = _mm512_i64gather_pd(i_index, (void const *)base,sizeof(double));
     //    std::cout <<"in 0 case " <<avx_swap_t[0] << " " << avx_swap_t[1] <<" "<<  avx_swap_t[2]<<" " <<  avx_swap_t[3]<<" "
     //   <<avx_swap_t[4] <<" "<< avx_swap_t[5] <<" "<<  avx_swap_t[6]<<" " <<  avx_swap_t[7] << std::endl;
        __m512d avx_swap_1 = _mm512_permutex_pd(avx_swap_t,78);
         _mm512_i64scatter_pd((void *)base, i_index, avx_swap_1, sizeof(double));
      }
      else{
        if (std::abs(slots[0]-slots[1]) == 1){
          int64_t __attribute__((aligned(64))) i_index_ar[8] = {2*pos_qubit_0,2*pos_qubit_0+1,2*(pos_qubit_1+pos_qubit_0),1+2*(pos_qubit_1+pos_qubit_0),2+pos_qubit_0*2,pos_qubit_0*2+3,(pos_qubit_1 + pos_qubit_0)*2+2,(pos_qubit_1+pos_qubit_0)*2+3};
          __m512i i_index = _mm512_load_epi64(i_index_ar);
          __m512d avx_swap_t = _mm512_i64gather_pd(i_index, (void const *)base,sizeof(double));
          __m512d avx_swap_1 = _mm512_permutex_pd(avx_swap_t,78);
          _mm512_i64scatter_pd((void *)base, i_index, avx_swap_1, sizeof(double));
        }
        else{
          int64_t __attribute__((aligned(64))) i_index_ar[8] = {2*pos_qubit_0,2*pos_qubit_0+1,2*(pos_qubit_1+pos_qubit_0),1+2*(pos_qubit_1+pos_qubit_0),4+pos_qubit_0*2,pos_qubit_0*2+5,(pos_qubit_1+pos_qubit_0)*2+4,(pos_qubit_1 + pos_qubit_0)*2+5};
          __m512i i_index = _mm512_load_epi64(i_index_ar);
          __m512d avx_swap_t = _mm512_i64gather_pd(i_index, (void const *)base,sizeof(double));
          __m512d avx_swap_1 = _mm512_permutex_pd(avx_swap_t,78);
     //      std::cout <<"in 2 case " <<avx_swap_t[0] << " " << avx_swap_t[1] <<" "<<  avx_swap_t[2]<<" " <<  avx_swap_t[3]<<" "
     //   <<avx_swap_t[4] <<" "<< avx_swap_t[5] <<" "<<  avx_swap_t[6]<<" " <<  avx_swap_t[7] << std::endl;
          _mm512_i64scatter_pd((void *)base, i_index, avx_swap_1, sizeof(double));
        }
      }
      rz_(base,1,&slots[1]);
      if (slots[0] != 0 && slots[1] != 0)
        rz_(base+(1ul << slots[0]),1,&slots[1]);
     if ((slots[0] == 0 && slots[1] == 1) || (slots[0] == 1 && slots[1] == 0)){
        int64_t __attribute__((aligned(64))) i_index_ar[8] = {2*pos_qubit_0,2*pos_qubit_0+1,2*(pos_qubit_0+pos_qubit_1),1+2*(pos_qubit_1+pos_qubit_0),(pos_qubit_0+4)*2,(pos_qubit_0+4)*2+1,(pos_qubit_1+pos_qubit_0+4)*2,(pos_qubit_1+pos_qubit_0+4)*2+1};
        __m512i i_index = _mm512_load_epi64(i_index_ar);
        __m512d avx_swap_t = _mm512_i64gather_pd(i_index, (void const *)base,sizeof(double));
        __m512d avx_swap_1 = _mm512_permutex_pd(avx_swap_t,78);
         _mm512_i64scatter_pd((void *)base, i_index, avx_swap_1, sizeof(double));
      }
      else{
        if (std::abs(slots[0]-slots[1]) == 1){
          int64_t __attribute__((aligned(64))) i_index_ar[8] = {2*pos_qubit_0,2*pos_qubit_0+1,2*(pos_qubit_1+pos_qubit_0),1+2*(pos_qubit_1+pos_qubit_0),2+pos_qubit_0*2,pos_qubit_0*2+3,(pos_qubit_1 + pos_qubit_0)*2+2,(pos_qubit_1+pos_qubit_0)*2+3};
          __m512i i_index = _mm512_load_epi64(i_index_ar);
          __m512d avx_swap_t = _mm512_i64gather_pd(i_index, (void const *)base,sizeof(double));
          __m512d avx_swap_1 = _mm512_permutex_pd(avx_swap_t,78);
          _mm512_i64scatter_pd((void *)base, i_index, avx_swap_1, sizeof(double));
        }
        else{
          int64_t __attribute__((aligned(64))) i_index_ar[8] = {2*pos_qubit_0,2*pos_qubit_0+1,2*(pos_qubit_1+pos_qubit_0),1+2*(pos_qubit_1+pos_qubit_0),4+pos_qubit_0*2,pos_qubit_0*2+5,(pos_qubit_1+pos_qubit_0)*2+4,(pos_qubit_1 + pos_qubit_0)*2+5};
          __m512i i_index = _mm512_load_epi64(i_index_ar);
          __m512d avx_swap_t = _mm512_i64gather_pd(i_index, (void const *)base,sizeof(double));
          __m512d avx_swap_1 = _mm512_permutex_pd(avx_swap_t,78);
      //      std::cout <<"in 2 case " <<avx_swap_t[0] << " " << avx_swap_t[1] <<" "<<  avx_swap_t[2]<<" " <<  avx_swap_t[3]<<" "
     //   <<avx_swap_t[4] <<" "<< avx_swap_t[5] <<" "<<  avx_swap_t[6]<<" " <<  avx_swap_t[7] << std::endl;
          _mm512_i64scatter_pd((void *)base, i_index, avx_swap_1, sizeof(double));
        }
      }
       if ((slots[0] == 0 && slots[1] == 1) || (slots[1] == 0 && slots[0] ==1)){
        rx3_(base,1,&slots[0]);
        rx4_(base,1,&slots[1]);
      }
      else 
      {
        if (slots[0] == 0){
          rx3_(base,1,&slots[0]);   
          rx4_(base,1,&slots[1]);
          rx4_(base+(1ul << slots[0]),1,&slots[1]);
        }
        else if (slots[1] == 0){
          rx3_(base,1,&slots[0]);
          rx3_(base+(1ul << slots[1]),1,&slots[0]);
          rx4_(base,1,&slots[1]);
        }
        else{
          rx3_(base,1,&slots[0]);
          rx3_(base+(1ul << slots[1]),1,&slots[0]);
          rx4_(base,1,&slots[1]);
          rx4_(base+(1ul << slots[0]),1,&slots[1]);
        }
      }
    #endif
  }
  Type theta_;
  RX rx1_;
  RX rx2_;
  RZ rz_;
  RX rx3_;
  RX rx4_;
};

struct RZZ {
  RZZ() = default;
  //RZZ(Type theta) : theta_{theta} {}
  RZZ(Type theta) : rz_{theta} {}
  constexpr GateType gate_type() const { return GateType::RZZ; }
  void operator()(Type* base, size_t num_slots, const Slot slots[]) const {
    assert(num_slots == 2);
    auto v0 = base[(0ul << slots[0]) + (0ul << slots[1])];
    auto v1 = base[(0ul << slots[0]) + (1ul << slots[1])];
    auto v2 = base[(1ul << slots[0]) + (0ul << slots[1])];
    auto v3 = base[(1ul << slots[0]) + (1ul << slots[1])];
    int pos_qubit_0 = 1ul << slots[0];
    int pos_qubit_1 = 1ul << slots[1];
    #ifndef AVX512
      std::swap(base[(1ul << slots[0]) + (0ul << slots[1])], base[(1ul << slots[0]) + (1ul << slots[1])]);
      rz_(base,1,&slots[1]);
      rz_(base+(1ul << slots[0]),1,&slots[1]);
      std::swap(base[(1ul << slots[0]) + (0ul << slots[1])], base[(1ul << slots[0]) + (1ul << slots[1])]);
    #else
     if ((slots[0] == 0 && slots[1] == 1) || (slots[0] == 1 && slots[1] == 0)){
        int64_t __attribute__((aligned(64))) i_index_ar[8] = {2*pos_qubit_0,2*pos_qubit_0+1,2*(pos_qubit_0+pos_qubit_1),1+2*(pos_qubit_1+pos_qubit_0),(pos_qubit_0+4)*2,(pos_qubit_0+4)*2+1,(pos_qubit_1+pos_qubit_0+4)*2,(pos_qubit_1+pos_qubit_0+4)*2+1};
        __m512i i_index = _mm512_load_epi64(i_index_ar);
        __m512d avx_swap_t = _mm512_i64gather_pd(i_index, (void const *)base,sizeof(double));
     //    std::cout <<"in 0 case " <<avx_swap_t[0] << " " << avx_swap_t[1] <<" "<<  avx_swap_t[2]<<" " <<  avx_swap_t[3]<<" "
    //    <<avx_swap_t[4] <<" "<< avx_swap_t[5] <<" "<<  avx_swap_t[6]<<" " <<  avx_swap_t[7] << std::endl;
        __m512d avx_swap_1 = _mm512_permutex_pd(avx_swap_t,78);
         _mm512_i64scatter_pd((void *)base, i_index, avx_swap_1, sizeof(double));
      }
      else{
        if (std::abs(slots[0]-slots[1]) == 1){
          int64_t __attribute__((aligned(64))) i_index_ar[8] = {2*pos_qubit_0,2*pos_qubit_0+1,2*(pos_qubit_1+pos_qubit_0),1+2*(pos_qubit_1+pos_qubit_0),2+pos_qubit_0*2,pos_qubit_0*2+3,(pos_qubit_1 + pos_qubit_0)*2+2,(pos_qubit_1+pos_qubit_0)*2+3};
          __m512i i_index = _mm512_load_epi64(i_index_ar);
          __m512d avx_swap_t = _mm512_i64gather_pd(i_index, (void const *)base,sizeof(double));
          __m512d avx_swap_1 = _mm512_permutex_pd(avx_swap_t,78);
          _mm512_i64scatter_pd((void *)base, i_index, avx_swap_1, sizeof(double));
        }
        else{
          int64_t __attribute__((aligned(64))) i_index_ar[8] = {2*pos_qubit_0,2*pos_qubit_0+1,2*(pos_qubit_1+pos_qubit_0),1+2*(pos_qubit_1+pos_qubit_0),4+pos_qubit_0*2,pos_qubit_0*2+5,(pos_qubit_1+pos_qubit_0)*2+4,(pos_qubit_1 + pos_qubit_0)*2+5};
          __m512i i_index = _mm512_load_epi64(i_index_ar);
          __m512d avx_swap_t = _mm512_i64gather_pd(i_index, (void const *)base,sizeof(double));
          __m512d avx_swap_1 = _mm512_permutex_pd(avx_swap_t,78);
      //     std::cout <<"in 2 case " <<avx_swap_t[0] << " " << avx_swap_t[1] <<" "<<  avx_swap_t[2]<<" " <<  avx_swap_t[3]<<" "
    //    <<avx_swap_t[4] <<" "<< avx_swap_t[5] <<" "<<  avx_swap_t[6]<<" " <<  avx_swap_t[7] << std::endl;
          _mm512_i64scatter_pd((void *)base, i_index, avx_swap_1, sizeof(double));
        }
      }
      rz_(base,1,&slots[1]);
      if (slots[0] != 0 && slots[1] != 0)
        rz_(base+(1ul << slots[0]),1,&slots[1]);
     if ((slots[0] == 0 && slots[1] == 1) || (slots[0] == 1 && slots[1] == 0)){
        int64_t __attribute__((aligned(64))) i_index_ar[8] = {2*pos_qubit_0,2*pos_qubit_0+1,2*(pos_qubit_0+pos_qubit_1),1+2*(pos_qubit_1+pos_qubit_0),(pos_qubit_0+4)*2,(pos_qubit_0+4)*2+1,(pos_qubit_1+pos_qubit_0+4)*2,(pos_qubit_1+pos_qubit_0+4)*2+1};
        __m512i i_index = _mm512_load_epi64(i_index_ar);
        __m512d avx_swap_t = _mm512_i64gather_pd(i_index, (void const *)base,sizeof(double));
        __m512d avx_swap_1 = _mm512_permutex_pd(avx_swap_t,78);
         _mm512_i64scatter_pd((void *)base, i_index, avx_swap_1, sizeof(double));
      }
      else{
        if (std::abs(slots[0]-slots[1]) == 1){
          int64_t __attribute__((aligned(64))) i_index_ar[8] = {2*pos_qubit_0,2*pos_qubit_0+1,2*(pos_qubit_1+pos_qubit_0),1+2*(pos_qubit_1+pos_qubit_0),2+pos_qubit_0*2,pos_qubit_0*2+3,(pos_qubit_1 + pos_qubit_0)*2+2,(pos_qubit_1+pos_qubit_0)*2+3};
          __m512i i_index = _mm512_load_epi64(i_index_ar);
          __m512d avx_swap_t = _mm512_i64gather_pd(i_index, (void const *)base,sizeof(double));
          __m512d avx_swap_1 = _mm512_permutex_pd(avx_swap_t,78);
          _mm512_i64scatter_pd((void *)base, i_index, avx_swap_1, sizeof(double));
        }
        else{
          int64_t __attribute__((aligned(64))) i_index_ar[8] = {2*pos_qubit_0,2*pos_qubit_0+1,2*(pos_qubit_1+pos_qubit_0),1+2*(pos_qubit_1+pos_qubit_0),4+pos_qubit_0*2,pos_qubit_0*2+5,(pos_qubit_1+pos_qubit_0)*2+4,(pos_qubit_1 + pos_qubit_0)*2+5};
          __m512i i_index = _mm512_load_epi64(i_index_ar);
          __m512d avx_swap_t = _mm512_i64gather_pd(i_index, (void const *)base,sizeof(double));
          __m512d avx_swap_1 = _mm512_permutex_pd(avx_swap_t,78);
          //  std::cout <<"in 2 case " <<avx_swap_t[0] << " " << avx_swap_t[1] <<" "<<  avx_swap_t[2]<<" " <<  avx_swap_t[3]<<" "
      //  <<avx_swap_t[4] <<" "<< avx_swap_t[5] <<" "<<  avx_swap_t[6]<<" " <<  avx_swap_t[7] << std::endl;
          _mm512_i64scatter_pd((void *)base, i_index, avx_swap_1, sizeof(double));
        }
      }
    #endif
  }
  Type theta_;
  RZ rz_;
};

// struct RCCX {
//   RCCX() = default;
//   RCCX(Type theta) : theta_{theta} {}
//   constexpr GateType gate_type() const { return GateType::RCCX; }
//   void operator()(Type* base, Slot slot0, Slot slot1, Slot slot2) {
//     auto v0 = base[(0ul << slots[0]) + (0ul << slots[1])];
//     auto v1 = base[(0ul << slots[0]) + (1ul << slots[1])];
//     auto v2 = base[(1ul << slots[0]) + (0ul << slots[1])];
//     auto v3 = base[(1ul << slots[0]) + (1ul << slots[1])];
//     NOT_IMPLEMENTED();
//   }
//   Type theta_;
// };

struct TDG {
  TDG() = default;
  constexpr GateType gate_type() const { return GateType::TDG; }
  void operator()(Type* base, size_t num_slots, const Slot slots[]) const {
    assert(num_slots == 1);
    using namespace std::complex_literals;
    // auto v0 = base[0ul << slots[0]];
    auto v1 = base[1ul << slots[0]];
    #ifndef AVX512
      base[1ul << slots[0]] = Constants::inv_sqrt_2 * (1.0 - 1.0i) * v1;
    #else
      int64_t pos_offset = 1ul << slots[0];
      if (pos_offset == 1){
        __m512d avx_tdg_0 = _mm512_load_pd((void const*)base);
        __m512d avx_tdg_1 = _mm512_permutex_pd(avx_tdg_0,180);
        //__m512d avx_tdg_timer = _mm512_set4_pd(-1.0,1.0,0.0,0.0);
         //std::cout <<avx_x_0[0] << avx_x_0[1] <<  avx_x_0[2] <<  avx_x_0[3]
         //<<avx_x_0[4] << avx_x_0[5] <<  avx_x_0[6] <<  avx_x_0[7] << std::endl;
         //std::cout <<avx_x_dst_0[0] << avx_x_dst_0[1] <<  avx_x_dst_0[2] <<  avx_x_dst_0[3]
         //<<avx_x_dst_0[4] << avx_x_dst_0[5] <<  avx_x_dst_0[6] <<  avx_x_dst_0[7] << std::endl;
        __m512d avx_tdg_tmp1 = _mm512_mul_pd(avx_tdg_1,avx_tdg_timer);
        __m512d avx_tdg_tmp2 = _mm512_add_pd(avx_tdg_tmp1,avx_tdg_0);
        // __m512d avx_tdg_timer_2 = _mm512_set4_pd(Constants::inv_sqrt_2,Constants::inv_sqrt_2,1.0,1.0);
         __m512d avx_tdg_tmp3 = _mm512_mul_pd(avx_tdg_tmp2,avx_tdg_timer_2);
        _mm512_store_pd((void *)base,avx_tdg_tmp3);
      }
      else{
        int64_t __attribute__((aligned(64))) i_index_ar[8] = {0,1,0+2*pos_offset,1+2*pos_offset,2,3,2+2*pos_offset,3+2*pos_offset};
        __m512i i_index = _mm512_load_epi64(i_index_ar);
        __m512d avx_tdg_t = _mm512_i64gather_pd(i_index, (void const *)base,sizeof(double));
        __m512d avx_tdg_1 = _mm512_permutex_pd(avx_tdg_t,180);
        __m512d avx_tdg_tmp1 = _mm512_mul_pd(avx_tdg_1,avx_tdg_timer);
        __m512d avx_tdg_tmp2 = _mm512_add_pd(avx_tdg_tmp1,avx_tdg_t);
        // __m512d avx_tdg_timer_2 = _mm512_set4_pd(Constants::inv_sqrt_2,Constants::inv_sqrt_2,1.0,1.0);
         __m512d avx_tdg_tmp3 = _mm512_mul_pd(avx_tdg_tmp2,avx_tdg_timer_2);
        _mm512_i64scatter_pd((void *)base, i_index, avx_tdg_tmp3, sizeof(double));
      }
    #endif
  }
};

struct SDG {
  SDG() = default;
  constexpr GateType gate_type() const { return GateType::SDG; }
  void operator()(Type* base, size_t num_slots, const Slot slots[]) const {
    assert(num_slots == 1);
    using namespace std::complex_literals;
    // auto v0 = base[0ul << slots[0]];
    auto v1 = base[1ul << slots[0]];
    #ifndef AVX512
      base[1ul << slots[0]] *= -1.0i;
    #else
       int64_t pos_offset = 1ul << slots[0];
      if (pos_offset == 1){
        __m512d avx_sdg_0 = _mm512_loadu_pd((void const*)base);
        __m512d avx_sdg_dst_0 = _mm512_permutex_pd(avx_sdg_0,180);
         //std::cout <<avx_x_0[0] << avx_x_0[1] <<  avx_x_0[2] <<  avx_x_0[3]
         //<<avx_x_0[4] << avx_x_0[5] <<  avx_x_0[6] <<  avx_x_0[7] << std::endl;
         //std::cout <<avx_x_dst_0[0] << avx_x_dst_0[1] <<  avx_x_dst_0[2] <<  avx_x_dst_0[3]
         //<<avx_x_dst_0[4] << avx_x_dst_0[5] <<  avx_x_dst_0[6] <<  avx_x_dst_0[7] << std::endl;
        //__m512d avx_sdg_timer = _mm512_set4_pd(-1.0,1.0,1.0,1.0);
        __m512d avx_sdg_dst_mul = _mm512_mul_pd(avx_sdg_dst_0,avx_sdg_timer);
        _mm512_storeu_pd((void *)base,avx_sdg_dst_mul);
      }
      else{
        int64_t __attribute__((aligned(64))) i_index_ar[8] = {0,1,0+2*pos_offset,1+2*pos_offset,2,3,2+2*pos_offset,3+2*pos_offset};
        __m512i i_index = _mm512_load_epi64(i_index_ar);
        __m512d avx_sdg_t = _mm512_i64gather_pd(i_index, (void const *)base,sizeof(double));
         __m512d avx_sdg_dst_0 = _mm512_permutex_pd(avx_sdg_t,180);
         //std::cout <<avx_x_0[0] << avx_x_0[1] <<  avx_x_0[2] <<  avx_x_0[3]
         //<<avx_x_0[4] << avx_x_0[5] <<  avx_x_0[6] <<  avx_x_0[7] << std::endl;
         //std::cout <<avx_x_dst_0[0] << avx_x_dst_0[1] <<  avx_x_dst_0[2] <<  avx_x_dst_0[3]
         //<<avx_x_dst_0[4] << avx_x_dst_0[5] <<  avx_x_dst_0[6] <<  avx_x_dst_0[7] << std::endl;
        //__m512d avx_sdg_timer = _mm512_set4_pd(-1.0,1.0,1.0,1.0);
        __m512d avx_sdg_dst_mul = _mm512_mul_pd(avx_sdg_dst_0,avx_sdg_timer);
        _mm512_i64scatter_pd((void *)base, i_index, avx_sdg_dst_mul, sizeof(double));
      }
    #endif
  }
};

// -- done
// SWAP	Swap
// X	Pauli-X bit flip
// Y	Pauli-Y bit and phase flip
// Z	Pauli-Z phase flip
// ID	Idle gate or identity
// H	Hadamard
// S	sqrt(Z) phase
// T	sqrt(S) phase
// RX	X-axis rotation
// RY	Y-axis rotation
// RZ	Z-axis rotation
// C1	Arbitrary 1-qubit gate
//	C2	Arbitrary 2-qubit gate
//
// -- todo
//	RXX	2-qubit XX rotation
//	RZZ	2-qubit ZZ rotation
// TDG	conjugate of sqrt(S)
// SDG	conjugate of sqrt(Z)
// U3	3 parameter 2 pulse 1-qubit
// U2	2 parameter 1 pulse 1-qubit
// U1	1 parameter 0 pulse 1-qubit

// controlled gates derived from above
//	RC3X Relative-phase 3-controlled X
//	RCCX	Relative-phase CXX
//	C3XSQRTX	3-controlled sqrt(X)
//	C3X	3-controlled X
// 	CH	Controlled H
// 	CU3	Controlled U3
// 	CU1	Controlled phase rotation
//	CSWAP	Fredkin
//	CRY	Controlled RY rotation
//	CRZ	Controlled RZ rotation
//	CRX	Controlled RX rotation
// CX	Controlled-NOT
// CCX	Toffoli
// CY	Controlled Y
// CZ	Controlled phase
// C4X	4-controlled X

};  // namespace BasicGates

union UnionGate {
  UnionGate() {}
  BasicGates::I I_;
  BasicGates::X X_;
  BasicGates::Y Y_;
  BasicGates::Z Z_;
  BasicGates::H H_;
  BasicGates::S S_;
  BasicGates::T T_;
  BasicGates::RX RX_;
  BasicGates::RY RY_;
  BasicGates::RZ RZ_;
  BasicGates::RI RI_;
  BasicGates::R1 R1_;
  BasicGates::RXFrac RXFrac_;
  BasicGates::RYFrac RYFrac_;
  BasicGates::RZFrac RZFrac_;
  BasicGates::RIFrac RIFrac_;
  BasicGates::R1Frac R1Frac_;
  BasicGates::SWAP SWAP_;
  BasicGates::C1 C1_;
  BasicGates::C2 C2_;
  BasicGates::U1 U1_;
  BasicGates::U2 U2_;
  BasicGates::U3 U3_;
  BasicGates::RXX RXX_;
  BasicGates::RZZ RZZ_;
  BasicGates::RYY RYY_;
  BasicGates::TDG TDG_;
  BasicGates::SDG SDG_;
};


inline GateType gate_name_to_type(std::string name) {
  for(auto& c: name) {
    c = ::toupper(c);
  }
  if(name == "I") { return GateType::I; }
  else if(name == "ALLOC") { return GateType::ALLOC; }
  else if(name == "FREE") { return GateType::FREE; }
  else if(name == "X") { return GateType::X; }
  else if(name == "Y") { return GateType::Y; }
  else if(name == "Z") { return GateType::Z; }
  else if(name == "H") { return GateType::H; }
  else if(name == "S") { return GateType::S; }
  else if(name == "T") { return GateType::T; }
  else if(name == "RX") { return GateType::RX; }
  else if(name == "RY") { return GateType::RY; }
  else if(name == "RZ") { return GateType::RZ; }
  else if(name == "RI") { return GateType::RI; }
  else if(name == "R1") { return GateType::R1; }
  else if(name == "RXFrac") { return GateType::RXFrac; }
  else if(name == "RYFrac") { return GateType::RYFrac; }
  else if(name == "RZFrac") { return GateType::RZFrac; }
  else if(name == "RIFrac") { return GateType::RIFrac; }
  else if(name == "R1Frac") { return GateType::R1Frac; }
  else if(name == "SWAP") { return GateType::SWAP; }
  else if(name == "C1") { return GateType::C1; }
  else if(name == "C2") { return GateType::C2; }
  else if(name == "U1") { return GateType::U1; }
  else if(name == "U2") { return GateType::U2; }
  else if(name == "U3") { return GateType::U3; }
  else if(name == "RXX") { return GateType::RXX; }
  else if(name == "RYY") { return GateType::RYY; }
  else if(name == "RZZ") { return GateType::RZZ; }
  else if(name == "TDG") { return GateType::TDG; }
  else if(name == "SDG") { return GateType::SDG; }
  else return GateType::invalid;
}




struct AggregateGate_0 {
  UnionGate union_gate_;
  GateType gate_type_;

  AggregateGate_0() : gate_type_{GateType::invalid} {}

  AggregateGate_0(const std::string& gate_name)
      : AggregateGate_0(gate_name_to_type(gate_name))
                     {}


  AggregateGate_0(GateType gate_type): gate_type_{gate_type} {
    switch (gate_type) {
      case GateType::I:
        new (&union_gate_.I_) BasicGates::I();
        break;
      case GateType::X:
        new (&union_gate_.X_) BasicGates::X();
        break;
      case GateType::Y:
        new (&union_gate_.Y_) BasicGates::Y();
        break;
      case GateType::Z:
        new (&union_gate_.Z_) BasicGates::Z();
        break;
      case GateType::H:
        new (&union_gate_.H_) BasicGates::H();
        break;
      case GateType::S:
        new (&union_gate_.S_) BasicGates::S();
        break;
      case GateType::T:
        new (&union_gate_.T_) BasicGates::T();
        break;
      case GateType::SWAP:
        new (&union_gate_.SWAP_) BasicGates::SWAP();
        break;
      case GateType::TDG:
        new (&union_gate_.TDG_) BasicGates::TDG();
        break;
      case GateType::SDG:
        new (&union_gate_.SDG_) BasicGates::SDG();
        break;
      case GateType::invalid:
        UNREACHABLE();
        break;
      default:
        NOT_IMPLEMENTED();
    }
  }
};

struct AggregateGate_1 {
  UnionGate union_gate_;
  GateType gate_type_;

  AggregateGate_1() : gate_type_{GateType::invalid} {}

  AggregateGate_1(const std::string& gate_name, Type theta)
      : AggregateGate_1(gate_name_to_type(gate_name),theta)
                     {}

  AggregateGate_1(GateType gate_type, Type theta):gate_type_{gate_type} {
    switch (gate_type) {
      case GateType::RI:
        new (&union_gate_.RI_) BasicGates::RI(theta);
        break;
      case GateType::RX:
        new (&union_gate_.RX_) BasicGates::RX(theta);
        break;
      case GateType::RY:
        new (&union_gate_.RY_) BasicGates::RY(theta);
        break;
      case GateType::RZ:
        new (&union_gate_.RZ_) BasicGates::RZ(theta);
        break;
      case GateType::R1:
        new (&union_gate_.R1_) BasicGates::R1(theta);
        break;
      case GateType::RXX:
        new (&union_gate_.RXX_) BasicGates::RXX(theta);
        break;
      case GateType::RZZ:
        new (&union_gate_.RZZ_) BasicGates::RZZ(theta);
        break;
       case GateType::RYY:
        new (&union_gate_.RYY_) BasicGates::RYY(theta);
        break;
      case GateType::U1:
        new (&union_gate_.U1_) BasicGates::U1(theta);
        break;
      case GateType::invalid:
        UNREACHABLE();
        break;
      default:
        NOT_IMPLEMENTED();
    }
  }
};


struct AggregateGate_2 {
  UnionGate union_gate_;
  GateType gate_type_;

  AggregateGate_2() : gate_type_{GateType::invalid} {}

  AggregateGate_2(const std::string& gate_name, Type phi, Type lambda)
      : AggregateGate_2(gate_name_to_type(gate_name), phi, lambda)
                     {}


  AggregateGate_2(GateType gate_type, Type phi, Type lambda):gate_type_{gate_type} {
    switch (gate_type) {
      case GateType::U2:
        new (&union_gate_.U2_) BasicGates::U2(phi, lambda);
        break;
      case GateType::invalid:
        UNREACHABLE();
        break;
      default:
        NOT_IMPLEMENTED();
    }
  }
};

struct AggregateGate_3 {
  UnionGate union_gate_;
  GateType gate_type_;

  AggregateGate_3() : gate_type_{GateType::invalid} {}

  AggregateGate_3(const std::string& gate_name, Type theta, Type phi, Type lambda)
      : AggregateGate_3(gate_name_to_type(gate_name), theta, phi, lambda)
                     {}


  AggregateGate_3(GateType gate_type, Type theta, Type phi, Type lambda):gate_type_{gate_type} {
    switch (gate_type) {
      case GateType::U3:
        new (&union_gate_.U3_) BasicGates::U3(theta, phi, lambda);
        break;
      case GateType::invalid:
        UNREACHABLE();
        break;
      default:
        NOT_IMPLEMENTED();
    }
  }
};

struct AggregateGate_Frac {
  UnionGate union_gate_;
  GateType gate_type_;

  AggregateGate_Frac() : gate_type_{GateType::invalid} {}

  AggregateGate_Frac(const std::string& gate_name, unsigned numerator, unsigned power)
      : AggregateGate_Frac(gate_name_to_type(gate_name), numerator, power)
                     {}


  AggregateGate_Frac(GateType gate_type, unsigned numerator, unsigned power):gate_type_{gate_type} {
    switch (gate_type) {
      case GateType::RIFrac:
        new (&union_gate_.RIFrac_) BasicGates::RIFrac(numerator,power);
        break;
      case GateType::R1Frac:
        new (&union_gate_.R1Frac_) BasicGates::R1Frac(numerator,power);
        break;
      case GateType::RXFrac:
        new (&union_gate_.RXFrac_) BasicGates::RXFrac(numerator,power);
        break;
      case GateType::RYFrac:
        new (&union_gate_.RYFrac_) BasicGates::RYFrac(numerator,power);
        break;
      case GateType::RZFrac:
        new (&union_gate_.RZFrac_) BasicGates::RZFrac(numerator,power);
        break;
      case GateType::invalid:
        UNREACHABLE();
        break;
      default:
        NOT_IMPLEMENTED();
    }
  }
};

struct AggregateGate_C1 {
  UnionGate union_gate_;
  GateType gate_type_;

  AggregateGate_C1() : gate_type_{GateType::invalid} {}

  AggregateGate_C1(const std::string& gate_name, Type c00, Type c01, Type c10, Type c11)
      : AggregateGate_C1(gate_name_to_type(gate_name), c00, c01, c10, c11)
                     {}


  AggregateGate_C1(GateType gate_type, Type c00, Type c01, Type c10, Type c11):gate_type_{gate_type} {
    switch (gate_type) {
      case GateType::C1:
        new (&union_gate_.C1_) BasicGates::C1(c00, c01, c10, c11);
        break;
      case GateType::invalid:
        UNREACHABLE();
        break;
      default:
        NOT_IMPLEMENTED();
    }
  }
};

struct AggregateGate_C2 {
  UnionGate union_gate_;
  GateType gate_type_;

  AggregateGate_C2() : gate_type_{GateType::invalid} {}

  AggregateGate_C2(const std::string& gate_name, Type c00, Type c01, Type c02, Type c03,
                                                 Type c10, Type c11, Type c12, Type c13,
                                                 Type c20, Type c21, Type c22, Type c23,
                                                 Type c30, Type c31, Type c32, Type c33)
      : AggregateGate_C2(gate_name_to_type(gate_name), c00, c01, c02, c03,
                                                      c10, c11, c12, c13,
                                                      c20, c21, c22, c23,
                                                      c30, c31, c32, c33)
                     {}


  AggregateGate_C2(GateType gate_type, Type c00, Type c01, Type c02, Type c03,
                                                 Type c10, Type c11, Type c12, Type c13,
                                                 Type c20, Type c21, Type c22, Type c23,
                                                 Type c30, Type c31, Type c32, Type c33):gate_type_{gate_type}
                                                  {
    switch (gate_type) {
      case GateType::C2:
        new (&union_gate_.C2_) BasicGates::C2(c00, c01, c02, c03,
                                                  c10, c11, c12, c13,
                                                  c20, c21, c22, c23,
                                                  c30, c31, c32, c33);
        break;
      case GateType::invalid:
        UNREACHABLE();
        break;
      default:
        NOT_IMPLEMENTED();
    }
  }
};

// cannot use pointer as the parameter due to pybind11
class AggregateGateFactory{
  public:
    AggregateGateFactory(AggregateGate_0 ag_0):ag_0_{ag_0} {
      ag_type_ = AGType::NOPARAM;
    }
    AggregateGateFactory(AggregateGate_1 ag_1):ag_1_{ag_1} {
      ag_type_ = AGType::ONEPARAM;
    }
    AggregateGateFactory(AggregateGate_2 ag_2):ag_2_{ag_2} {
      ag_type_ = AGType::TWOPARAM;
    }
    AggregateGateFactory(AggregateGate_3 ag_3):ag_3_{ag_3} {
      ag_type_ = AGType::THREEPARAM;
    }
    AggregateGateFactory(AggregateGate_C1 ag_c1):ag_c1_{ag_c1} {
      ag_type_ = AGType::C1;
    }
    AggregateGateFactory(AggregateGate_C2 ag_c2):ag_c2_{ag_c2} {
      ag_type_ = AGType::C2;
    }
    AggregateGateFactory(AggregateGate_Frac ag_frac):ag_frac_{ag_frac} {
      ag_type_ = AGType::FRAC;
    }
    
    AGType get_type(){
      return ag_type_;
    }

    AggregateGate_0 & get_ag0(){
      return ag_0_;
    }

    AggregateGate_1 & get_ag1(){
      return ag_1_;
    }

    AggregateGate_2 & get_ag2(){
      return ag_2_;
    }

    AggregateGate_3 & get_ag3(){
      return ag_3_;
    }

    AggregateGate_C1 & get_agC1(){
      return ag_c1_;
    }

    AggregateGate_C2 & get_agC2(){
      return ag_c2_;
    }

    AggregateGate_Frac & get_agFrac(){
      return ag_frac_;
    }
  private:
    AggregateGate_0 ag_0_;
    AggregateGate_1 ag_1_;
    AggregateGate_2 ag_2_;
    AggregateGate_3 ag_3_;
    AggregateGate_C1 ag_c1_;
    AggregateGate_C2 ag_c2_;
    AggregateGate_Frac ag_frac_;
    AGType ag_type_ = AGType::invalid;
    

};


#if 0
class AggregateGateFactory{

  public:
    AggregateGateFactory(const std::string &gate_name){
      AggregateGate ag(gate_name);
      ag_ = &ag;
    }
    AggregateGateFactory(const std::string &gate_name, Type theta){
      AggregateGate ag(gate_name,theta);
      ag_ = &ag;
    }
    AggregateGateFactory(const std::string & gate_name, Type phi, Type lambda){
      AggregateGate ag(gate_name, phi, lambda);
      ag_ = &ag;
    }
    AggregateGateFactory(const std::string & gate_name, Type theta, Type phi, Type lambda){
      AggregateGate ag(gate_name, theta, phi, lambda);
      ag_ = &ag;
    }
    AggregateGateFactory(const std::string & gate_name, unsigned numerator, unsigned power){
      AggregateGate ag(gate_name, numerator, power);
      ag_ = &ag;
    }
    AggregateGateFactory(const std::string & gate_name, Type c00, Type c01, Type c10, Type c11){
      AggregateGate ag(gate_name, c00, c01, c10, c11);
      ag_ = &ag;
    }

    AggregateGateFactory(const std::string & gate_name, Type c00, Type c01, Type c02, Type c03, Type c10, Type c11, Type c12,
     Type c13, Type c20, Type c21, Type c22, Type c23, Type c30, Type c31,
     Type c32, Type c33){
       AggregateGate ag(gate_name, c00, c01, c02, c03,
                                      c10, c11, c12, c13,
                                      c20, c21, c22, c23,
                                      c30, c31, c32, c33);
       ag_ = &ag;

     }

     AggregateGate get(){
       return *ag_;
     }
  private:
    AggregateGate * ag_;
};
#endif
};  // namespace SvSim

#endif  // BASIC_GATES_HPP__
