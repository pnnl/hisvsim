#ifndef LOOP_H__
#define LOOP_H__

#include <vector>
#include <chrono>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

using LoopInfo = std::pair<int64_t, int64_t>;
static int n_threads = 1;
extern long long slot_loop_timer;
extern long long inside_loop_func;
void set_num_threads(int num){
  n_threads = num;
}
#define STR(x) #x
#define STRINGIFY(x) STR(x) 
#define CONCATENATE(X,Y) X ( Y )

//#define OMP _Pragma("omp parallel for proc_bind(spread) num_threads(128)")
#define OMP_BLOCK(nthread) _Pragma( STRINGIFY(CONCATENATE(omp parallel num_threads,nthread)))

#define OMP_FOR _Pragma( STRINGIFY(omp for))

//#define OMP_FOR(nthread) _Pragma( STRINGIFY( CONCATENATE(CONCATENATE(omp parallel for proc_bind,spread) num_threads,nthread)))

#define OMP_FOR_COLLAPSE(n_collapse) _Pragma( STRINGIFY( CONCATENATE(omp for collapse,n_collapse)))

#define LOOP(loop_info,start_lo,start_count,stride,count,skip,hi,func) {\
    size_t c = start_count;\
    size_t hi_0 = hi[0];\
    size_t skip_0 = skip[0];\
    size_t i0_bound = (hi_0-start_lo)/skip_0;\
    size_t i0_bound_new = i0_bound/stride;\
    OMP_BLOCK(n_threads)\
    {\
      const auto tid = omp_get_thread_num();\
      OMP_FOR\
      for(size_t i0 = 0; i0 < i0_bound_new; i0 ++){\
        func(start_lo + i0*stride*skip_0, i0,tid);\
      }\
    }\
    c += i0_bound;\
  }\

#define LOOP_2(loop_info,start_lo,start_count,stride,count,skip,hi,func) {\
      size_t c = start_count;\
      size_t hi_0 = hi[0];\
      size_t skip_0 = skip[0];\
      size_t hi_1 = hi[1];\
      size_t skip_1 = skip[1];\
      size_t i0_bound = (hi_0-start_lo)/skip_0;\
      size_t i1_bound = hi_1/skip_1;\
      size_t i1_bound_new = i1_bound/stride;\
      OMP_BLOCK(n_threads)\
      {\
        int tid = omp_get_thread_num();\
        OMP_FOR_COLLAPSE(2) \
        for (size_t i0 = 0; i0 < i0_bound; i0++) {\
          for (size_t i1 = 0; i1 < i1_bound_new; i1+=1) {\
            func(start_lo + i0*skip_0 + i1*skip_1*stride, i0*i1_bound + i1,tid);\
          }\
        }\
      }\
      c+= i0_bound*i1_bound;\
    }\

#define LOOP_3(loop_info,start_lo,start_count,stride,count,skip,hi,func) {\
    size_t c = start_count;\
    size_t hi_0 = hi[0];\
    size_t skip_0 = skip[0];\
    size_t hi_1 = hi[1];\
    size_t skip_1 = skip[1];\
    size_t hi_2 = hi[2];\
    size_t skip_2 = skip[2];\
    size_t i0_bound = (hi_0-start_lo)/skip_0;\
    size_t i1_bound = hi_1/skip_1;\
    size_t i2_bound = hi_2/skip_2;\
    size_t i2_bound_new = i2_bound/stride;\
    OMP_BLOCK(n_threads)\
    {\
      const int tid = omp_get_thread_num();\
      OMP_FOR_COLLAPSE(3)\
      for (size_t i0 = 0; i0 < i0_bound; i0 ++) {\
          for (size_t i1 = 0; i1 < i1_bound; i1 ++) {\
            for (size_t i2 = 0; i2 < i2_bound_new; i2 ++) {\
              func(start_lo+i0*skip_0+i1*skip_1+i2*skip_2*stride,\
              i0*i1_bound*i2_bound+i1*i2_bound+i2,tid);\
              }\
          }\
      }\
    }\
    c += i0_bound*i1_bound*i2_bound;\
  }\
   
#define LOOP_4(loop_info,start_lo,start_count,stride,count,skip,hi,func) {\
    size_t c = start_count;\
    size_t hi_0 = hi[0];\
    size_t skip_0 = skip[0];\
    size_t hi_1 = hi[1];\
    size_t skip_1 = skip[1];\
    size_t hi_2 = hi[2];\
    size_t skip_2 = skip[2];\
    size_t i0_bound = (hi_0-start_lo)/skip_0;\
    size_t i1_bound = hi_1/skip_1;\
    size_t i2_bound = hi_2/skip_2;\
    size_t hi_3 = hi[3];\
    size_t skip_3 = skip[3];\
    size_t i3_bound = hi_3/skip_3;\
    size_t i3_bound_new = i3_bound/stride;\
    OMP_BLOCK(n_threads)\
    {\
      const auto tid = omp_get_thread_num();\
      OMP_FOR_COLLAPSE(4)\
      for (size_t i0 = 0; i0 < i0_bound; i0 ++) {\
        for (size_t i1 = 0; i1 < i1_bound; i1 ++) {\
          for (size_t i2 = 0; i2 < i2_bound; i2 ++) {\
            for (size_t i3 = 0; i3 < i3_bound_new; i3 ++) {\
               func(start_lo+i0*skip_0+i1*skip_1+i2*skip_2+i3*skip_3*stride,\
                  i0*i1_bound*i2_bound*i3_bound+\
                  i1*i2_bound*i3_bound+\
                  i2*i3_bound+i3, tid);\
            }\
          }\
        }\
      }\
    }\
    c += i0_bound*i1_bound*i2_bound*i3_bound;\
}\

#define LOOP_5(loop_info,start_lo,start_count,stride,count,skip,hi,func) {\
    size_t c = start_count;\
    size_t hi_0 = hi[0];\
    size_t skip_0 = skip[0];\
    size_t hi_1 = hi[1];\
    size_t skip_1 = skip[1];\
    size_t hi_2 = hi[2];\
    size_t skip_2 = skip[2];\
    size_t i0_bound = (hi_0-start_lo)/skip_0;\
    size_t i1_bound = hi_1/skip_1;\
    size_t i2_bound = hi_2/skip_2;\
    size_t hi_3 = hi[3];\
    size_t skip_3 = skip[3];\
    size_t i3_bound = hi_3/skip_3;\
    size_t hi_4 = hi[4];\
    size_t skip_4 = skip[4];\
    size_t i4_bound = hi_4/skip_4;\
    size_t i4_bound_new = i4_bound/stride;\
    OMP_BLOCK(n_threads)\
    {\
      const auto tid = omp_get_thread_num();\
      OMP_FOR_COLLAPSE(5)\
      for (size_t i0 = 0; i0 < i0_bound; i0 ++) {\
        for (size_t i1 = 0; i1 < i1_bound; i1 ++) {\
          for (size_t i2 = 0; i2 < i2_bound; i2 ++) {\
            for (size_t i3 = 0; i3 < i3_bound; i3 ++) {\
              for (size_t i4 = 0; i4 < i4_bound_new; i4 ++) {\
              func(start_lo+i0*skip_0+i1*skip_1+i2*skip_2+i3*skip_3+i4*skip_4*stride,\
                  i0*i1_bound*i2_bound*i3_bound*i4_bound+\
                  i1*i2_bound*i3_bound*i4_bound+\
                  i2*i3_bound*i4_bound+\
                  i3*i4_bound+i4,tid);\
              }\
            }\
          }\
        }\
     }\
    }\
    c += i0_bound*i1_bound*i2_bound*i3_bound*i4_bound;\
}\

#define LOOP_6(loop_info,start_lo,start_count,stride,count,skip,hi,func) {\
    size_t c = start_count;\
    size_t hi_0 = hi[0];\
    size_t skip_0 = skip[0];\
    size_t hi_1 = hi[1];\
    size_t skip_1 = skip[1];\
    size_t hi_2 = hi[2];\
    size_t skip_2 = skip[2];\
    size_t i0_bound = (hi_0-start_lo)/skip_0;\
    size_t i1_bound = hi_1/skip_1;\
    size_t i2_bound = hi_2/skip_2;\
    size_t hi_3 = hi[3];\
    size_t skip_3 = skip[3];\
    size_t i3_bound = hi_3/skip_3;\
    size_t hi_4 = hi[4];\
    size_t skip_4 = skip[4];\
    size_t i4_bound = hi_4/skip_4;\
    size_t hi_5 = hi[5];\
    size_t skip_5 = skip[5];\
    size_t i5_bound = hi_5/skip_5;\
    size_t i5_bound_new = i5_bound/stride;\
    OMP_BLOCK(n_threads)\
    {\
      const auto tid = omp_get_thread_num();\
      OMP_FOR_COLLAPSE(6)\
      for (size_t i0 = 0; i0 < i0_bound; i0 ++) {\
        for (size_t i1 = 0; i1 < i1_bound; i1 ++) {\
          for (size_t i2 = 0; i2 < i2_bound; i2 ++) {\
            for (size_t i3 = 0; i3 < i3_bound; i3 ++) {\
              for (size_t i4 = 0; i4 < i4_bound; i4 ++) {\
                for (size_t i5 = 0; i5 < i5_bound_new; i5 ++) {\
                func(start_lo+i0*skip_0+i1*skip_1+i2*skip_2+i3*skip_3+i4*skip_4+i5*skip_5*stride,\
                    i0*i1_bound*i2_bound*i3_bound*i4_bound*i5_bound+\
                    i1*i2_bound*i3_bound*i4_bound*i5_bound+\
                    i2*i3_bound*i4_bound*i5_bound+\
                    i3*i4_bound*i5_bound+\
                    i4*i5_bound+i5, tid);\
                }\
              }\
            }\
          }\
        }\
      }\
    }\
    c += i0_bound*i1_bound*i2_bound*i3_bound*i4_bound*i5_bound;\
}\

#define LOOP_7(loop_info,start_lo,start_count,stride,count,skip,hi,func) {\
    size_t c = start_count;\
    size_t hi_0 = hi[0];\
    size_t skip_0 = skip[0];\
    size_t hi_1 = hi[1];\
    size_t skip_1 = skip[1];\
    size_t hi_2 = hi[2];\
    size_t skip_2 = skip[2];\
    size_t i0_bound = (hi_0-start_lo)/skip_0;\
    size_t i1_bound = hi_1/skip_1;\
    size_t i2_bound = hi_2/skip_2;\
    size_t hi_3 = hi[3];\
    size_t skip_3 = skip[3];\
    size_t i3_bound = hi_3/skip_3;\
    size_t hi_4 = hi[4];\
    size_t skip_4 = skip[4];\
    size_t i4_bound = hi_4/skip_4;\
    size_t hi_5 = hi[5];\
    size_t skip_5 = skip[5];\
    size_t i5_bound = hi_5/skip_5;\
    size_t hi_6 = hi[6];\
    size_t skip_6 = skip[6];\
    size_t i6_bound = hi_6/skip_6;\
    size_t i6_bound_new = i6_bound/stride;\
    OMP_BLOCK(n_threads)\
    {\
      const auto tid = omp_get_thread_num();\
      OMP_FOR_COLLAPSE(7)\
      for (size_t i0 = 0; i0 < i0_bound; i0 ++) {\
        for (size_t i1 = 0; i1 < i1_bound; i1 ++) {\
          for (size_t i2 = 0; i2 < i2_bound; i2 ++) {\
            for (size_t i3 = 0; i3 < i3_bound; i3 ++) {\
              for (size_t i4 = 0; i4 < i4_bound; i4 ++) {\
                for (size_t i5 = 0; i5 < i5_bound; i5 ++) {\
                  for (size_t i6 = 0; i6 < i6_bound_new; i6 ++) {\
                    func(start_lo+i0*skip_0+i1*skip_1+i2*skip_2+i3*skip_3+i4*skip_4+i5*skip_5+i6*skip_6*stride,\
                      i0*i1_bound*i2_bound*i3_bound*i4_bound*i5_bound*i6_bound+\
                      i1*i2_bound*i3_bound*i4_bound*i5_bound*i6_bound+\
                      i2*i3_bound*i4_bound*i5_bound*i6_bound+\
                      i3*i4_bound*i5_bound*i6_bound+\
                      i4*i5_bound*i6_bound+\
                      i5*i6_bound+i6,tid);\
                  }\
                }\
              }\
            }\
          }\
        }\
      }\
    }\
    c += i0_bound*i1_bound*i2_bound*i3_bound*i4_bound*i5_bound*i6_bound;\
}\

#define LOOP_8(loop_info,start_lo,start_count,stride,count,skip,hi,func) {\
    size_t c = start_count;\
    size_t hi_0 = hi[0];\
    size_t skip_0 = skip[0];\
    size_t hi_1 = hi[1];\
    size_t skip_1 = skip[1];\
    size_t hi_2 = hi[2];\
    size_t skip_2 = skip[2];\
    size_t i0_bound = (hi_0-start_lo)/skip_0;\
    size_t i1_bound = hi_1/skip_1;\
    size_t i2_bound = hi_2/skip_2;\
    size_t hi_3 = hi[3];\
    size_t skip_3 = skip[3];\
    size_t i3_bound = hi_3/skip_3;\
    size_t hi_4 = hi[4];\
    size_t skip_4 = skip[4];\
    size_t i4_bound = hi_4/skip_4;\
    size_t hi_5 = hi[5];\
    size_t skip_5 = skip[5];\
    size_t i5_bound = hi_5/skip_5;\
    size_t hi_6 = hi[6];\
    size_t skip_6 = skip[6];\
    size_t i6_bound = hi_6/skip_6;\
    size_t hi_7 = hi[7];\
    size_t skip_7 = skip[7];\
    size_t i7_bound = hi_7/skip_7;\
    size_t i7_bound_new = i7_bound/stride;\
    OMP_BLOCK(n_threads)\
    {\
      const auto tid = omp_get_thread_num();\
      OMP_FOR_COLLAPSE(8)\
      for (size_t i0 = 0; i0 < i0_bound; i0 ++) {\
        for (size_t i1 = 0; i1 < i1_bound; i1 ++) {\
          for (size_t i2 = 0; i2 < i2_bound; i2 ++) {\
            for (size_t i3 = 0; i3 < i3_bound; i3 ++) {\
              for (size_t i4 = 0; i4 < i4_bound; i4 ++) {\
                for (size_t i5 = 0; i5 < i5_bound; i5 ++) {\
                  for (size_t i6 = 0; i6 < i6_bound; i6 ++) {\
                    for (size_t i7 = 0; i7 < i7_bound_new; i7 ++) {\
                      func(start_lo+i0*skip_0+i1*skip_1+i2*skip_2+i3*skip_3+i4*skip_4+i5*skip_5+i6*skip_6+i7*skip_7*stride,\
                        i0*i1_bound*i2_bound*i3_bound*i4_bound*i5_bound*i6_bound*i7_bound+\
                        i1*i2_bound*i3_bound*i4_bound*i5_bound*i6_bound*i7_bound+\
                        i2*i3_bound*i4_bound*i5_bound*i6_bound*i7_bound+\
                        i3*i4_bound*i5_bound*i6_bound*i7_bound+\
                        i4*i5_bound*i6_bound*i7_bound+\
                        i5*i6_bound*i7_bound+\
                        i6*i7_bound+i7,tid);\
                    }\
                  }\
                }\
              }\
            }\
          }\
        }\
      }\
    }\
    c += i0_bound*i1_bound*i2_bound*i3_bound*i4_bound*i5_bound*i6_bound*i7_bound;\
}\

#define LOOP_9(loop_info,start_lo,start_count,stride,count,skip,hi,func){\
    size_t c = start_count;\
    size_t hi_0 = hi[0];\
    size_t skip_0 = skip[0];\
    size_t hi_1 = hi[1];\
    size_t skip_1 = skip[1];\
    size_t hi_2 = hi[2];\
    size_t skip_2 = skip[2];\
    size_t i0_bound = (hi_0-start_lo)/skip_0;\
    size_t i1_bound = hi_1/skip_1;\
    size_t i2_bound = hi_2/skip_2;\
    size_t hi_3 = hi[3];\
    size_t skip_3 = skip[3];\
    size_t i3_bound = hi_3/skip_3;\
    size_t hi_4 = hi[4];\
    size_t skip_4 = skip[4];\
    size_t i4_bound = hi_4/skip_4;\
    size_t hi_5 = hi[5];\
    size_t skip_5 = skip[5];\
    size_t i5_bound = hi_5/skip_5;\
    size_t hi_6 = hi[6];\
    size_t skip_6 = skip[6];\
    size_t i6_bound = hi_6/skip_6;\
    size_t hi_7 = hi[7];\
    size_t skip_7 = skip[7];\
    size_t i7_bound = hi_7/skip_7;\
    size_t hi_8 = hi[8];\
    size_t skip_8 = skip[8];\
    size_t i8_bound = hi_8/skip_8;\
    size_t i8_bound_new = i8_bound/stride;\
    OMP_BLOCK(n_threads)\
    {\
      const auto tid = omp_get_thread_num();\
      OMP_FOR_COLLAPSE(9)\
      for (size_t i0 = 0; i0 < i0_bound; i0 ++) {\
        for (size_t i1 = 0; i1 < i1_bound; i1 ++) {\
          for (size_t i2 = 0; i2 < i2_bound; i2 ++) {\
            for (size_t i3 = 0; i3 < i3_bound; i3 ++) {\
              for (size_t i4 = 0; i4 < i4_bound; i4 ++) {\
                for (size_t i5 = 0; i5 < i5_bound; i5 ++) {\
                  for (size_t i6 = 0; i6 < i6_bound; i6 ++) {\
                    for (size_t i7 = 0; i7 < i7_bound; i7 ++) {\
                      for (size_t i8 = 0; i8 < i8_bound_new; i8++)  {\
                        func(start_lo+i0*skip_0+i1*skip_1+i2*skip_2+i3*skip_3+i4*skip_4+i5*skip_5+i6*skip_6+i7*skip_7+i8*skip_8*stride,\
                            i0*i1_bound*i2_bound*i3_bound*i4_bound*i5_bound*i6_bound*i7_bound*i8_bound+\
                            i1*i2_bound*i3_bound*i4_bound*i5_bound*i6_bound*i7_bound*i8_bound+\
                            i2*i3_bound*i4_bound*i5_bound*i6_bound*i7_bound*i8_bound+\
                            i3*i4_bound*i5_bound*i6_bound*i7_bound*i8_bound+\
                            i4*i5_bound*i6_bound*i7_bound*i8_bound+\
                            i5*i6_bound*i7_bound*i8_bound+\
                            i6*i7_bound*i8_bound+\
                            i7*i8_bound+i8,tid);\
                      }\
                    }\
                  }\
                }\
              }\
            }\
          }\
        }\
      }\
    }\
    c += i0_bound*i1_bound*i2_bound*i3_bound*i4_bound*i5_bound*i6_bound*i7_bound*i8_bound;\
}\

#define LOOP_10(loop_info,start_lo,start_count,stride,count,skip,hi,func){\
    size_t c = start_count;\
    size_t hi_0 = hi[0];\
    size_t skip_0 = skip[0];\
    size_t hi_1 = hi[1];\
    size_t skip_1 = skip[1];\
    size_t hi_2 = hi[2];\
    size_t skip_2 = skip[2];\
    size_t i0_bound = (hi_0-start_lo)/skip_0;\
    size_t i1_bound = hi_1/skip_1;\
    size_t i2_bound = hi_2/skip_2;\
    size_t hi_3 = hi[3];\
    size_t skip_3 = skip[3];\
    size_t i3_bound = hi_3/skip_3;\
    size_t hi_4 = hi[4];\
    size_t skip_4 = skip[4];\
    size_t i4_bound = hi_4/skip_4;\
    size_t hi_5 = hi[5];\
    size_t skip_5 = skip[5];\
    size_t i5_bound = hi_5/skip_5;\
    size_t hi_6 = hi[6];\
    size_t skip_6 = skip[6];\
    size_t i6_bound = hi_6/skip_6;\
    size_t hi_7 = hi[7];\
    size_t skip_7 = skip[7];\
    size_t i7_bound = hi_7/skip_7;\
    size_t hi_8 = hi[8];\
    size_t skip_8 = skip[8];\
    size_t i8_bound = hi_8/skip_8;\
    size_t hi_9 = hi[9];\
    size_t skip_9 = skip[9];\
    size_t i9_bound = hi_9/skip_9;\
    size_t i9_bound_new = i9_bound/stride;\
    OMP_BLOCK(n_threads)\
    {\
      const auto tid = omp_get_thread_num();\
      OMP_FOR_COLLAPSE(10)\
      for (size_t i0 = 0; i0 < i0_bound; i0 ++) {\
        for (size_t i1 = 0; i1 < i1_bound; i1 ++) {\
          for (size_t i2 = 0; i2 < i2_bound; i2 ++) {\
            for (size_t i3 = 0; i3 < i3_bound; i3 ++) {\
              for (size_t i4 = 0; i4 < i4_bound; i4 ++) {\
                for (size_t i5 = 0; i5 < i5_bound; i5 ++) {\
                  for (size_t i6 = 0; i6 < i6_bound; i6 ++) {\
                    for (size_t i7 = 0; i7 < i7_bound; i7 ++) {\
                      for (size_t i8 = 0; i8 < i8_bound; i8++) {\
                        for(size_t i9 = 0; i9 < i9_bound_new; i9++) {\
                     func(start_lo+i0*skip_0+i1*skip_1+i2*skip_2+i3*skip_3+i4*skip_4+i5*skip_5+i6*skip_6+i7*skip_7+i8*skip_8+i9*skip_9*stride,\
                          i0*i1_bound*i2_bound*i3_bound*i4_bound*i5_bound*i6_bound*i7_bound*i8_bound*i9_bound+\
                          i1*i2_bound*i3_bound*i4_bound*i5_bound*i6_bound*i7_bound*i8_bound*i9_bound+\
                          i2*i3_bound*i4_bound*i5_bound*i6_bound*i7_bound*i8_bound*i9_bound+\
                          i3*i4_bound*i5_bound*i6_bound*i7_bound*i8_bound*i9_bound+\
                          i4*i5_bound*i6_bound*i7_bound*i8_bound*i9_bound+\
                          i5*i6_bound*i7_bound*i8_bound*i9_bound+\
                          i6*i7_bound*i8_bound*i9_bound+\
                          i7*i8_bound*i9_bound+\
                          i8*i9_bound+i9,tid);\
                        }\
                      }\
                    }\
                  }\
                }\
              }\
            }\
          }\
        }\
      }\
  }\
  c += i0_bound*i1_bound*i2_bound*i3_bound*i4_bound*i5_bound*i6_bound*i7_bound*i8_bound*i9_bound;\
}\

template <typename Func>
size_t loop(const std::vector<LoopInfo>& loop_info, size_t start_lo,
            size_t start_count, size_t stride, const std::vector<size_t> &count, const std::vector<size_t>& skip, const std::vector<size_t>& hi, Func&& func) {
  stride = 1;
  const size_t N = loop_info.size();
  //std::cout << "num of thread is " << n_threads << " N is " << N << std::endl;
  size_t c = start_count;
  if (N == 0) {
    int tid = omp_get_thread_num();
    func(start_lo, c,tid);
    //std::cout << "Enter N "<< N << std::endl;
  } else if (N == 1) {
    //static schedule 
    /*
    omp_set_dynamic(0);
    omp_set_num_threads(n_threads);
    #pragma omp parallel
    {
      size_t c_tmp = omp_get_thread_num()+c;
      #pragma omp parallel for
      for (size_t i0 = start_lo; i0 < hi[0]; i0 += skip[0]) {
      
        //std::cout<< "N is " << N << " tid is "<< tid << " i0 " << i0 << " c "<< c << std::endl;
        func(i0, c_tmp);
        c_tmp += n_threads;
      }
    }*/
    size_t hi_0 = hi[0];
    size_t skip_0 = skip[0];
    size_t i0_bound = (hi_0-start_lo)/skip_0;
    size_t i0_bound_new = i0_bound/stride;
    
       //auto start_slot_loop = std::chrono::high_resolution_clock::now();
    //{
      #pragma omp parallel for proc_bind(spread) num_threads(n_threads)
      for (size_t i0 = 0; i0 < i0_bound_new; i0 ++) {
       // auto start_func_inloop = std::chrono::high_resolution_clock::now();
        int tid = omp_get_thread_num();
        func(start_lo + i0*stride*skip_0, i0,tid);
        // auto func_loop_elapsed = std::chrono::high_resolution_clock::now() - start_func_inloop;
    //inside_loop_func += std::chrono::duration_cast<std::chrono::microseconds>(func_loop_elapsed).count();
      }
    //}
    c += i0_bound;
   /* std::cout << "Enter N "<< N << " loop bound is "<< i0_bound <<  std::endl;
    std::cout <<" skip_0 "<< skip_0 <<  std::endl;
    std::cout <<" hi_0 "<< hi_0 <<  std::endl;
    std::cout <<" start lo "<< start_lo <<  std::endl;*/
  } 
  else if (N == 2) {
    
    size_t hi_0 = hi[0];
    size_t skip_0 = skip[0];
    size_t hi_1 = hi[1];
    size_t skip_1 = skip[1];
    size_t i0_bound = (hi_0-start_lo)/skip_0;
    size_t i1_bound = hi_1/skip_1;
    #pragma omp parallel for proc_bind(spread) num_threads(n_threads) collapse(2)
    for (size_t i0 = 0; i0 < i0_bound; i0++) {
      for (size_t i1 = 0; i1 < i1_bound; i1+=1) {
        int tid = omp_get_thread_num();
         //std::cout << "the thread id is " <<tid << std::endl;
        //std::cout << i0*skip_0 +i1*skip_1 << std::endl;
        func(start_lo + i0*skip_0 + i1*skip_1, i0*i1_bound + i1,tid);
      }
    }
    //assume none of bounds are 0 
    c+= i0_bound*i1_bound;
   /* std::cout << "Enter N "<< N << " loop bound is "<< i0_bound << " "<< i1_bound << std::endl;
    std::cout <<" skip_0 "<< skip_0 <<  std::endl;
    std::cout <<" hi_0 "<< hi_0 <<  std::endl;
    std::cout <<" start lo "<< start_lo <<  std::endl;
    std::cout <<" skip_1 "<< skip_1 <<  std::endl;
    std::cout <<" hi_1 "<< hi_1 <<  std::endl;*/
  } else if (N == 3) {
    /*
    omp_set_dynamic(0);
    omp_set_num_threads(n_threads);
    #pragma omp parallel
    {
      size_t c_tmp = omp_get_thread_num()+c;
      #pragma omp parallel for collapse(3)
      for (size_t i0 = start_lo; i0 < hi[0]; i0 += skip[0]) {
        for (size_t i1 = 0; i1 < hi[1]; i1 += skip[1]) {
          for (size_t i2 = 0; i2 < hi[2]; i2 += skip[2]) {
            //std::cout<< "N is " << N << " tid is "<< tid << " i0 " << i0  << " i1 " << i1 << " i2" << i2<< " c "<< c << std::endl;
            func(i0+i1+i2, c_tmp);
            c_tmp += n_threads;
          }
       }
      }
    }*/
    size_t hi_0 = hi[0];
    size_t skip_0 = skip[0];
    size_t hi_1 = hi[1];
    size_t skip_1 = skip[1];
    size_t hi_2 = hi[2];
    size_t skip_2 = skip[2];
    size_t i0_bound = (hi_0-start_lo)/skip_0;
    size_t i1_bound = hi_1/skip_1;
    size_t i2_bound = hi_2/skip_2;
    #pragma omp parallel for proc_bind(spread) num_threads(n_threads) collapse(3)
    for (size_t i0 = 0; i0 < i0_bound; i0 ++) {
        for (size_t i1 = 0; i1 < i1_bound; i1 ++) {
          for (size_t i2 = 0; i2 < i2_bound; i2 +=1) {
            //std::cout<< "N is " << N << " tid is "<< tid << " i0 " << i0  << " i1 " << i1 << " i2" << i2<< " c "<< c << std::endl;
            int tid = omp_get_thread_num();
            func(start_lo+i0*skip_0+i1*skip_1+i2*skip_2,
                i0*i1_bound*i2_bound+
                i1*i2_bound+i2,tid);
          }
       }
    }
    c += i0_bound*i1_bound*i2_bound;
    /*std::cout << "Enter N "<< N << " loop bound is "<< i0_bound << " " << i1_bound <<" " <<i2_bound << std::endl;
    std::cout << "Enter N "<< N << " loop bound is "<< i0_bound << " "<< i1_bound << std::endl;
    std::cout <<" skip_0 "<< skip_0 <<  std::endl;
    std::cout <<" hi_0 "<< hi_0 <<  std::endl;
    std::cout <<" start lo "<< start_lo <<  std::endl;
    std::cout <<" skip_1 "<< skip_1 <<  std::endl;
    std::cout <<" hi_1 "<< hi_1 <<  std::endl;
     std::cout <<" skip_2 "<< skip_2 <<  std::endl;
    std::cout <<" hi_2 "<< hi_2 << std::endl;*/
  } else if (N == 4) {
    size_t hi_0 = hi[0];
    size_t skip_0 = skip[0];
    size_t hi_1 = hi[1];
    size_t skip_1 = skip[1];
    size_t hi_2 = hi[2];
    size_t skip_2 = skip[2];
    size_t i0_bound = (hi_0-start_lo)/skip_0;
    size_t i1_bound = hi_1/skip_1;
    size_t i2_bound = hi_2/skip_2;
    size_t hi_3 = hi[3];
    size_t skip_3 = skip[3];
    size_t i3_bound = hi_3/skip_3;
    #pragma omp parallel for proc_bind(spread) num_threads(n_threads) collapse(4)
    for (size_t i0 = 0; i0 < i0_bound; i0 ++) {
      for (size_t i1 = 0; i1 < i1_bound; i1 ++) {
        for (size_t i2 = 0; i2 < i2_bound; i2 ++) {
          for (size_t i3 = 0; i3 < i3_bound; i3 ++) {
            //int tid = omp_get_thread_num();
            //std::cout<< "N is " << N << " tid is "<< tid << " i0 " << i0  << " i1 " << i1 << " i2" << i2 << " i3" << i3 << " c "<< c << std::endl;
             int tid = omp_get_thread_num();
             func(start_lo+i0*skip_0+i1*skip_1+i2*skip_2+i3*skip_3,
                  i0*i1_bound*i2_bound*i3_bound+
                  i1*i2_bound*i3_bound+
                  i2*i3_bound+i3, tid);
          }
        }
      }
    }
    c += i0_bound*i1_bound*i2_bound*i3_bound;
    //std::cout << "Enter N "<< N << " loop bound is "<< i0_bound*i1_bound*i2_bound*i3_bound << std::endl;
  } else if (N == 5) {
    size_t hi_0 = hi[0];
    size_t skip_0 = skip[0];
    size_t hi_1 = hi[1];
    size_t skip_1 = skip[1];
    size_t hi_2 = hi[2];
    size_t skip_2 = skip[2];
    size_t i0_bound = (hi_0-start_lo)/skip_0;
    size_t i1_bound = hi_1/skip_1;
    size_t i2_bound = hi_2/skip_2;
    size_t hi_3 = hi[3];
    size_t skip_3 = skip[3];
    size_t i3_bound = hi_3/skip_3;
    size_t hi_4 = hi[4];
    size_t skip_4 = skip[4];
    size_t i4_bound = hi_4/skip_4;
    #pragma omp parallel for proc_bind(spread) num_threads(n_threads) collapse(5)
    for (size_t i0 = 0; i0 < i0_bound; i0 ++) {
      for (size_t i1 = 0; i1 < i1_bound; i1 ++) {
        for (size_t i2 = 0; i2 < i2_bound; i2 ++) {
          for (size_t i3 = 0; i3 < i3_bound; i3 ++) {
            for (size_t i4 = 0; i4 < i4_bound; i4 ++) {
              int tid = omp_get_thread_num();
              func(start_lo+i0*skip_0+i1*skip_1+i2*skip_2+i3*skip_3+i4*skip_4, 
                  i0*i1_bound*i2_bound*i3_bound*i4_bound+
                  i1*i2_bound*i3_bound*i4_bound+
                  i2*i3_bound*i4_bound+
                  i3*i4_bound+i4,tid);
            }
          }
        }
      }
    }
    c += i0_bound*i1_bound*i2_bound*i3_bound*i4_bound;
    //std::cout << "Enter N "<< N << " loop bound is "<< i0_bound*i1_bound*i2_bound*i3_bound*i4_bound << std::endl;
  } else if (N == 6) {
    size_t hi_0 = hi[0];
    size_t skip_0 = skip[0];
    size_t hi_1 = hi[1];
    size_t skip_1 = skip[1];
    size_t hi_2 = hi[2];
    size_t skip_2 = skip[2];
    size_t i0_bound = (hi_0-start_lo)/skip_0;
    size_t i1_bound = hi_1/skip_1;
    size_t i2_bound = hi_2/skip_2;
    size_t hi_3 = hi[3];
    size_t skip_3 = skip[3];
    size_t i3_bound = hi_3/skip_3;
    size_t hi_4 = hi[4];
    size_t skip_4 = skip[4];
    size_t i4_bound = hi_4/skip_4;
    size_t hi_5 = hi[5];
    size_t skip_5 = skip[5];
    size_t i5_bound = hi_5/skip_5;
    #pragma omp parallel for proc_bind(spread) num_threads(n_threads) collapse(6)
    for (size_t i0 = 0; i0 < i0_bound; i0 ++) {
      for (size_t i1 = 0; i1 < i1_bound; i1 ++) {
        for (size_t i2 = 0; i2 < i2_bound; i2 ++) {
          for (size_t i3 = 0; i3 < i3_bound; i3 ++) {
            for (size_t i4 = 0; i4 < i4_bound; i4 ++) {
              for (size_t i5 = 0; i5 < i5_bound; i5 ++) {
                int tid = omp_get_thread_num();
                func(start_lo+i0*skip_0+i1*skip_1+i2*skip_2+i3*skip_3+i4*skip_4+i5*skip_5, 
                    i0*i1_bound*i2_bound*i3_bound*i4_bound*i5_bound+
                    i1*i2_bound*i3_bound*i4_bound*i5_bound+
                    i2*i3_bound*i4_bound*i5_bound+
                    i3*i4_bound*i5_bound+
                    i4*i5_bound+i5, tid);
              }
            }
          }
        }
      }
    }
    c += i0_bound*i1_bound*i2_bound*i3_bound*i4_bound*i5_bound;
    //std::cout << "Enter N "<< N << " loop bound is "<< i0_bound*i1_bound*i2_bound*i3_bound*i4_bound*i5_bound << std::endl;
  } else if (N == 7) {
    size_t hi_0 = hi[0];
    size_t skip_0 = skip[0];
    size_t hi_1 = hi[1];
    size_t skip_1 = skip[1];
    size_t hi_2 = hi[2];
    size_t skip_2 = skip[2];
    size_t i0_bound = (hi_0-start_lo)/skip_0;
    size_t i1_bound = hi_1/skip_1;
    size_t i2_bound = hi_2/skip_2;
    size_t hi_3 = hi[3];
    size_t skip_3 = skip[3];
    size_t i3_bound = hi_3/skip_3;
    size_t hi_4 = hi[4];
    size_t skip_4 = skip[4];
    size_t i4_bound = hi_4/skip_4;
    size_t hi_5 = hi[5];
    size_t skip_5 = skip[5];
    size_t i5_bound = hi_5/skip_5;
    size_t hi_6 = hi[6];
    size_t skip_6 = skip[6];
    size_t i6_bound = hi_6/skip_6;
    #pragma omp parallel for proc_bind(spread) num_threads(n_threads) collapse(7)
    for (size_t i0 = 0; i0 < i0_bound; i0 ++) {
      for (size_t i1 = 0; i1 < i1_bound; i1 ++) {
        for (size_t i2 = 0; i2 < i2_bound; i2 ++) {
          for (size_t i3 = 0; i3 < i3_bound; i3 ++) {
            for (size_t i4 = 0; i4 < i4_bound; i4 ++) {
              for (size_t i5 = 0; i5 < i5_bound; i5 ++) {
                for (size_t i6 = 0; i6 < i6_bound; i6 ++) {
                 int tid = omp_get_thread_num();
                 func(start_lo+i0*skip_0+i1*skip_1+i2*skip_2+i3*skip_3+i4*skip_4+i5*skip_5+i6*skip_6, 
                      i0*i1_bound*i2_bound*i3_bound*i4_bound*i5_bound*i6_bound+
                      i1*i2_bound*i3_bound*i4_bound*i5_bound*i6_bound+
                      i2*i3_bound*i4_bound*i5_bound*i6_bound+
                      i3*i4_bound*i5_bound*i6_bound+
                      i4*i5_bound*i6_bound+
                      i5*i6_bound+i6,tid);
                }
              }
            }
          }
        }
      }
    }
    c += i0_bound*i1_bound*i2_bound*i3_bound*i4_bound*i5_bound*i6_bound;
    //std::cout << "Enter N "<< N << " loop bound is "<< i0_bound*i1_bound*i2_bound*i3_bound*i4_bound*i5_bound*i6_bound << std::endl;
  } else if (N == 8) {
    size_t hi_0 = hi[0];
    size_t skip_0 = skip[0];
    size_t hi_1 = hi[1];
    size_t skip_1 = skip[1];
    size_t hi_2 = hi[2];
    size_t skip_2 = skip[2];
    size_t i0_bound = (hi_0-start_lo)/skip_0;
    size_t i1_bound = hi_1/skip_1;
    size_t i2_bound = hi_2/skip_2;
    size_t hi_3 = hi[3];
    size_t skip_3 = skip[3];
    size_t i3_bound = hi_3/skip_3;
    size_t hi_4 = hi[4];
    size_t skip_4 = skip[4];
    size_t i4_bound = hi_4/skip_4;
    size_t hi_5 = hi[5];
    size_t skip_5 = skip[5];
    size_t i5_bound = hi_5/skip_5;
    size_t hi_6 = hi[6];
    size_t skip_6 = skip[6];
    size_t i6_bound = hi_6/skip_6;
    size_t hi_7 = hi[7];
    size_t skip_7 = skip[7];
    size_t i7_bound = hi_7/skip_7;
    #pragma omp parallel for proc_bind(spread) num_threads(n_threads) collapse(8)
    for (size_t i0 = 0; i0 < i0_bound; i0 ++) {
      for (size_t i1 = 0; i1 < i1_bound; i1 ++) {
        for (size_t i2 = 0; i2 < i2_bound; i2 ++) {
          for (size_t i3 = 0; i3 < i3_bound; i3 ++) {
            for (size_t i4 = 0; i4 < i4_bound; i4 ++) {
              for (size_t i5 = 0; i5 < i5_bound; i5 ++) {
                for (size_t i6 = 0; i6 < i6_bound; i6 ++) {
                  for (size_t i7 = 0; i7 < i7_bound; i7 ++) {
                     int tid = omp_get_thread_num();
                     func(start_lo+i0*skip_0+i1*skip_1+i2*skip_2+i3*skip_3+i4*skip_4+i5*skip_5+i6*skip_6+i7*skip_7,
                        i0*i1_bound*i2_bound*i3_bound*i4_bound*i5_bound*i6_bound*i7_bound+
                        i1*i2_bound*i3_bound*i4_bound*i5_bound*i6_bound*i7_bound+
                        i2*i3_bound*i4_bound*i5_bound*i6_bound*i7_bound+
                        i3*i4_bound*i5_bound*i6_bound*i7_bound+
                        i4*i5_bound*i6_bound*i7_bound+
                        i5*i6_bound*i7_bound+
                        i6*i7_bound+i7,tid);
                  }
                }
              }
            }
          }
        }
      }
    }
    c += i0_bound*i1_bound*i2_bound*i3_bound*i4_bound*i5_bound*i6_bound*i7_bound;
    //std::cout << "Enter N "<< N << " loop bound is "<< i0_bound*i1_bound*i2_bound*i3_bound*i4_bound*i5_bound*i6_bound*i7_bound << std::endl;
  } else if (N == 9) {
    size_t hi_0 = hi[0];
    size_t skip_0 = skip[0];
    size_t hi_1 = hi[1];
    size_t skip_1 = skip[1];
    size_t hi_2 = hi[2];
    size_t skip_2 = skip[2];
    size_t i0_bound = (hi_0-start_lo)/skip_0;
    size_t i1_bound = hi_1/skip_1;
    size_t i2_bound = hi_2/skip_2;
    size_t hi_3 = hi[3];
    size_t skip_3 = skip[3];
    size_t i3_bound = hi_3/skip_3;
    size_t hi_4 = hi[4];
    size_t skip_4 = skip[4];
    size_t i4_bound = hi_4/skip_4;
    size_t hi_5 = hi[5];
    size_t skip_5 = skip[5];
    size_t i5_bound = hi_5/skip_5;
    size_t hi_6 = hi[6];
    size_t skip_6 = skip[6];
    size_t i6_bound = hi_6/skip_6;
    size_t hi_7 = hi[7];
    size_t skip_7 = skip[7];
    size_t i7_bound = hi_7/skip_7;
    size_t hi_8 = hi[8];
    size_t skip_8 = skip[8];
    size_t i8_bound = hi_8/skip_8;
    #pragma omp parallel for proc_bind(spread) num_threads(n_threads) collapse(9)
    for (size_t i0 = 0; i0 < i0_bound; i0 ++) {
      for (size_t i1 = 0; i1 < i1_bound; i1 ++) {
        for (size_t i2 = 0; i2 < i2_bound; i2 ++) {
          for (size_t i3 = 0; i3 < i3_bound; i3 ++) {
            for (size_t i4 = 0; i4 < i4_bound; i4 ++) {
              for (size_t i5 = 0; i5 < i5_bound; i5 ++) {
                for (size_t i6 = 0; i6 < i6_bound; i6 ++) {
                  for (size_t i7 = 0; i7 < i7_bound; i7 ++) {
                    for (size_t i8 = 0; i8 < i8_bound; i8++)  {
                     int tid = omp_get_thread_num();
                     func(start_lo+i0*skip_0+i1*skip_1+i2*skip_2+i3*skip_3+i4*skip_4+i5*skip_5+i6*skip_6+i7*skip_7+i8*skip_8, 
                          i0*i1_bound*i2_bound*i3_bound*i4_bound*i5_bound*i6_bound*i7_bound*i8_bound+
                          i1*i2_bound*i3_bound*i4_bound*i5_bound*i6_bound*i7_bound*i8_bound+
                          i2*i3_bound*i4_bound*i5_bound*i6_bound*i7_bound*i8_bound+
                          i3*i4_bound*i5_bound*i6_bound*i7_bound*i8_bound+
                          i4*i5_bound*i6_bound*i7_bound*i8_bound+
                          i5*i6_bound*i7_bound*i8_bound+
                          i6*i7_bound*i8_bound+
                          i7*i8_bound+i8,tid);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    c += i0_bound*i1_bound*i2_bound*i3_bound*i4_bound*i5_bound*i6_bound*i7_bound*i8_bound;
    //std::cout << "Enter N "<< N << " loop bound is "<< i0_bound*i1_bound*i2_bound*i3_bound*i4_bound*i5_bound*i6_bound*i7_bound*i8_bound << std::endl;
  } else if (N == 10) {
    size_t hi_0 = hi[0];
    size_t skip_0 = skip[0];
    size_t hi_1 = hi[1];
    size_t skip_1 = skip[1];
    size_t hi_2 = hi[2];
    size_t skip_2 = skip[2];
    size_t i0_bound = (hi_0-start_lo)/skip_0;
    size_t i1_bound = hi_1/skip_1;
    size_t i2_bound = hi_2/skip_2;
    size_t hi_3 = hi[3];
    size_t skip_3 = skip[3];
    size_t i3_bound = hi_3/skip_3;
    size_t hi_4 = hi[4];
    size_t skip_4 = skip[4];
    size_t i4_bound = hi_4/skip_4;
    size_t hi_5 = hi[5];
    size_t skip_5 = skip[5];
    size_t i5_bound = hi_5/skip_5;
    size_t hi_6 = hi[6];
    size_t skip_6 = skip[6];
    size_t i6_bound = hi_6/skip_6;
    size_t hi_7 = hi[7];
    size_t skip_7 = skip[7];
    size_t i7_bound = hi_7/skip_7;
    size_t hi_8 = hi[8];
    size_t skip_8 = skip[8];
    size_t i8_bound = hi_8/skip_8;
    size_t hi_9 = hi[9];
    size_t skip_9 = skip[9];
    size_t i9_bound = hi_9/skip_9;
    #pragma omp parallel for proc_bind(spread) num_threads(n_threads) collapse(10)
    for (size_t i0 = 0; i0 < i0_bound; i0 ++) {
      for (size_t i1 = 0; i1 < i1_bound; i1 ++) {
        for (size_t i2 = 0; i2 < i2_bound; i2 ++) {
          for (size_t i3 = 0; i3 < i3_bound; i3 ++) {
            for (size_t i4 = 0; i4 < i4_bound; i4 ++) {
              for (size_t i5 = 0; i5 < i5_bound; i5 ++) {
                for (size_t i6 = 0; i6 < i6_bound; i6 ++) {
                  for (size_t i7 = 0; i7 < i7_bound; i7 ++) {
                    for (size_t i8 = 0; i8 < i8_bound; i8++) {
                      for(size_t i9 = 0; i9 < i9_bound; i9++) {
                        int tid = omp_get_thread_num();
                     func(start_lo+i0*skip_0+i1*skip_1+i2*skip_2+i3*skip_3+i4*skip_4+i5*skip_5+i6*skip_6+i7*skip_7+i8*skip_8+i9*skip_9, 
                          i0*i1_bound*i2_bound*i3_bound*i4_bound*i5_bound*i6_bound*i7_bound*i8_bound*i9_bound+
                          i1*i2_bound*i3_bound*i4_bound*i5_bound*i6_bound*i7_bound*i8_bound*i9_bound+
                          i2*i3_bound*i4_bound*i5_bound*i6_bound*i7_bound*i8_bound*i9_bound+
                          i3*i4_bound*i5_bound*i6_bound*i7_bound*i8_bound*i9_bound+
                          i4*i5_bound*i6_bound*i7_bound*i8_bound*i9_bound+
                          i5*i6_bound*i7_bound*i8_bound*i9_bound+
                          i6*i7_bound*i8_bound*i9_bound+
                          i7*i8_bound*i9_bound+
                          i8*i9_bound+i9,tid);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    c += i0_bound*i1_bound*i2_bound*i3_bound*i4_bound*i5_bound*i6_bound*i7_bound*i8_bound*i9_bound;
    //std::cout << "Enter N "<< N << " loop bound is "<< i0_bound*i1_bound*i2_bound*i3_bound*i4_bound*i5_bound*i6_bound*i7_bound*i8_bound*i9_bound << std::endl;
  } else {
    std::vector<LoopInfo> rest_loop_info{loop_info.begin() + 10,
                                         loop_info.end()};
    size_t ld = 1;
    for (int i = 0; i < 10; ++i) {
      ld *= count[i];
    }
    std::vector<size_t> count_ex(rest_loop_info.size());
    std::vector<size_t> skip_ex(rest_loop_info.size());
    std::vector<size_t> hi_ex(rest_loop_info.size());
    for (size_t i = 0; i < rest_loop_info.size(); i++) {
      std::tie(count_ex[i], skip_ex[i]) = rest_loop_info[i];
      hi_ex[i] = count_ex[i] * skip_ex[i];
    }
    #pragma omp parallel for num_threads(n_threads)
    for (size_t i0 = start_lo; i0 < hi_ex[0]; i0 += skip_ex[0]) {
      for (size_t i1 = i0; i1 < i0 + hi_ex[1]; i1 += skip_ex[1]) {
        for (size_t i2 = i1; i2 < i1 + hi_ex[2]; i2 += skip_ex[2]) {
          for (size_t i3 = i2; i3 < i2 + hi_ex[3]; i3 += skip_ex[3]) {
            for (size_t i4 = i3; i4 < i3 + hi_ex[4]; i4 += skip_ex[4]) {
              for (size_t i5 = i4; i5 < i4 + hi_ex[5]; i5 += skip_ex[5]) {
                for (size_t i6 = i5; i6 < i5 + hi_ex[6]; i6 += skip_ex[6]) {
                  for (size_t i7 = i6; i7 < i6 + hi_ex[7]; i7 += skip_ex[7]) {
                    for (size_t i8 = i7; i8 < i7 + hi_ex[8]; i8 += skip_ex[8]) {
                      for (size_t i9 = i8; i9 < i8 + hi_ex[9]; i9 += skip_ex[9]) {
                        std::cout<< "N is " << N << std::endl;
                        c = loop(rest_loop_info, i9, c, stride,count_ex,skip_ex,hi_ex,
                                 std::forward<Func>(func));
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return c;
}

#endif  // LOOP_H__
