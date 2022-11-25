#ifndef STATE_VECTOR_HPP_
#define STATE_VECTOR_HPP_

#include <algorithm>
#include <array>
#include <cassert>
#include <complex>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <vector>
#include <bitset>
#include <sstream>
#include "numa.h"

// #include "boundvec.hpp"
#include "types.hpp"
#include "loop.hpp"

#include "basic_gates.hpp"

#include <chrono>

long long slot_loop_timer = 0;
long long inside_loop_func = 0;
namespace SvSim {
std::vector<long long> gate_counter_t(32*128,0);
static long long timer = 0;
static int init_num_thread = 1;

static long long gate_counter = 0;
static long long lambda_timer = 0;
static long long mem_move_counter = 0;

void set_init_num_threads(int num){
    init_num_thread = num;
}

class StateVector {
 public:
  using SlotBitVector = uint64_t;
  static constexpr Slot MAX_SLOTS = 64;
  static constexpr Qubit MAX_QUBITS = 64;
  static constexpr Qubit qubit_invalid = -1;
  static constexpr Slot slot_invalid = -1;

  // StateVector() = default;
  StateVector(const StateVector&) = default;

  StateVector(Qubit num_qubits, Type * state)
      : StateVector{num_qubits} {
    state_ = state;
    EXPECTS(is_normalized(),
            "Constructing state with non-normalized state. Please normalize "
            "before construction.");
  }

  StateVector(Qubit num_qubits=0,int numa=1)
      : StateVector{identity_qubit_mapping(num_qubits),{},numa} {}

  bool is_normalized() const {
    double prob = 0;
    for (size_t i = 0; i < sv_size_; i++){
      prob += std::abs(state_[i] * state_[i]);
    }
    /*for(auto v : state_) {
      prob += std::abs(v * v);
    }*/
    return approx_equal(prob, 1);
  }

  StateVector(const std::vector<Qubit>& stored_qubits,
              const std::set<Qubit>& enabled_control_qubits = {},int numa = 1)
      : enabled_control_qubits_{enabled_control_qubits},
        num_slots_{0},
        slot_to_qubit_(MAX_SLOTS, -1),
        qubit_to_slot_(MAX_QUBITS, -1),
        numa_(numa) {
    // assert(stored_qubits.size() > 0);
    assert(stored_qubits.size() <= MAX_SLOTS);
    // control qubits should not be stored qubits
    for (const auto q : stored_qubits) {
      assert(q >= 0);
      assert(enabled_control_qubits.find(q) == enabled_control_qubits.end());
    }
    // // store qubits into slots in order
    // num_slots_ = stored_qubits.size();
    // std::copy_n(stored_qubits_.begin(), stored_qubits.size(),
    //             slot_to_qubit_.begin());
    // // max_num_qubits_ = *std::max_element(stored_qubits.begin(),
    // // stored_qubits.end()); assert(max_num_qubits_ < MAX_QUBITS);
    // // qubit_to_slot_ = std::vector<Slot>(
    // //     1 + *std::max_element(stored_qubits.begin(), stored_qubits.end()),
    // //     -1);
    // for (size_t s = 0; s < stored_qubits_.size(); ++s) {
    //   EXPECTS(slot_to_qubit_[s] < MAX_QUBITS,
    //           "Qubit id larger than max allowed. Increase MAX_QUBITS and "
    //           "recompile.");
    //   qubit_to_slot_[slot_to_qubit_[s]] = static_cast<Slot>(s);
    // }
    // state_ = std::vector<Type>(1ul << num_slots_, 0);
    if (numa_ == 1){
      state_ = (Type*)numa_alloc_interleaved(sizeof(Type)*1);
    }
    else
      state_ = (Type*)malloc(sizeof(Type)*1);
    sv_size_ = 1;
    for(size_t i = 0; i < sv_size_; i++)
      state_[i] = 1;
    //memset(state_,1,sv_size_);
    //state_.resize(1, 1);
   /* for (auto sq: stored_qubits) {
      allocate(sq);
    }*/
    allocate(stored_qubits);
    state_[0] = 1;  // initial state
  }

  StateVector& operator=(StateVector sv) {
    using std::swap;
    swap(*this, sv);
    return *this;
  }

  /*this can help me dump state */
  void dump_state(std::ostream& os) {
    //for (size_t i = 0; i < state_.size(); ++i) {
    for (size_t i = 0; i < sv_size_; ++i) {
      os << std::bitset<16>(i) << ": " << state_[i] << "\n";
    }
  }

   void dump_state_new() {
    //for (size_t i = 0; i < state_.size(); ++i) {
    for (size_t i = 0; i < sv_size_; ++i) {
      std::cout << std::bitset<16>(i) << ": " << state_[i] << "\n";
    }
  }

  Slot num_slots() const { return num_slots_; }

  const std::vector<Qubit>& qubits() const { return stored_qubits_; }

  Slot qubit_to_slot(Qubit qubit) const {
    // std::cerr<<__LINE__<<" qubit_to_slot_.size()="<<qubit_to_slot_.size()<<"\n";
    assert(qubit >= 0 && qubit < static_cast<int>(qubit_to_slot_.size()));
    return qubit_to_slot_[qubit];
  }

  Qubit allocate_new_qubit() {
    assert(num_slots_ < MAX_SLOTS);
    Qubit new_qubit = qubit_invalid;
    for(size_t i=0; i<qubit_to_slot_.size(); ++i) {
      if(qubit_to_slot_[i] == slot_invalid) {
        new_qubit = static_cast<Qubit>(i);
        break;
      }
    }
    if (new_qubit == qubit_invalid) {
      qubit_to_slot_.push_back(slot_invalid);
      new_qubit = qubit_to_slot_.size() - 1;
    }
    allocate(new_qubit);
    return new_qubit;
  }

  void allocate(Qubit qubit) {
    allocate(std::vector<Qubit>{qubit});
  }

  void free_state(){
    delete state_;
    state_ = NULL;
  }

  void allocate(const std::vector<Qubit>& qubits) {
    assert(num_slots_ < MAX_SLOTS);
    for (auto qubit : qubits) {
      assert(!has_qubit(qubit));
      assert(qubit < static_cast<Qubit>(qubit_to_slot_.size()));
    }
    slot_to_qubit_.resize(num_slots_ + qubits.size());
    for (auto qubit : qubits) {
      qubit_to_slot_[qubit] = num_slots_;
      slot_to_qubit_[num_slots_++] = qubit;
    }
    if (numa_ == 1){
      numa_free(state_,sv_size_*sizeof(Type));
      sv_size_ = (1ul << qubits.size()) * sv_size_;
      state_ = (Type*)numa_alloc_interleaved(sizeof(Type)*sv_size_);
      //state_.resize((1ul << qubits.size()) * state_.size(), 0);
      if (state_ == NULL){
        std::cout << "NUMA Memory allocation failed!" << std::endl;
        exit(EXIT_FAILURE);
      }
      /*else{
        std::cout << "NUMA Memory allocation SUCCESS!" << std::endl;
      }*/
      #pragma omp parallel for num_threads(init_num_thread)
      for (size_t i = 0; i < sv_size_; i++){
        state_[i] = 0;
      }
      
    }
    else if (numa_ == 0){
      free(state_);
      state_ = NULL;
      sv_size_ = (1ul << qubits.size()) * sv_size_;
      posix_memalign((void **)&state_, ALIGNMENT, sizeof(Type)*sv_size_);
      //state_ = (Type*)malloc(sizeof(Type)*sv_size_);
      //state_.resize((1ul << qubits.size()) * state_.size(), 0);
      if (state_ == NULL){
        std::cout << "NON-NUMA Memory allocation failed!" << std::endl;
        exit(EXIT_FAILURE);
      }
     /* else{
        std::cout << "NON-NUMA Memory allocation SUCCESS!" << std::endl;
      }*/
      //{
        #pragma omp parallel for num_threads(init_num_thread)
        for (size_t i = 0; i < sv_size_; i++){
          state_[i] = 0;
        }
      //}
    }
    else{
      free(state_);
      state_ = NULL;
      sv_size_ = (1ul << qubits.size()) * sv_size_;
      posix_memalign((void **)&state_, ALIGNMENT, sizeof(Type)*sv_size_);
      //state_ = (Type*)malloc(sizeof(Type)*sv_size_);
      //state_.resize((1ul << qubits.size()) * state_.size(), 0);
      if (state_ == NULL){
        std::cout << "NON-NUMA Memory allocation failed!" << std::endl;
        exit(EXIT_FAILURE);
      }
     /* else{
        std::cout << "NON-NUMA Memory allocation SUCCESS!" << std::endl;
      }*/
      for (size_t i = 0; i < sv_size_; i++){
        state_[i] = 0;
      }
    }
    
    //memset(state_,0,sv_size_);
  }

  Qubit slot_to_qubit(Slot slot) const {
    /*
    if (slot < 0 || slot >= num_slots_){
      std::cout << "crashing slot " << slot << std::endl;
      for (auto q: qubit_to_slot_){
      std::cout << q << " ";
    }
    std::cout << std::endl;
    for (auto s: slot_to_qubit_){
      std::cout << s << " ";
    }
    std::cout << std::endl;
    }*/
    assert(slot >= 0 && slot < num_slots_);
    return slot_to_qubit_[slot];
  }

    std::vector<LoopInfo> loop_info_for_slots(
        const std::vector<Slot>& loop_slots) {
      assert(std::is_sorted(loop_slots.begin(), loop_slots.end()));
      std::vector<LoopInfo> loop_info;
      for (int i = -1 + static_cast<int>(loop_slots.size()); i >= 0; --i) {
        if (i < static_cast<int>(loop_slots.size())-1 &&
            loop_slots[i] == loop_slots[i + 1] - 1) {
          auto [count, skip] = loop_info.back();
          count *= 2;
          skip /= 2;
          loop_info.back() = std::make_pair(count, skip);
        } else {
          loop_info.push_back({2, 1ul << loop_slots[i]});
        }
      }
      return loop_info;
    }

/*
  template <typename Func>
  inline void slot_loop(Func&& func, std::vector<Slot> slots std::vector<LoopInfo>& loop_info) {
  //void slot_loop(const Func& func, std::vector<Slot> slots) {
    
    //std::cerr << __PRETTY_FUNCTION__ << " " << __LINE__ << "\n";

    // std::cerr<<"Slot loop. Loop info:\n";*/
    /*for(const auto& li: loop_info) {
       std::cout<<std::get<0>(li)<<" "<<std::get<1>(li)<<"\n";
    }*/
    //std::cout << "size of loopinfo is " << loop_info.size() << std::endl;
    //std::cerr << __PRETTY_FUNCTION__ << " " << __LINE__ << "\n";
   /* loop(loop_info, 0, 0, [&](auto loop_itr, auto trip_count, int tid) {
      //auto start_slot_loop = std::chrono::high_resolution_clock::now();
      //std::forward<Func>(func)(loop_itr, trip_count, tid);
      //auto slot_loop_elapsed = std::chrono::high_resolution_clock::now() - start_slot_loop;
      //slot_loop_timer += std::chrono::duration_cast<std::chrono::microseconds>(slot_loop_elapsed).count();
      func(loop_itr, trip_count, tid);
    });*/
   // loop(loop_info,0,0,0, std::forward<Func>(func));
    //loop(loop_info, 0, 0, std::forward<Func>(func));
    //std::cerr << __PRETTY_FUNCTION__ << " " << __LINE__ << "\n";
  //}

  template <typename Func>
  void qubit_loop(Func&& func, const std::vector<Qubit>& qubits) {
    // unique qubits
    assert((std::set<Qubit>{qubits.begin(), qubits.end()}.size()) ==
           qubits.size());
    //std::cerr << __PRETTY_FUNCTION__ << " " << __LINE__ << "\n";
    std::vector<Slot> slots(qubits.size());
    //std::cout << "inside qubit loop the size of remaining qubit "<<qubits.size() << std::endl;  
    //std::cerr << __PRETTY_FUNCTION__ << " " << __LINE__ << "\n";  
    for (size_t i = 0; i < qubits.size(); ++i) {
      slots[i] = qubit_to_slot_[qubits[i]];
      //std::cout << "slot is " << slots[i] << std::endl;
    }
    //std::cout << __PRETTY_FUNCTION__ << " " << __LINE__ << "\n";
    std::sort(slots.begin(), slots.end());
    // unique slots
    //std::cerr << __PRETTY_FUNCTION__ << " " << __LINE__ << "\n";
    assert((std::set<Slot>{slots.begin(), slots.end()}.size()) == slots.size());
    std::vector<LoopInfo> loop_info = loop_info_for_slots(slots);
    //auto start_slot_loop = std::chrono::high_resolution_clock::now();
    std::vector<size_t> count(loop_info.size());
    std::vector<size_t> skip(loop_info.size());
    std::vector<size_t> hi(loop_info.size());
    for (size_t i = 0; i < loop_info.size(); i++) {
      std::tie(count[i], skip[i]) = loop_info[i];
      hi[i] = count[i] * skip[i];
    }
    loop(loop_info,0,0,STRIDE,count,skip,hi, std::forward<Func>(func));
    //slot_loop(std::forward<Func>(func), loop_info);
    //auto slot_loop_elapsed = std::chrono::high_resolution_clock::now() - start_slot_loop;
    //slot_loop_timer += std::chrono::duration_cast<std::chrono::microseconds>(slot_loop_elapsed).count();
    //std::cout << __PRETTY_FUNCTION__ << " " << __LINE__ << "\n";
  }

  // template <typename Func>
  // size_t slot_loop(Func&& func, size_t num_controls, const Slot control_slots[],
  //                size_t num_targets, const Slot target_slots[]) {
  //   for (size_t i=0; i<num_controls; ++i) {
  //     assert(has_slot(control_slots[i]));
  //   }
  //   for (size_t i = 0; i < num_targets; ++i) {
  //     assert(has_slot(target_slots[i]));
  //   }
  //   std::set<Slot> all_slots{&target_slots[0], &target_slots[num_targets]};
  //   all_slots.insert(&control_slots[0], &control_slots[num_targets]);
  //   // unique slots
  //   assert(all_slots.size() == num_controls + num_targets);
  //   auto loop_slots = remaining_slots(all_slots);

  //   std::vector<LoopInfo> loop_info;
  //   for (int i = -1 + static_cast<int>(loop_slots.size()); i >= 0; --i) {
  //     if (i < static_cast<int>(loop_slots.size()) && loop_slots[i] == loop_slots[i + 1] - 1) {
  //       auto [count, skip] = loop_info.back();
  //       count *= 2;
  //       skip /= 2;
  //       loop_info.back() = std::make_pair(count, skip);
  //     } else {
  //       loop_info.push_back({2, 1ul << loop_slots[i]});
  //     }
  //   }
  //   uint64_t control_offset = 0;
  //   for (size_t i=0; i<num_controls; ++i) {
  //     control_offset += (1ul << control_slots[i]);
  //   }
  //   return loop(loop_info, 0, 0, [&](auto loop_itr, auto /*trip_count*/) {
  //     std::forward<Func>(func)(&state_[loop_itr + control_offset], num_targets,
  //                              target_slots);
  //   });
  // }

  // template <typename Func>
  // void qubit_loop(Func&& func, size_t num_controls, const Qubit control_qubits[],
  //                 size_t num_targets, const Qubit target_qubits[]) {
  //   Slot control_slots[num_controls];
  //   Slot target_slots[num_targets];
  //   for (size_t i = 0; i < num_controls; ++i) {
  //     control_slots[i] = qubit_to_slot_[control_qubits[i]];
  //   }
  //   for (size_t i = 0; i < num_targets; ++i) {
  //     target_slots[i] = qubit_to_slot_[target_qubits[i]];
  //   }
  //   slot_loop(std::forward<Func>(func), num_controls, control_slots,
  //             num_targets, target_slots);
  // }

  template <typename Gate>
  StateVector& apply(Gate&& gate, size_t num_targets,
                  const Qubit target_qubits[]) {
    return apply(std::forward<Gate>(gate), 0, nullptr, num_targets,
               target_qubits);
  }

  template <typename Gate>
  StateVector& apply(Gate&& gate, size_t num_controls,
                  const Qubit control_qubits[], size_t num_targets,
                  const Qubit target_qubits[]) {
    Slot control_slots[num_controls];
    Slot target_slots[num_targets];

    // compute control offsets
    size_t control_offset = 0;
    for (size_t i = 0; i < num_controls; ++i) {
      if (!is_control_qubit(control_qubits[i])) {
        EXPECTS(has_qubit(control_qubits[i]), "Operation on an unallocated qubit");
        control_slots[i] = qubit_to_slot(control_qubits[i]);
        control_offset += (1ul << control_slots[i]);
      }
    }
    for (size_t i = 0; i < num_targets; ++i) {
      EXPECTS(has_qubit(target_qubits[i]), "Operation on an unallocated qubit");
      target_slots[i] = qubit_to_slot(target_qubits[i]);
    }
    // std::cerr<<__FUNCTION__<<" "<<__LINE__<<"\n";
    // check all slots are unique
    std::set<Slot> all_slots{&target_slots[0], &target_slots[num_targets]};
    all_slots.insert(&control_slots[0], &control_slots[num_controls]);
    assert(all_slots.size() == num_controls + num_targets);
    
    //slots to loop over -- non-control and non-target slots
    auto loop_slots = remaining_slots(all_slots);
    /*slot_loop(
        [&](size_t loop_itr, size_t trip_count) {
          gate(&state_[control_offset + loop_itr], num_targets, target_slots);
        },
        loop_slots);*/
    std::sort(loop_slots.begin(), loop_slots.end());
    // unique slots
    //std::cerr << __PRETTY_FUNCTION__ << " " << __LINE__ << "\n";
    assert((std::set<Slot>{loop_slots.begin(), loop_slots.end()}.size()) == loop_slots.size());
    std::vector<LoopInfo> loop_info = loop_info_for_slots(loop_slots);
    /*slot_loop(
        [&](size_t loop_itr, size_t trip_count, int tid) {
          gate(&state_[control_offset + loop_itr], num_targets, target_slots);
          gate_counter++;
        },
        loop_info);*/
    std::vector<size_t> count(loop_info.size());
    std::vector<size_t> skip(loop_info.size());
    std::vector<size_t> hi(loop_info.size());
    for (size_t i = 0; i < loop_info.size(); i++) {
      std::tie(count[i], skip[i]) = loop_info[i];
      hi[i] = count[i] * skip[i];
    }
    //auto start_slot_loop = std::chrono::high_resolution_clock::now();  
    #ifndef AVX512
       loop(loop_info,0,0,STRIDE,count,skip,hi,[&](size_t loop_itr, size_t trip_count, int tid) {
          gate(&state_[control_offset + loop_itr], num_targets, target_slots);
        });
    #else
    size_t N = loop_info.size();
    if (N == 1)
      LOOP(loop_info,0,0,STRIDE,count,skip,hi,[&](size_t loop_itr, size_t trip_count, int tid) {
          gate(&state_[control_offset + loop_itr], num_targets, target_slots);
        });
    if (N == 2)
      LOOP_2(loop_info,0,0,STRIDE,count,skip,hi,[&](size_t loop_itr, size_t trip_count, int tid) {
          gate(&state_[control_offset + loop_itr], num_targets, target_slots);
        });
    if (N == 3)
      LOOP_3(loop_info,0,0,STRIDE,count,skip,hi,[&](size_t loop_itr, size_t trip_count, int tid) {
          gate(&state_[control_offset + loop_itr], num_targets, target_slots);
        });
    if (N == 4)
      LOOP_4(loop_info,0,0,STRIDE,count,skip,hi,[&](size_t loop_itr, size_t trip_count, int tid) {
          gate(&state_[control_offset + loop_itr], num_targets, target_slots);
        });
    if (N == 5)
      LOOP_5(loop_info,0,0,STRIDE,count,skip,hi,[&](size_t loop_itr, size_t trip_count, int tid) {
          gate(&state_[control_offset + loop_itr], num_targets, target_slots);
        });
    if (N == 6)
      LOOP_6(loop_info,0,0,STRIDE,count,skip,hi,[&](size_t loop_itr, size_t trip_count, int tid) {
          gate(&state_[control_offset + loop_itr], num_targets, target_slots);
        });
    if (N == 7)
      LOOP_7(loop_info,0,0,STRIDE,count,skip,hi,[&](size_t loop_itr, size_t trip_count, int tid) {
          gate(&state_[control_offset + loop_itr], num_targets, target_slots);
        });
    if (N == 8)
      LOOP_8(loop_info,0,0,STRIDE,count,skip,hi,[&](size_t loop_itr, size_t trip_count, int tid) {
          gate(&state_[control_offset + loop_itr], num_targets, target_slots);
        });
    if (N == 9)
      LOOP_9(loop_info,0,0,STRIDE,count,skip,hi,[&](size_t loop_itr, size_t trip_count, int tid) {
          gate(&state_[control_offset + loop_itr], num_targets, target_slots);
        });
    if (N == 10)
      LOOP_10(loop_info,0,0,STRIDE,count,skip,hi,[&](size_t loop_itr, size_t trip_count, int tid) {
          gate(&state_[control_offset + loop_itr], num_targets, target_slots);
        });
   /* if (N > 10)
      loop(loop_info,0,0,STRIDE,count,skip,hi,[&](size_t loop_itr, size_t trip_count, int tid) {
          gate(&state_[control_offset + loop_itr], num_targets, target_slots);
        });*/
    #endif
    //auto slot_loop_elapsed = std::chrono::high_resolution_clock::now() - start_slot_loop;
    //slot_loop_timer += std::chrono::duration_cast<std::chrono::microseconds>(slot_loop_elapsed).count();
    //std::cout << "the main loop takes " << slot_loop_timer << std::endl;
    //std::cout << "total gates " << std::accumulate(gate_counter_t.begin(),gate_counter_t.end(),0)<< std::endl;
    return *this;
  }



  template <typename Gate>
  StateVector& apply_part(Gate&& gate, size_t num_targets,
                   Slot target_slots[], size_t control_offset,std::vector<LoopInfo>& loop_info, std::vector<size_t>& count, std::vector<size_t>& skip, std::vector<size_t>& hi) {
    //auto start = std::chrono::high_resolution_clock::now();
   
    /*slot_loop(
        [&](size_t loop_itr, size_t trip_count, int tid) {
          gate(&state_[control_offset + loop_itr], num_targets, target_slots);
          gate_counter++;
        },
        loop_info);*/
    #ifndef AVX512
      loop(loop_info,0,0,STRIDE,count,skip,hi,[&](size_t loop_itr, size_t trip_count, int tid) {
          gate(&state_[control_offset + loop_itr], num_targets, target_slots);
        });
    #else
    size_t N = loop_info.size();
    if (N == 1)
      LOOP(loop_info,0,0,STRIDE,count,skip,hi,[&](size_t loop_itr, size_t trip_count, int tid) {
          gate(&state_[control_offset + loop_itr], num_targets, target_slots);
        });
    if (N == 2)
      LOOP_2(loop_info,0,0,STRIDE,count,skip,hi,[&](size_t loop_itr, size_t trip_count, int tid) {
          gate(&state_[control_offset + loop_itr], num_targets, target_slots);
        });
    if (N == 3)
      LOOP_3(loop_info,0,0,STRIDE,count,skip,hi,[&](size_t loop_itr, size_t trip_count, int tid) {
          gate(&state_[control_offset + loop_itr], num_targets, target_slots);
        });
    if (N == 4)
      LOOP_4(loop_info,0,0,STRIDE,count,skip,hi,[&](size_t loop_itr, size_t trip_count, int tid) {
          gate(&state_[control_offset + loop_itr], num_targets, target_slots);
        });
    if (N == 5)
      LOOP_5(loop_info,0,0,STRIDE,count,skip,hi,[&](size_t loop_itr, size_t trip_count, int tid) {
          gate(&state_[control_offset + loop_itr], num_targets, target_slots);
        });
    if (N == 6)
      LOOP_6(loop_info,0,0,STRIDE,count,skip,hi,[&](size_t loop_itr, size_t trip_count, int tid) {
          gate(&state_[control_offset + loop_itr], num_targets, target_slots);
        });
    if (N == 7)
      LOOP_7(loop_info,0,0,STRIDE,count,skip,hi,[&](size_t loop_itr, size_t trip_count, int tid) {
          gate(&state_[control_offset + loop_itr], num_targets, target_slots);
        });
    if (N == 8)
      LOOP_8(loop_info,0,0,STRIDE,count,skip,hi,[&](size_t loop_itr, size_t trip_count, int tid) {
          gate(&state_[control_offset + loop_itr], num_targets, target_slots);
        });
    if (N == 9)
      LOOP_9(loop_info,0,0,STRIDE,count,skip,hi,[&](size_t loop_itr, size_t trip_count, int tid) {
          gate(&state_[control_offset + loop_itr], num_targets, target_slots);
        });
    if (N == 10)
      LOOP_10(loop_info,0,0,STRIDE,count,skip,hi,[&](size_t loop_itr, size_t trip_count, int tid) {
          gate(&state_[control_offset + loop_itr], num_targets, target_slots);
        });
    //auto elapsed = std::chrono::high_resolution_clock::now() - start;
    //timer += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    #endif
    return *this;
  }

  std::vector<Slot> get_slot_gather(const std::vector<Qubit>& qubits,StateVector& to_state_vector){
    // assume gathering sorted slots
    std::vector<Slot> this_slots, other_slots;
    for (auto q : qubits) {
      EXPECTS(has_qubit(q), "Operation on an unallocated qubit");
      this_slots.push_back(qubit_to_slot(q));
      assert(to_state_vector.has_qubit(q));
      other_slots.push_back(to_state_vector.qubit_to_slot(q));
    }
    /*for (size_t i = 0; i < this_slots.size(); ++i) {
      std::cout << "this slot is " << this_slots[i] << std::endl;
    }*/
    // assume gathering sorted slots
    assert(std::is_sorted(this_slots.begin(), this_slots.end()));
    //assume we are gathering to fill the destination state vector
    assert(other_slots.size() == static_cast<size_t>(to_state_vector.num_slots()));
    // assume string ordering: to_slots[i] = i
    assert(std::is_sorted(other_slots.begin(), other_slots.end()));
    return this_slots;
  }

   std::vector<Slot> get_slot_scatter(const std::vector<Qubit>& qubits,StateVector& from_state_vector){
    // assume gathering sorted slots
    // assume gathering sorted slots
    std::vector<Slot> this_slots, other_slots;
    for (auto q : qubits) {
      EXPECTS(has_qubit(q), "Operation on an unallocated qubit");
      this_slots.push_back(qubit_to_slot(q));
      assert(from_state_vector.has_qubit(q));
      other_slots.push_back(from_state_vector.qubit_to_slot(q));
    }
    // assume gathering sorted slots
    assert(std::is_sorted(this_slots.begin(), this_slots.end()));
    // assume we are gathering to fill the destination state vector
    assert(other_slots.size() ==
           static_cast<size_t>(from_state_vector.num_slots()));
    // assume string ordering: to_slots[i] = i
    assert(std::is_sorted(other_slots.begin(), other_slots.end()));
    return this_slots;
  }
  void gather_qubits(/*const std::vector<Qubit>& qubits,*/
                     /*const std::vector<Slot>& this_slots,*/
                     std::vector<LoopInfo>& loop_info, 
                     std::vector<size_t>& count,
                     std::vector<size_t>& skip,
                     std::vector<size_t>& hi,
                     size_t remaining_qubits_offset,
                     StateVector& to_state_vector) {
 
    
    //std::cout << __PRETTY_FUNCTION__ << " " << __LINE__ << "\n";
    //auto start_slot = std::chrono::high_resolution_clock::now();
   /* slot_loop([&](size_t loop_itr, size_t trip_count, int ){
     // to_state_vector.state_[trip_count] =
       //   state_[remaining_qubits_offset + loop_itr];
          //std::cout << "inner state index " << trip_count << std::endl;
          //std::cout << "outer state index " << remaining_qubits_offset + loop_itr << std::endl;
    }, loop_info);*/
    //std::cout << "remaining offset " << remaining_qubits_offset << std::endl; 
    loop(loop_info,0,0,STRIDE, count,skip,hi, [&](size_t loop_itr, size_t trip_count, int ){
     to_state_vector.state_[trip_count] =
       state_[remaining_qubits_offset + loop_itr];
       //std::cout << "inner state index " << trip_count << std::endl;
       //std::cout << "outer state index " <<  loop_itr << std::endl;
       //std::cout << "----------------" << std::endl;
          //std::cout << "inner state index " << trip_count << std::endl;
          //std::cout << "outer state index " << remaining_qubits_offset + loop_itr << std::endl;
    });
    //auto slot_elapsed = std::chrono::high_resolution_clock::now() - start_slot;
    //slot_loop_timer += std::chrono::duration_cast<std::chrono::microseconds>(slot_elapsed).count();
    //std::cout << "Slot inside gather take " << slot_loop_timer << std::endl;
    //std::cout << __PRETTY_FUNCTION__ << " " << __LINE__ << "\n";
  }

  void scatter_qubits(/*const std::vector<Qubit>& qubits,*/
                      /*const std::vector<Slot>& this_slots,*/
                      std::vector<LoopInfo>& loop_info,
                      std::vector<size_t>& count,
                     std::vector<size_t>& skip,
                     std::vector<size_t>& hi,
                      size_t remaining_qubits_offset,
                      StateVector& from_state_vector) {
    
    
    //std::cout << __PRETTY_FUNCTION__ << " " << __LINE__ << "\n";
    /*slot_loop(
        [&](size_t loop_itr, size_t trip_count, int) {
          state_[remaining_qubits_offset + loop_itr] =
              from_state_vector.state_[trip_count];
        },
        loop_info);*/
    loop(loop_info,0,0,STRIDE, count,skip,hi, [&](size_t loop_itr, size_t trip_count, int) {
          state_[remaining_qubits_offset + loop_itr] =
              from_state_vector.state_[trip_count];
        });
  //  std::cout << __PRETTY_FUNCTION__ << " " << __LINE__ << "\n";
  }

  Type prefetch_state(){
    Type a;
    for (size_t i = 0; i < sv_size_; i++){
      a = state_[i];  
    }
    return a;
  }

  std::vector<Qubit> qubits_ordered_by_slots(
      const std::vector<Qubit>& qubits) const {
    std::vector<Slot> slots;
    for (auto q : qubits) {
      EXPECTS(has_qubit(q), "Operation on an unallocated qubit");
      slots.push_back(qubit_to_slot(q));
    }
    std::sort(slots.begin(), slots.end());
    std::vector<Qubit> ret;
    for (auto s : slots) {
      ret.push_back(slot_to_qubit(s));
    }
    return ret;
  }

  void get_qubit_slot_map(){
    for (auto q: qubit_to_slot_){
      std::cout << q << " ";
    }
    std::cout << std::endl;
    for (auto s: slot_to_qubit_){
      std::cout << s << " ";
    }
    std::cout << std::endl;
  }

  // template <typename Func, typename... Qubits>
  // void qubit_loop(Func&& func, SvSim::Qubit qubit, SvSim::Qubits&&... qubits) {
  //   Qubit target_qubits[] = {qubit, std::forward<Qubits>(qubits)...};
  //   qubit_loop(std::forward<Func>(func), 0, nullptr, std::size(target_qubits),
  //              target_qubits);
  // }

  // void I(Qubit) {}

  // void X(Qubit qubit) { qubit_loop(BasicGates::X(), SvSim::qubit); }

  // void Y(Qubit qubit) { qubit_loop(BasicGates::Y(), SvSim::qubit); }

  // void H(Qubit qubit) { qubit_loop(BasicGates::H(), SvSim::qubit); }

  // void Z(Qubit qubit) { qubit_loop(BasicGates::Z(), SvSim::qubit); }

  // void S(Qubit qubit) { qubit_loop(BasicGates::S(), SvSim::qubit); }

  // // void T(Qubit qubit) { qubit_loop(BasicGates::T());, SvSim::qubit }

  // void RX(Qubit qubit, Type theta) { qubit_loop(BasicGates::RX(theta), SvSim::qubit); }

  // void RY(Qubit qubit, Type theta) { qubit_loop(BasicGates::RY(theta), SvSim::qubit); }

  // void RZ(Qubit qubit, Type theta) { qubit_loop(BasicGates::RZ(theta), SvSim::qubit); }

  // void RI(Qubit qubit, Type theta) { qubit_loop(BasicGates::RI(theta), SvSim::qubit); }

  // void R1(Qubit qubit, Type theta) { qubit_loop(BasicGates::R1(theta), SvSim::qubit); }

  // void RXFrac(Qubit qubit, unsigned numerator, unsigned power) {
  //   qubit_loop(BasicGates::RXFrac(numerator, power), SvSim::qubit);
  // }

  // void RYFrac(Qubit qubit, unsigned numerator, unsigned power) {
  //   qubit_loop(BasicGates::RYFrac(numerator, power), SvSim::qubit);
  // }

  // void RZFrac(Qubit qubit, unsigned numerator, unsigned power) {
  //   qubit_loop(BasicGates::RZFrac(numerator, power), SvSim::qubit);
  // }

  // void RIFrac(Qubit qubit, unsigned numerator, unsigned power) {
  //   qubit_loop(BasicGates::RIFrac(numerator, power), SvSim::qubit);
  // }

  // void R1Frac(Qubit qubit, unsigned numerator, unsigned power) {
  //   qubit_loop(BasicGates::R1Frac(numerator, power), SvSim::qubit);
  // }

  // void SWAP(Qubit qubit0, SvSim::Qubit qubit1) {
  //   qubit_loop(BasicGates::SWAP(), std::array<Qubit, 2>{qubit0, SvSim::qubit1});
  // }

  // void CNOT(Qubit control, SvSim::Qubit target) { CX(control, target); }

  // void CX(Qubit control, SvSim::Qubit target) {
  //   qubit_loop(BasicGates::X(), std::array<Qubit, 1>{control},
  //              std::array<Qubit, 1>{target});
  // }

  // void CY(Qubit control, SvSim::Qubit target) {
  //   qubit_loop(BasicGates::Y(), std::array<Qubit, 1>{control},
  //              std::array<Qubit, 1>{target});
  // }

  // void CZ(Qubit control, SvSim::Qubit target) {
  //   qubit_loop(BasicGates::Z(), std::array<Qubit, 1>{control},
  //              std::array<Qubit, 1>{target});
  // }

  // void CSWAP(Qubit control, SvSim::Qubit qubit0, SvSim::Qubit qubit1) {
  //   qubit_loop(BasicGates::SWAP(), std::array<Qubit, 1>{control},
  //              std::array<Qubit, 2>{qubit0, SvSim::qubit1});
  // }

  // template <unsigned N>
  // void CSTAR_X(const std::array<Qubit, N>& controls, SvSim::Qubit target) {
  //   qubit_loop(BasicGates::X(), controls, std::array<Qubit, 1>{target});
  // }

  // void CCX(std::array<Qubit, 2> controls, SvSim::Qubit target) {
  //   qubit_loop(BasicGates::X(), controls, std::array<Qubit, 1>{target});
  // }

  // void C3X(std::array<Qubit, 3> controls, SvSim::Qubit target) {
  //   qubit_loop(BasicGates::X(), controls, std::array<Qubit, 1>{target});
  // }

  // void C4X(std::array<Qubit, 4> controls, SvSim::Qubit target) {
  //   qubit_loop(BasicGates::X(), controls, std::array<Qubit, 1>{target});
  // }

  bool has_qubit(Qubit qubit) const {
    // std::cerr << "qubit_to_slot.size=" << qubit_to_slot_.size() << "\n";
    // std::cerr << "qubit=" << qubit << "\n";
    // std::cerr << "qubit_to_slot[qubit]=" << qubit_to_slot_[qubit] << "\n";
    return (qubit >= 0) &&
           (qubit < static_cast<Qubit>(qubit_to_slot_.size())) &&
           (qubit_to_slot_[qubit] != qubit_invalid);
  }

  bool has_slot(Slot slot) const { return slot >= 0 && slot < num_slots_; }

  bool is_control_qubit(Qubit qubit) const {
    return enabled_control_qubits_.find(qubit) != enabled_control_qubits_.end();
  }

  std::vector<Qubit> remaining_qubits(const std::set<Slot>& in_qubits) {
    std::vector<Qubit> out_qubits;
    for (Slot slot = 0; slot < num_slots_; ++slot) {
      if (has_slot(slot) && std::find(in_qubits.begin(), in_qubits.end(),
                                      slot_to_qubit(slot)) == in_qubits.end()) {
        out_qubits.push_back(slot_to_qubit(slot));
      }
    }
    return out_qubits;
  }

  void project(const std::vector<Qubit>& qubits, std::vector<int> values = {},
               bool renormalize = true) {
    for (auto q : qubits) {
      EXPECTS(has_qubit(q), "Operation on an unallocated qubit");
    }
    for(auto v: values) {
      assert(v==0 || v==1);
    }
    //unique qubits
    assert((std::set<Qubit>{qubits.begin(), qubits.end()}.size()) == qubits.size());
    if(values.size() != qubits.size()) { //measure for zero value by default
      values.resize(qubits.size(), 0);
    }
    size_t p_state_size_ = 0;
    Type *projected_state_ = NULL;
    p_state_size_ = 1ul<<(num_slots_ - qubits.size());
    if (numa_ == 1){
      projected_state_ = (Type*)numa_alloc_interleaved(sizeof(Type)*p_state_size_);
    }
    else
      projected_state_ = (Type*)malloc(sizeof(Type)*p_state_size_);
    assert(projected_state_ != NULL);
    //std::vector<Type> projected_state_(1ul<<(num_slots_ - qubits.size()));
    size_t project_offset = 0;
    for (size_t i = 0; i < qubits.size(); ++i) {
      project_offset += (values[i] << qubit_to_slot(qubits[i]));
    }
    std::vector<Qubit> outer_qubits = remaining_qubits(
        std::set<Qubit>{qubits.begin(), qubits.end()});

    qubit_loop(
        [&](size_t loop_itr, size_t trip_count,int) {
          projected_state_[trip_count] =
              state_[project_offset + loop_itr];
        },
        outer_qubits);

    if (renormalize) {
      double denominator = 0;
      /*for(const auto v : projected_state_) {
        denominator += std::abs(v * v);
      }*/
      for(size_t i = 0; i < p_state_size_; i++){
        denominator += std::abs(projected_state_[i] * projected_state_[i]);
      }
      assert(denominator >= 0 && denominator <= 1);
      const double threshold = 1e-12;
      if (denominator > threshold) {
        /*for (auto& p : projected_state_) {
          p /= denominator;
        }*/
        for(size_t i = 0; i < p_state_size_; i++){
          projected_state_[i] /= denominator;
        }
      }
    }

    //update state
    stored_qubits_ = outer_qubits;
    //no change to enabled_control_qubits_;
    num_slots_ -= qubits.size();
    for (auto q : qubits) {
      slot_to_qubit_[qubit_to_slot_[q]] = qubit_invalid;
      qubit_to_slot_[q] = slot_invalid;
    }
    state_ = projected_state_;
  }

  double probability(Qubit qubit, int value=0) {
    EXPECTS(has_qubit(qubit), "Operation on an unallocated qubit");
    EXPECTS(value == 0 || value == 1,
            "Did you intend to request probability of qubit being of value "
            "neither zero nor one? Check the function parameter.");

    //std::cerr << __PRETTY_FUNCTION__ << " " << __LINE__ << "\n";
    std::vector<Qubit> outer_qubits = remaining_qubits(std::set<Qubit>{qubit});
    //std::cerr << __PRETTY_FUNCTION__ << " " << __LINE__ << "\n";
    double probability_of_0 = 0;
    //std::cerr<<__PRETTY_FUNCTION__<<" "<<__LINE__<<"\n";
    qubit_loop(
        [&](size_t loop_itr, size_t trip_count,int) {
          // std::cerr << "qubit loop. loop_itr :" << loop_itr
          //           << " value:" << state_[loop_itr] << "\n";
          probability_of_0 += std::abs(state_[loop_itr] * state_[loop_itr]);
        },
        outer_qubits);
    // std::cerr << __FUNCTION__ << " " << __LINE__ << "\n";

    // assert(probability_of_0 >= 0 && probability_of_0 <= 1);
    if(value == 0) {
      return probability_of_0;
    } else {
      return 1 - probability_of_0;
    }
  }

  std::vector<std::vector<Value>> measure(const std::vector<Qubit>& qubits,
                                          unsigned num_samples = 1) {
    if (num_samples == 0) {
      return {};
    }
    std::vector<Slot> slots;
    for (auto q : qubits) {
      EXPECTS(has_qubit(q), "Operation on an unallocated qubit");
      slots.push_back(qubit_to_slot(q));
    }
    double sample_probabilities[num_samples];
    std::random_device
        rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with
                             // rd()
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for(unsigned i=0; i<num_samples; ++i) {
      sample_probabilities[i] = dis(gen);
    }
    std::sort(sample_probabilities, sample_probabilities + num_samples);
    for (unsigned i = 0; i < num_samples; ++i) {
      std::cerr<<"Prob(measu)["<<i<<"] :"<<sample_probabilities[i]<<"\n";
    }

    std::vector<std::vector<Value>> ret;
    double cumulative_prob=0;
    unsigned sample_pos = 0;
    //for(size_t pos = 0; pos < state_.size() && sample_pos < num_samples; ++pos) {
    for(size_t pos = 0; pos < sv_size_ && sample_pos < num_samples; ++pos) {
      //while(sample_probabilities[sample_pos] < cumulative_prob+std::abs(state_[pos]*state_[pos])) {
      while(sample_pos < num_samples && sample_probabilities[sample_pos] < cumulative_prob+std::abs(state_[pos]*state_[pos])){
        //report qubits in this state
        std::cerr<<"Measurement outcome="<<std::hex<<pos<<"\n";
        std::vector<Value> result;
        for(Slot s : slots){
          result.push_back((pos>>s)&1);
        } 
        //ret.push_back(slot_bit_vector_to_qubits(pos));
        ret.push_back(result);
        sample_pos += 1;
      }
      cumulative_prob += std::abs(state_[pos] * state_[pos]);
    }
    std::cerr << "Cumulative probability:" << cumulative_prob << "\n";
    for (; sample_pos < num_samples; ++sample_pos) {
      //std::cerr << "Measurement outcome=" << std::hex << state_.size()-1 << "\n";
      std::cerr << "Measurement outcome=" << std::hex << sv_size_-1 << "\n";
      std::vector<Value> result;
      for (Slot s : slots){
        result.push_back(1);
      }
      //ret.push_back(slot_bit_vector_to_qubits(state_.size() - 1));
      ret.push_back(result);
    }
    assert(ret.size() == num_samples);
    return ret;
  }

  std::vector<Qubit> slot_bit_vector_to_qubits(SlotBitVector sbv) {
    std::vector<Qubit> ret;
    for (Slot i = 0; i < num_slots_; ++i) {
      if ((sbv >> i) & 1) {
        ret.push_back(slot_to_qubit(i));
      }
    }
    return ret;
  }

  std::vector<Slot> remaining_slots(const std::set<Slot>& in_slots) {
    std::vector<Slot> outer_slots;
    for (Slot slot = 0; slot < num_slots_; ++slot) {
      if (std::find(in_slots.begin(), in_slots.end(), slot) == in_slots.end()) {
        outer_slots.push_back(slot);
      }
    }
    return outer_slots;
  }

  bool state_equal(StateVector& rsim){
      for (size_t i = 0; i < sv_size_; ++i) {
        if (!approx_equal(state_[i], rsim.state_[i])) {
          return false;
        }
      }
      return true;
  }
  
  Type * state() {return state_;}

  const Type* state() const { return state_; }

 private:
  std::vector<Qubit> identity_qubit_mapping(Qubit num_qubits) {
    std::vector<Qubit> ret;
    for (Qubit q = 0; q < num_qubits; ++q) {
      ret.push_back(q);
    }
    return ret;
  }

  friend void swap(StateVector& first, StateVector& second) {
    using std::swap;
    swap(first.stored_qubits_, second.stored_qubits_);
    swap(first.num_slots_, second.num_slots_);
    //swap(first.max_num_qubits_, second.max_num_qubits_);
    swap(first.stored_qubits_, second.stored_qubits_);
    swap(first.slot_to_qubit_, second.slot_to_qubit_);
    swap(first.qubit_to_slot_, second.qubit_to_slot_);
    swap(first.state_, second.state_);
  }

  //std::vector<Type>& state() { return state_; }
  //const std::vector<Type>& state() const { return state_; }

 


  std::vector<Qubit> stored_qubits_;
  std::set<Qubit> enabled_control_qubits_;
  Slot num_slots_;
  //Qubit max_num_qubits_;
  std::vector<Qubit> slot_to_qubit_;
  std::vector<Slot> qubit_to_slot_;
  //std::vector<Type> state_;
  Type *state_;
  size_t sv_size_;
  int numa_;

  friend bool operator == (const StateVector& sv1, const StateVector& sv2) {
    if (sv1.stored_qubits_ == sv2.stored_qubits_ &&
        sv1.enabled_control_qubits_ == sv2.enabled_control_qubits_ &&
        sv1.num_slots_ == sv2.num_slots_ &&
        sv1.slot_to_qubit_ == sv2.slot_to_qubit_ &&
        sv1.qubit_to_slot_ == sv2.qubit_to_slot_ &&
        sv1.sv_size_ == sv2.sv_size_) {
        //sv1.state_.size() == sv2.state_.size()) {
      //for (size_t i = 0; i < sv1.state_.size(); ++i) {
      for (size_t i = 0; i < sv1.sv_size_; ++i) {
        std::cout << sv1.state_[i] << std::endl;
        std::cout << sv2.state_[i] << std::endl;
        if (!approx_equal(sv1.state_[i], sv2.state_[i])) {
          return false;
        }
      }
      return true;
    }
    return false;
  }
  friend bool operator!=(const StateVector& sv1, const StateVector& sv2) {
    return !(sv1 == sv2);
  }
};  // class StateVector

class GateBuffer{
  public:
    GateBuffer(StateVector& sv,/*BasicGates::AbstractGate* */ AggregateGateFactory& ag, size_t num_controls, const Qubit control_qubits[], size_t num_targets, const Qubit target_qubits[]):
    sv_{sv},
    //ab_gate_{ab_gate},
    ag_{ag},
    num_controls_{num_controls},
    control_qubits_{control_qubits},
    num_targets_{num_targets},
    target_qubits_{target_qubits}{
      Slot control_slots[num_controls_];
      Slot target_slots[num_targets_];
  
      // compute control offsets
      size_t control_offset = 0;
      
      for (size_t i = 0; i < num_controls_; ++i) {
        if (!sv_.is_control_qubit(control_qubits_[i])) {
          EXPECTS(sv_.has_qubit(control_qubits_[i]), "Operation on an unallocated qubit");
          control_slots[i] = sv_.qubit_to_slot(control_qubits_[i]);
          control_offset += (1ul << control_slots[i]);
        }
      }
      control_offset_ = control_offset;
      for (size_t i = 0; i < num_targets_; ++i) {
         // std::cout << "target qubit is " << target_qubits_[i] << std::endl;
          EXPECTS(sv_.has_qubit(target_qubits_[i]), "Operation on an unallocated qubit");
          target_slots[i] = sv_.qubit_to_slot(target_qubits_[i]);
          target_slots_.push_back(sv_.qubit_to_slot(target_qubits_[i]));
      }
        // std::cerr<<__FUNCTION__<<" "<<__LINE__<<"\n";
        // check all slots are unique
      std::set<Slot> all_slots{&target_slots[0], &target_slots[num_targets_]};
      all_slots.insert(&control_slots[0], &control_slots[num_controls_]);
      assert(all_slots.size() == num_controls_ + num_targets_);
        
        //slots to loop over -- non-control and non-target slots
        
      auto loop_slots = sv_.remaining_slots(all_slots);
      std::sort(loop_slots.begin(), loop_slots.end());
        // unique slots
        //std::cerr << __PRETTY_FUNCTION__ << " " << __LINE__ << "\n";
      assert((std::set<Slot>{loop_slots.begin(), loop_slots.end()}.size()) == loop_slots.size());
      std::vector<LoopInfo> loop_info = sv_.loop_info_for_slots(loop_slots);
      loop_info_ = loop_info;
      std::vector<size_t> count(loop_info.size());
      std::vector<size_t> skip(loop_info.size());
      std::vector<size_t> hi(loop_info.size());
      for (size_t i = 0; i < loop_info.size(); i++) {
          std::tie(count[i], skip[i]) = loop_info[i];
          hi[i] = count[i] * skip[i];
      }
      count_ = count;
      skip_ = skip;
      hi_ = hi;
    }
   /*
  UnionGate get_ugate(){
    return ugate_;
  }*/
  /*inline BasicGates::AbstractGate* get_ab_gate(){
    return ab_gate_;
  }*/

  AggregateGateFactory& get_ag(){
    return ag_;
  }

  inline size_t get_num_control(){
    return num_controls_;
  }

  inline size_t get_num_targets(){
    return num_targets_;
  }

  inline const Qubit* get_controls(){
    return control_qubits_;
  }
  inline const Qubit* get_targets(){
    return target_qubits_;
  }

  inline std::vector<LoopInfo>& get_loop_info(){
    return loop_info_;
  }

  inline size_t get_control_offset(){
    return control_offset_;
  }

  inline std::vector<size_t>& get_count(){
    return count_;
  }

  inline std::vector<size_t>& get_skip(){
    return skip_;
  }

  inline std::vector<size_t>& get_hi(){
    return hi_;
  }

  inline std::vector<Slot>& get_target_slots(){
    return target_slots_;
  }

  private:
    /*UnionGate ugate_;*/
    //BasicGates::AbstractGate* ab_gate_;
    AggregateGateFactory ag_;
    size_t num_controls_;
    const Qubit* control_qubits_;
    size_t num_targets_;
    const Qubit* target_qubits_;
    StateVector &sv_;
    std::vector<LoopInfo> loop_info_;
    size_t control_offset_;
    std::vector<Slot> target_slots_;
    std::vector<size_t> count_;
    std::vector<size_t> skip_;
    std::vector<size_t> hi_;
};

  void apply_gate(StateVector& state_vector, AggregateGateFactory& ag_factory,
             size_t num_controls,
             const Qubit control_qubits[], size_t num_targets,
             const Qubit target_qubits[]) {
      if (ag_factory.get_type() == AGType::NOPARAM){
          AggregateGate_0& ag = ag_factory.get_ag0();
          switch (ag.gate_type_) {
            case GateType::ALLOC: state_vector.allocate(std::vector<Qubit>(target_qubits, target_qubits + num_targets)); break;
            case GateType::FREE: state_vector.project(std::vector<Qubit>(target_qubits, target_qubits + num_targets)); break;
            case GateType::I: state_vector.apply(ag.union_gate_.I_, num_controls, control_qubits, num_targets, target_qubits); break;
            case GateType::X: state_vector.apply(ag.union_gate_.X_, num_controls, control_qubits, num_targets, target_qubits); break;
            case GateType::Y: state_vector.apply(ag.union_gate_.Y_, num_controls, control_qubits, num_targets, target_qubits); break;
            case GateType::Z: state_vector.apply(ag.union_gate_.Z_, num_controls, control_qubits, num_targets, target_qubits); break;
            case GateType::H: state_vector.apply(ag.union_gate_.H_, num_controls, control_qubits, num_targets, target_qubits); break;
            case GateType::S: state_vector.apply(ag.union_gate_.S_, num_controls, control_qubits, num_targets, target_qubits); break;
            case GateType::T: state_vector.apply(ag.union_gate_.T_, num_controls, control_qubits, num_targets, target_qubits); break;
            case GateType::SWAP: state_vector.apply(ag.union_gate_.SWAP_, num_controls, control_qubits, num_targets, target_qubits); break;
            case GateType::TDG: state_vector.apply(ag.union_gate_.TDG_, num_controls, control_qubits, num_targets, target_qubits); break;
            case GateType::SDG: state_vector.apply(ag.union_gate_.SDG_, num_controls, control_qubits, num_targets, target_qubits); break;
            case GateType::invalid:
              UNREACHABLE();
              break;
            default:
            NOT_IMPLEMENTED();
          }
      }

      if (ag_factory.get_type() == AGType::ONEPARAM){
          AggregateGate_1 & ag = ag_factory.get_ag1();
          switch (ag.gate_type_) {
            case GateType::ALLOC: state_vector.allocate(std::vector<Qubit>(target_qubits, target_qubits + num_targets)); break;
            case GateType::FREE: state_vector.project(std::vector<Qubit>(target_qubits, target_qubits + num_targets)); break;
            case GateType::RX: state_vector.apply(ag.union_gate_.RX_, num_controls, control_qubits, num_targets, target_qubits); break;
            case GateType::RY: state_vector.apply(ag.union_gate_.RY_, num_controls, control_qubits, num_targets, target_qubits); break;
            case GateType::RZ: state_vector.apply(ag.union_gate_.RZ_, num_controls, control_qubits, num_targets, target_qubits); break;
            case GateType::RI: state_vector.apply(ag.union_gate_.RI_, num_controls, control_qubits, num_targets, target_qubits); break;
            case GateType::R1: state_vector.apply(ag.union_gate_.R1_, num_controls, control_qubits, num_targets, target_qubits); break;
            case GateType::U1: state_vector.apply(ag.union_gate_.U1_, num_controls, control_qubits, num_targets, target_qubits); break;
            case GateType::RXX: state_vector.apply(ag.union_gate_.RXX_, num_controls, control_qubits, num_targets, target_qubits); break;
            case GateType::RZZ: state_vector.apply(ag.union_gate_.RZZ_, num_controls, control_qubits, num_targets, target_qubits); break;
            case GateType::RYY: state_vector.apply(ag.union_gate_.RYY_, num_controls, control_qubits, num_targets, target_qubits); break;
            case GateType::invalid:
              UNREACHABLE();
              break;
            default:
              NOT_IMPLEMENTED();
          }
      }
      if (ag_factory.get_type() == AGType::TWOPARAM){
          AggregateGate_2 & ag = ag_factory.get_ag2();
          switch (ag.gate_type_) {
            case GateType::ALLOC: state_vector.allocate(std::vector<Qubit>(target_qubits, target_qubits + num_targets)); break;
            case GateType::FREE: state_vector.project(std::vector<Qubit>(target_qubits, target_qubits + num_targets)); break;
            case GateType::U2: state_vector.apply(ag.union_gate_.U2_, num_controls, control_qubits, num_targets, target_qubits); break;
            case GateType::invalid:
              UNREACHABLE();
              break;
            default:
              NOT_IMPLEMENTED();
          }
      }
      if (ag_factory.get_type() == AGType::THREEPARAM){
          AggregateGate_3 & ag = ag_factory.get_ag3();
          switch (ag.gate_type_) {
            case GateType::ALLOC: state_vector.allocate(std::vector<Qubit>(target_qubits, target_qubits + num_targets)); break;
            case GateType::FREE: state_vector.project(std::vector<Qubit>(target_qubits, target_qubits + num_targets)); break;
            case GateType::U3: state_vector.apply(ag.union_gate_.U3_, num_controls, control_qubits, num_targets, target_qubits); break;
            case GateType::invalid:
              UNREACHABLE();
              break;
            default:
              NOT_IMPLEMENTED();
          }
      }
      if (ag_factory.get_type() == AGType::C1){
          AggregateGate_C1 & ag = ag_factory.get_agC1();
          switch (ag.gate_type_) {
            case GateType::ALLOC: state_vector.allocate(std::vector<Qubit>(target_qubits, target_qubits + num_targets)); break;
            case GateType::FREE: state_vector.project(std::vector<Qubit>(target_qubits, target_qubits + num_targets)); break;
            case GateType::C1: state_vector.apply(ag.union_gate_.C1_, num_controls, control_qubits, num_targets, target_qubits); break;
            case GateType::invalid:
              UNREACHABLE();
              break;
            default:
              NOT_IMPLEMENTED();
          }
      }
      if (ag_factory.get_type() == AGType::C2){
          AggregateGate_C2 & ag = ag_factory.get_agC2();
          switch (ag.gate_type_) {
            case GateType::ALLOC: state_vector.allocate(std::vector<Qubit>(target_qubits, target_qubits + num_targets)); break;
            case GateType::FREE: state_vector.project(std::vector<Qubit>(target_qubits, target_qubits + num_targets)); break;
            case GateType::C2: state_vector.apply(ag.union_gate_.C2_, num_controls, control_qubits, num_targets, target_qubits); break;
            case GateType::invalid:
              UNREACHABLE();
              break;
            default:
              NOT_IMPLEMENTED();
          }
      }
      if (ag_factory.get_type() == AGType::FRAC){
          AggregateGate_Frac & ag = ag_factory.get_agFrac();
          switch (ag.gate_type_) {
            case GateType::ALLOC: state_vector.allocate(std::vector<Qubit>(target_qubits, target_qubits + num_targets)); break;
            case GateType::FREE: state_vector.project(std::vector<Qubit>(target_qubits, target_qubits + num_targets)); break;
            case GateType::RXFrac: state_vector.apply(ag.union_gate_.RXFrac_, num_controls, control_qubits, num_targets, target_qubits); break;
            case GateType::RYFrac: state_vector.apply(ag.union_gate_.RYFrac_, num_controls, control_qubits, num_targets, target_qubits); break;
            case GateType::RZFrac: state_vector.apply(ag.union_gate_.RZFrac_, num_controls, control_qubits, num_targets, target_qubits); break;
            case GateType::RIFrac: state_vector.apply(ag.union_gate_.RIFrac_, num_controls, control_qubits, num_targets, target_qubits); break;
            case GateType::R1Frac: state_vector.apply(ag.union_gate_.R1Frac_, num_controls, control_qubits, num_targets, target_qubits); break;
            case GateType::invalid:
              UNREACHABLE();
              break;
            default:
              NOT_IMPLEMENTED();
          }
      }
    
  }

GateBuffer apply_gate_buffer(StateVector& state_vector, AggregateGateFactory& ag_factory,
             size_t num_controls,
             const Qubit control_qubits[], size_t num_targets,
             const Qubit target_qubits[]) {
   
      GateBuffer *gatebuffer = new GateBuffer(state_vector, ag_factory, num_controls, control_qubits, num_targets, target_qubits);
      return *gatebuffer;
  }


class BoundGate {
 public:
  BoundGate() = default;
  BoundGate(const BoundGate&) = default;
  BoundGate& operator=(const BoundGate&) = default;
  BoundGate& operator=(BoundGate&&) = default;

  BoundGate(AggregateGateFactory& ag_factory, const std::vector<Qubit>& controls,
            const std::vector<Qubit>& targets)
      : ag_factory_{ag_factory},
        num_controls_{controls.size()},
        num_targets_{targets.size()} {
    std::copy_n(controls.begin(), controls.size(), controls_.begin());
    std::copy_n(targets.begin(), targets.size(), targets_.begin());
    // assert(qubits_.size() == gate_.dim());
  }

  void apply(StateVector& sv) {
    apply_gate(sv, ag_factory_, num_controls_, &controls_[0], num_targets_, &targets_[0]);
  }

  GateBuffer apply_to_buffer(StateVector& sv){
    return apply_gate_buffer(sv,ag_factory_,num_controls_,&controls_[0], num_targets_,&targets_[0]);
  }

  const std::array<Qubit, MAX_CONTROL_QUBITS>& controls() const {
    return controls_;
  }

  const std::array<Qubit, MAX_TARGET_QUBITS>& targets() const {
    return targets_;
  }

  unsigned num_controls() const {
    return num_controls_;
  }

  unsigned num_targets() const {
    return num_targets_;
  }

 private:
  AggregateGateFactory ag_factory_;
  size_t num_controls_;
  std::array<Qubit, MAX_CONTROL_QUBITS> controls_;
  size_t num_targets_;
  std::array<Qubit, MAX_TARGET_QUBITS> targets_;
};

long long obtain_accumulate_timer(){
  return timer;
}

long long obtain_gate_counter(){
  return gate_counter;
}

long long obtain_move_counter(){
  return mem_move_counter;
}



}  // namespace SvSim
#if 0
#include <bitset>
#include <cstdlib>
#include <iostream>

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"

TEST_CASE("Simple state vector checks") {
  using namespace SvSim;
  StateVector sv{1};

  CHECK_UNARY(sv.has_qubit(0));
  // sv.dump_state(std::cout);
  sv.allocate(2);
  CHECK_UNARY(sv.has_qubit(2));
  CHECK_UNARY(!sv.has_qubit(1));
  std::cout << "----------Allocated qubit 2\n";
  // sv.dump_state(std::cout);
  CHECK_NE(sv.qubit_to_slot(0), sv.qubit_to_slot(2));
  std::cerr << sv.probability(SvSim::Qubit{0}, 0)<<"\n";
  CHECK_UNARY(approx_equal(sv.probability(SvSim::Qubit{0}), 1));
  std::cerr << sv.probability(SvSim::Qubit{2}, 0) << "\n";
  CHECK_UNARY(approx_equal(sv.probability(SvSim::Qubit{2}, 1), 0));
}

TEST_CASE("Simple gate checks") {
  using namespace SvSim;
  using namespace std::complex_literals;
  StateVector sv{1};
  StateVector sv_01(1, {0, 1});
  StateVector sv_10(1, {1, 0});
  StateVector sv_0n1(1, {0, -1});
  StateVector sv_n10(1, {-1, 0});
  StateVector sv_0i(1, {0, 1.0i});
  StateVector sv_0si(1, {0, std::sqrt(1.0i)});
  StateVector sv_0ni(1, {0, -1.0i});
  StateVector sv_i0(1, {1.0i, 0});
  StateVector sv_ni0(1, {-1.0i, 0});
  StateVector sv_hh(1, {Constants::inv_sqrt_2, Constants::inv_sqrt_2});
  StateVector sv_hnh(1, {Constants::inv_sqrt_2, -Constants::inv_sqrt_2});
  SvSim::Qubit targets[] = {0};

  sv = sv_10;
  sv.apply(BasicGates::I(), 0, nullptr, 1, targets);
  CHECK_UNARY(sv == sv_10);
  sv = sv_01;
  sv.apply(BasicGates::I(), 0, nullptr, 1, targets);
  CHECK_UNARY(sv == sv_01);

  sv = sv_10;
  sv.apply(BasicGates::X(), 0, nullptr, 1, targets);
  CHECK_UNARY(sv == sv_01);
  sv = sv_01;
  sv.apply(BasicGates::X(), 0, nullptr, 1, targets);
  CHECK_UNARY(sv == sv_10);

  sv = sv_10;
  sv.apply(BasicGates::Y(), 0, nullptr, 1, targets);
  CHECK_UNARY(sv == sv_0i);
  sv = sv_01;
  sv.apply(BasicGates::Y(), 0, nullptr, 1, targets);
  CHECK_UNARY(sv == sv_ni0);

  sv = sv_10;
  sv.apply(BasicGates::Z(), 0, nullptr, 1, targets);
  CHECK_UNARY(sv == sv_10);
  sv = sv_01;
  sv.apply(BasicGates::Z(), 0, nullptr, 1, targets);
  CHECK_UNARY(sv == sv_0n1);

  sv = sv_10;
  sv.apply(BasicGates::H(), 0, nullptr, 1, targets);
  CHECK_UNARY(sv == sv_hh);
  sv = sv_01;
  sv.apply(BasicGates::H(), 0, nullptr, 1, targets);
  CHECK_UNARY(sv == sv_hnh);

  sv = sv_10;
  sv.apply(BasicGates::S(), 0, nullptr, 1, targets);
  CHECK_UNARY(sv == sv_10);
  sv = sv_01;
  sv.apply(BasicGates::S(), 0, nullptr, 1, targets);
  CHECK_UNARY(sv == sv_0i);

  sv = sv_10;
  sv.apply(BasicGates::T(), 0, nullptr, 1, targets);
  CHECK_UNARY(sv == sv_10);
  sv = sv_01;
  sv.apply(BasicGates::T(), 0, nullptr, 1, targets);
  CHECK_UNARY(sv == sv_0si);

  sv = sv_10;
  sv.apply(BasicGates::SDG(), 0, nullptr, 1, targets);
  sv.apply(BasicGates::S(), 0, nullptr, 1, targets);
  CHECK_UNARY(sv == sv_10);
  sv = sv_01;
  sv.apply(BasicGates::SDG(), 0, nullptr, 1, targets);
  sv.apply(BasicGates::S(), 0, nullptr, 1, targets);
  CHECK_UNARY(sv == sv_01);

  sv = sv_10;
  sv.apply(BasicGates::TDG(), 0, nullptr, 1, targets);
  sv.apply(BasicGates::T(), 0, nullptr, 1, targets);
  CHECK_UNARY(sv == sv_10);
  sv = sv_01;
  sv.apply(BasicGates::TDG(), 0, nullptr, 1, targets);
  sv.apply(BasicGates::T(), 0, nullptr, 1, targets);
  CHECK_UNARY(sv == sv_01);

  //todo: check all other gates
}

TEST_CASE("Simple gates: 2 qubit gates") {
  using namespace SvSim;
  using namespace std::complex_literals;
  SvSim::Qubit targets[] = {0, 1};
  StateVector sv;
  StateVector sv_1000(2, {1, 0, 0, 0});
  StateVector sv_0100(2, {0, 1, 0, 0});
  StateVector sv_0010(2, {0, 0, 1, 0});
  StateVector sv_0001(2, {0, 0, 0, 1});

  sv = sv_1000;
  sv.apply(BasicGates::SWAP(), 0, nullptr, 2, targets);
  CHECK_UNARY(sv == sv_1000);

  sv = sv_0100;
  sv.apply(BasicGates::SWAP(), 0, nullptr, 2, targets);
  CHECK_UNARY(sv == sv_0010);

  sv = sv_0010;
  sv.apply(BasicGates::SWAP(), 0, nullptr, 2, targets);
  CHECK_UNARY(sv == sv_0100);

  sv = sv_0001;
  sv.apply(BasicGates::SWAP(), 0, nullptr, 2, targets);
  CHECK_UNARY(sv == sv_0001);
}

TEST_CASE("Simple gates: CX") {
  using namespace SvSim;
  using namespace std::complex_literals;
  StateVector sv;
  StateVector sv_1000(2, {1, 0, 0, 0});
  StateVector sv_0100(2, {0, 1, 0, 0});
  StateVector sv_0010(2, {0, 0, 1, 0});
  StateVector sv_0001(2, {0, 0, 0, 1});

  SvSim::Qubit targets[] = {0};
  SvSim::Qubit controls[] = {1};

  // CX
  sv = sv_1000;
  sv.apply(BasicGates::X(), 1, controls, 1, targets);
  CHECK_UNARY(sv == sv_1000);

  sv = sv_0100;
  sv.apply(BasicGates::X(), 1, controls, 1, targets);
  CHECK_UNARY(sv == sv_0100);

  sv = sv_0010;
  sv.apply(BasicGates::X(), 1, controls, 1, targets);
  CHECK_UNARY(sv == sv_0001);

  sv = sv_0001;
  sv.apply(BasicGates::X(), 1, controls, 1, targets);
  CHECK_UNARY(sv == sv_0010);
}

TEST_CASE("Simple gates: CY") {
  using namespace SvSim;
  using namespace std::complex_literals;
  StateVector sv;
  StateVector sv_1000(2, {1, 0, 0, 0});
  StateVector sv_0100(2, {0, 1, 0, 0});
  StateVector sv_0010(2, {0, 0, 1, 0});
  StateVector sv_0001(2, {0, 0, 0, 1});
  StateVector sva, svb;

  // CY
  auto cy_compute = [&](auto& sva, auto& svb, SvSim::Qubit control, SvSim::Qubit target) {
    SvSim::Qubit controls[] = {control};
    SvSim::Qubit targets[] = {target};
    sva.apply(BasicGates::Y(), 1, controls, 1, targets);
    svb.apply(BasicGates::SDG(), 1, targets)
        .apply(BasicGates::X(), 1, controls, 1, targets)
        .apply(BasicGates::S(), 1, targets);
  };
  sva = svb = sv_1000;
  cy_compute(sva, svb, 0, 1);
  CHECK_UNARY(sva == svb);
  sva = svb = sv_0100;
  cy_compute(sva, svb, 0, 1);
  CHECK_UNARY(sva == svb);
  sva = svb = sv_0010;
  cy_compute(sva, svb, 0, 1);
  CHECK_UNARY(sva == svb);
  sva = svb = sv_0001;
  cy_compute(sva, svb, 0, 1);
  CHECK_UNARY(sva == svb);
}

TEST_CASE("Simple gates: CZ") {
  using namespace SvSim;
  using namespace std::complex_literals;
  StateVector sv;
  StateVector sv_1000(2, {1, 0, 0, 0});
  StateVector sv_0100(2, {0, 1, 0, 0});
  StateVector sv_0010(2, {0, 0, 1, 0});
  StateVector sv_0001(2, {0, 0, 0, 1});
  StateVector sva, svb;

  // CZ
  auto cz_compute = [&](auto& sva, auto& svb, SvSim::Qubit control, SvSim::Qubit target) {
    SvSim::Qubit controls[] = {control};
    SvSim::Qubit targets[] = {target};
    sva.apply(BasicGates::Z(), 1, controls, 1, targets);
    svb.apply(BasicGates::H(), 1, targets)
        .apply(BasicGates::X(), 1, controls, 1, targets)
        .apply(BasicGates::H(), 1, targets);
  };
  sva = svb = sv_1000;
  cz_compute(sva, svb, 0, 1);
  CHECK_UNARY(sva == svb);

  sva = svb = sv_0100;
  cz_compute(sva, svb, 0, 1);
  CHECK_UNARY(sva == svb);

  sva = svb = sv_0010;
  cz_compute(sva, svb, 0, 1);
  CHECK_UNARY(sva == svb);

  sva = svb = sv_0001;
  cz_compute(sva, svb, 0, 1);
  CHECK_UNARY(sva == svb);
}

TEST_CASE("Simple gates: CH") {
  using namespace SvSim;
  using namespace std::complex_literals;
  StateVector sv;
  StateVector sv_1000(2, {1, 0, 0, 0});
  StateVector sv_0100(2, {0, 1, 0, 0});
  StateVector sv_0010(2, {0, 0, 1, 0});
  StateVector sv_0001(2, {0, 0, 0, 1});
  StateVector sv_00hh(2, {0, 0, Constants::inv_sqrt_2, Constants::inv_sqrt_2});
  StateVector sv_00hnh(2, {0, 0, Constants::inv_sqrt_2, -Constants::inv_sqrt_2});
  StateVector sva;

  SvSim::Qubit controls[] = {1};
  SvSim::Qubit targets[] = {0};

  sva = sv_1000;
  sva.apply(BasicGates::H(), 1, controls, 1, targets);
  CHECK_UNARY(sva == sv_1000);

  sva = sv_0100;
  sva.apply(BasicGates::H(), 1, controls, 1, targets);
  CHECK_UNARY(sva == sv_0100);

  sva = sv_0010;
  sva.apply(BasicGates::H(), 1, controls, 1, targets);
  CHECK_UNARY(sva == sv_00hh);

  sva = sv_0001;
  sva.apply(BasicGates::H(), 1, controls, 1, targets);
  CHECK_UNARY(sva == sv_00hnh);
}

TEST_CASE("Simple gates") {
  using namespace SvSim;
  StateVector sv;

  sv.allocate(0);
  SvSim::Qubit targets[] = {0};

  CHECK_UNARY(sv.has_qubit(0));
  CHECK_UNARY(approx_equal(sv.probability(SvSim::Qubit{0}, 0), 1));
  CHECK_UNARY(approx_equal(sv.probability(SvSim::Qubit{0}, 1), 0));
  sv.apply(BasicGates::X(), 0, nullptr, 1, targets);
  CHECK_UNARY(approx_equal(sv.probability(SvSim::Qubit{0}, 0), 0));
  CHECK_UNARY(approx_equal(sv.probability(SvSim::Qubit{0}, 1), 1));
  sv.apply(BasicGates::X(), 0, nullptr, 1, targets);
  CHECK_UNARY(approx_equal(sv.probability(SvSim::Qubit{0}, 0), 1));
  CHECK_UNARY(approx_equal(sv.probability(SvSim::Qubit{0}, 1), 0));

  sv.apply(BasicGates::Y(), 0, nullptr, 1, targets);
  CHECK_UNARY(approx_equal(sv.probability(SvSim::Qubit{0}, 0), 0));
  CHECK_UNARY(approx_equal(sv.probability(SvSim::Qubit{0}, 1), 1));
  sv.apply(BasicGates::Y(), 0, nullptr, 1, targets);
  CHECK_UNARY(approx_equal(sv.probability(SvSim::Qubit{0}, 0), 1));
  CHECK_UNARY(approx_equal(sv.probability(SvSim::Qubit{0}, 1), 0));

  sv.apply(BasicGates::Z(), 0, nullptr, 1, targets);
  CHECK_UNARY(approx_equal(sv.probability(SvSim::Qubit{0}, 0), 1));
  CHECK_UNARY(approx_equal(sv.probability(SvSim::Qubit{0}, 1), 0));
  sv.apply(BasicGates::Z(), 0, nullptr, 1, targets);
  CHECK_UNARY(approx_equal(sv.probability(SvSim::Qubit{0}, 0), 1));
  CHECK_UNARY(approx_equal(sv.probability(SvSim::Qubit{0}, 1), 0));
}

TEST_CASE("Simple gates") {
  using namespace SvSim;
  StateVector sv;

  sv.allocate(0);
  SvSim::Qubit targets[] = {0};

  CHECK_UNARY(sv.has_qubit(0));
  CHECK_UNARY(approx_equal(sv.probability(SvSim::Qubit{0}, 0), 1));
  CHECK_UNARY(approx_equal(sv.probability(SvSim::Qubit{0}, 1), 0));
  sv.apply(BasicGates::X(), 0, nullptr, 1, targets);
  sv.apply(BasicGates::Y(), 0, nullptr, 1, targets);
  sv.apply(BasicGates::Z(), 0, nullptr, 1, targets);
  CHECK_UNARY(approx_equal(sv.probability(SvSim::Qubit{0}, 0), 1));
  CHECK_UNARY(approx_equal(sv.probability(SvSim::Qubit{0}, 1), 0));
}

TEST_CASE("Simple gates: H") {
  using namespace SvSim;
  StateVector sv;

  sv.allocate(0);
  SvSim::Qubit targets[] = {0};

  // sv.dump_state(std::cout);
  std::cerr << sv.probability(SvSim::Qubit{0}, 0) << "\n";
  CHECK_UNARY(sv.has_qubit(0));
  CHECK_UNARY(approx_equal(sv.probability(SvSim::Qubit{0}, 0), 1));
  CHECK_UNARY(approx_equal(sv.probability(SvSim::Qubit{0}, 1), 0));
  sv.apply(BasicGates::H(), 0, nullptr, 1, targets);
  // sv.dump_state(std::cout);
  std::cerr << sv.probability(SvSim::Qubit{0}, 0) << "\n";
  CHECK_UNARY(approx_equal(sv.probability(SvSim::Qubit{0}, 0), 0.5));
  CHECK_UNARY(approx_equal(sv.probability(SvSim::Qubit{0}, 1), 0.5));
  sv.apply(BasicGates::H(), 0, nullptr, 1, targets);
  // sv.dump_state(std::cout);
  std::cerr << sv.probability(SvSim::Qubit{0}, 0) << "\n";
  CHECK_UNARY(approx_equal(sv.probability(SvSim::Qubit{0}, 0), 1));
  CHECK_UNARY(approx_equal(sv.probability(SvSim::Qubit{0}, 1), 0));
}
#endif
#endif  // STATE_VECTOR_HPP_
