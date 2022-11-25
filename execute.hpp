#ifndef EXECUTE_HPP_
#define EXECUTE_HPP_

#include "state_vector.hpp"
#include <chrono>
#include <omp.h>
#include <algorithm>
#include <vector>
#include <numeric>
#ifdef SVMPI
  #include "mpi_redistributer.hpp"
  using namespace mpi_redistributer;
  #include "svsim-mpi.hpp"
#endif

namespace SvSim {

static long long gather_timer = 0;
std::vector<long long> pure_gather_timers(32*128,0);
std::vector<long long> pure_scatter_timers(32*128,0);
std::vector<long long> pure_compute_timers(32*128,0); 
static long long mpi_communication_timer = 0;
static long long mpi_compute_timer = 0;
static long long pure_scatter_timer = 0;
static long long pure_execute_timer = 0;
static long long pure_creation_timer = 0;
static long long pure_func_overhead = 0;
static int opt_slots = 0;


long long obtain_gater_timer(){
  return gather_timer;
}

void set_opt_slots(int opt){
  opt_slots = opt;
}

void execute_on(StateVector& state_vector,
                        std::vector<GateBuffer>& buffer_gates) {
  
  //AggregateGate_0 ag_0;
  //state_vector.apply(ag_0.union_gate_.H_,0,nullptr,1,targets);
  //state_vector.apply(ag_0.union_gate_.X_,0,nullptr,1,targets);
  //state_vector.apply(ag_0.union_gate_.Z_,0,nullptr,1,targets);
  //state_vector.apply_part(BasicGates::H(),0,nullptr,1,targets);
  //state_vector.apply_part(BasicGates::X(),0,nullptr,1,targets);
  //state_vector.apply_part(BasicGates::Z(),0,nullptr,1,targets);
  for (auto& bg: buffer_gates){
     AggregateGateFactory& ag_factory = bg.get_ag();
    if (ag_factory.get_type() == AGType::NOPARAM){
          AggregateGate_0& ag = ag_factory.get_ag0();
          switch (ag.gate_type_) {
            //case GateType::ALLOC: state_vector.allocate(std::vector<Qubit>(target_qubits, target_qubits + num_targets)); break;
            //case GateType::FREE: state_vector.project(std::vector<Qubit>(target_qubits, target_qubits + num_targets)); break;
            //case GateType::I: gatebuffer = new GateBuffer(state_vector, ag.union_gate_.I_, num_controls, control_qubits, num_targets, target_qubits);break;
            case GateType::X: state_vector.apply_part(ag.union_gate_.X_,bg.get_num_targets(),&(bg.get_target_slots()[0]),bg.get_control_offset(),bg.get_loop_info(), bg.get_count(),bg.get_skip(),bg.get_hi()); break;
            case GateType::Y: state_vector.apply_part(ag.union_gate_.Y_,bg.get_num_targets(),&(bg.get_target_slots()[0]),bg.get_control_offset(),bg.get_loop_info(), bg.get_count(),bg.get_skip(),bg.get_hi()); break;
            case GateType::Z: state_vector.apply_part(ag.union_gate_.Z_,bg.get_num_targets(),&(bg.get_target_slots()[0]),bg.get_control_offset(),bg.get_loop_info(), bg.get_count(),bg.get_skip(),bg.get_hi()); break;
            case GateType::H: state_vector.apply_part(ag.union_gate_.H_,bg.get_num_targets(),&(bg.get_target_slots()[0]),bg.get_control_offset(),bg.get_loop_info(), bg.get_count(),bg.get_skip(),bg.get_hi()); break;
            case GateType::S: state_vector.apply_part(ag.union_gate_.S_,bg.get_num_targets(),&(bg.get_target_slots()[0]),bg.get_control_offset(),bg.get_loop_info(), bg.get_count(),bg.get_skip(),bg.get_hi()); break;
            case GateType::T: state_vector.apply_part(ag.union_gate_.T_,bg.get_num_targets(),&(bg.get_target_slots()[0]),bg.get_control_offset(),bg.get_loop_info(), bg.get_count(),bg.get_skip(),bg.get_hi()); break;
            case GateType::SWAP: state_vector.apply_part(ag.union_gate_.SWAP_,bg.get_num_targets(),&(bg.get_target_slots()[0]),bg.get_control_offset(),bg.get_loop_info(), bg.get_count(),bg.get_skip(),bg.get_hi()); break;
            case GateType::TDG: state_vector.apply_part(ag.union_gate_.TDG_,bg.get_num_targets(),&(bg.get_target_slots()[0]),bg.get_control_offset(),bg.get_loop_info(), bg.get_count(),bg.get_skip(),bg.get_hi()); break;
            case GateType::SDG: state_vector.apply_part(ag.union_gate_.SDG_,bg.get_num_targets(),&(bg.get_target_slots()[0]),bg.get_control_offset(),bg.get_loop_info(), bg.get_count(),bg.get_skip(),bg.get_hi()); break;
            case GateType::invalid:
              UNREACHABLE();
              break;
            default:
            NOT_IMPLEMENTED();
          }
      }
      else if (ag_factory.get_type() == AGType::ONEPARAM){
          AggregateGate_1 & ag = ag_factory.get_ag1();
          switch (ag.gate_type_) {
            //case GateType::ALLOC: state_vector.allocate(std::vector<Qubit>(target_qubits, target_qubits + num_targets)); break;
            //case GateType::FREE: state_vector.project(std::vector<Qubit>(target_qubits, target_qubits + num_targets)); break;
            case GateType::RX: state_vector.apply_part(ag.union_gate_.RX_,bg.get_num_targets(),&(bg.get_target_slots()[0]),bg.get_control_offset(),bg.get_loop_info(), bg.get_count(),bg.get_skip(),bg.get_hi()); break;
            case GateType::RY: state_vector.apply_part(ag.union_gate_.RY_,bg.get_num_targets(),&(bg.get_target_slots()[0]),bg.get_control_offset(),bg.get_loop_info(), bg.get_count(),bg.get_skip(),bg.get_hi()); break;
            case GateType::RZ: state_vector.apply_part(ag.union_gate_.RZ_,bg.get_num_targets(),&(bg.get_target_slots()[0]),bg.get_control_offset(),bg.get_loop_info(), bg.get_count(),bg.get_skip(),bg.get_hi()); break;
            case GateType::RI: state_vector.apply_part(ag.union_gate_.RI_,bg.get_num_targets(),&(bg.get_target_slots()[0]),bg.get_control_offset(),bg.get_loop_info(), bg.get_count(),bg.get_skip(),bg.get_hi()); break;
            case GateType::R1: state_vector.apply_part(ag.union_gate_.R1_,bg.get_num_targets(),&(bg.get_target_slots()[0]),bg.get_control_offset(),bg.get_loop_info(), bg.get_count(),bg.get_skip(),bg.get_hi()); break;
            case GateType::U1: state_vector.apply_part(ag.union_gate_.U1_,bg.get_num_targets(),&(bg.get_target_slots()[0]),bg.get_control_offset(),bg.get_loop_info(), bg.get_count(),bg.get_skip(),bg.get_hi()); break;
            case GateType::RXX: state_vector.apply_part(ag.union_gate_.RXX_,bg.get_num_targets(),&(bg.get_target_slots()[0]),bg.get_control_offset(),bg.get_loop_info(), bg.get_count(),bg.get_skip(),bg.get_hi()); break;
            case GateType::RZZ: state_vector.apply_part(ag.union_gate_.RZZ_,bg.get_num_targets(),&(bg.get_target_slots()[0]),bg.get_control_offset(),bg.get_loop_info(), bg.get_count(),bg.get_skip(),bg.get_hi()); break;
            case GateType::RYY: state_vector.apply_part(ag.union_gate_.RYY_,bg.get_num_targets(),&(bg.get_target_slots()[0]),bg.get_control_offset(),bg.get_loop_info(), bg.get_count(),bg.get_skip(),bg.get_hi()); break;
            case GateType::invalid:
              UNREACHABLE();
              break;
            default:
              NOT_IMPLEMENTED();
          }
      }
    else if (ag_factory.get_type() == AGType::TWOPARAM){
          AggregateGate_2 & ag = ag_factory.get_ag2();
          switch (ag.gate_type_) {
            //case GateType::ALLOC: state_vector.allocate(std::vector<Qubit>(target_qubits, target_qubits + num_targets)); break;
            //case GateType::FREE: state_vector.project(std::vector<Qubit>(target_qubits, target_qubits + num_targets)); break;
            case GateType::U2: state_vector.apply_part(ag.union_gate_.U2_,bg.get_num_targets(),&(bg.get_target_slots()[0]),bg.get_control_offset(),bg.get_loop_info(), bg.get_count(),bg.get_skip(),bg.get_hi()); break;
            case GateType::invalid:
              UNREACHABLE();
              break;
            default:
              NOT_IMPLEMENTED();
          }
      }
    else if (ag_factory.get_type() == AGType::THREEPARAM){
          AggregateGate_3 & ag = ag_factory.get_ag3();
          switch (ag.gate_type_) {
            //case GateType::ALLOC: state_vector.allocate(std::vector<Qubit>(target_qubits, target_qubits + num_targets)); break;
            //case GateType::FREE: state_vector.project(std::vector<Qubit>(target_qubits, target_qubits + num_targets)); break;
            case GateType::U3: state_vector.apply_part(ag.union_gate_.U3_, bg.get_num_targets(),&(bg.get_target_slots()[0]),bg.get_control_offset(),bg.get_loop_info(), bg.get_count(),bg.get_skip(),bg.get_hi()); break;
            case GateType::invalid:
              UNREACHABLE();
              break;
            default:
              NOT_IMPLEMENTED();
          }
      }
    else if (ag_factory.get_type() == AGType::C1){
          AggregateGate_C1 & ag = ag_factory.get_agC1();
          switch (ag.gate_type_) {
            //case GateType::ALLOC: state_vector.allocate(std::vector<Qubit>(target_qubits, target_qubits + num_targets)); break;
            //case GateType::FREE: state_vector.project(std::vector<Qubit>(target_qubits, target_qubits + num_targets)); break;
            case GateType::C1: state_vector.apply_part(ag.union_gate_.C1_, bg.get_num_targets(),&(bg.get_target_slots()[0]),bg.get_control_offset(),bg.get_loop_info(), bg.get_count(),bg.get_skip(),bg.get_hi()); break;
            case GateType::invalid:
              UNREACHABLE();
              break;
            default:
              NOT_IMPLEMENTED();
          }
      }
    else if (ag_factory.get_type() == AGType::C2){
          AggregateGate_C2 & ag = ag_factory.get_agC2();
          switch (ag.gate_type_) {
            //case GateType::ALLOC: state_vector.allocate(std::vector<Qubit>(target_qubits, target_qubits + num_targets)); break;
            //case GateType::FREE: state_vector.project(std::vector<Qubit>(target_qubits, target_qubits + num_targets)); break;
            case GateType::C2: state_vector.apply_part(ag.union_gate_.C2_, bg.get_num_targets(),&(bg.get_target_slots()[0]),bg.get_control_offset(),bg.get_loop_info(), bg.get_count(),bg.get_skip(),bg.get_hi()); break;
            case GateType::invalid:
              UNREACHABLE();
              break;
            default:
              NOT_IMPLEMENTED();
          }
      }
    else if (ag_factory.get_type() == AGType::FRAC){
          AggregateGate_Frac & ag = ag_factory.get_agFrac();
          switch (ag.gate_type_) {
            //case GateType::ALLOC: state_vector.allocate(std::vector<Qubit>(target_qubits, target_qubits + num_targets)); break;
            //case GateType::FREE: state_vector.project(std::vector<Qubit>(target_qubits, target_qubits + num_targets)); break;
            case GateType::RXFrac: state_vector.apply_part(ag.union_gate_.RXFrac_, bg.get_num_targets(),&(bg.get_target_slots()[0]),bg.get_control_offset(),bg.get_loop_info(), bg.get_count(),bg.get_skip(),bg.get_hi()); break;
            case GateType::RYFrac: state_vector.apply_part(ag.union_gate_.RYFrac_, bg.get_num_targets(),&(bg.get_target_slots()[0]),bg.get_control_offset(),bg.get_loop_info(), bg.get_count(),bg.get_skip(),bg.get_hi()); break;
            case GateType::RZFrac: state_vector.apply_part(ag.union_gate_.RZFrac_, bg.get_num_targets(),&(bg.get_target_slots()[0]),bg.get_control_offset(),bg.get_loop_info(), bg.get_count(),bg.get_skip(),bg.get_hi()); break;
            case GateType::RIFrac: state_vector.apply_part(ag.union_gate_.RIFrac_, bg.get_num_targets(),&(bg.get_target_slots()[0]),bg.get_control_offset(),bg.get_loop_info(), bg.get_count(),bg.get_skip(),bg.get_hi()); break;
            case GateType::R1Frac: state_vector.apply_part(ag.union_gate_.R1Frac_, bg.get_num_targets(),&(bg.get_target_slots()[0]),bg.get_control_offset(),bg.get_loop_info(), bg.get_count(),bg.get_skip(),bg.get_hi()); break;
            case GateType::invalid:
              UNREACHABLE();
              break;
            default:
              NOT_IMPLEMENTED();
          }
      }
      else{
        NOT_IMPLEMENTED();
      }       
  
  }
}


 void gather_and_execute_on(StateVector& state_vector,
                                   std::vector<BoundGate>& bound_gates) {
  auto start = std::chrono::high_resolution_clock::now();
  std::set<Qubit> needed_qubits;
  for (const auto& bg : bound_gates) {
    needed_qubits.insert(bg.targets().begin(),
                         bg.targets().begin() + bg.num_targets());
    needed_qubits.insert(bg.controls().begin(),
                         bg.controls().begin() + bg.num_controls());
  }
    std::vector<Qubit> qubit_vec{needed_qubits.begin(), needed_qubits.end()};
  // adding fast slots
  #ifndef SVMPI
  if (opt_slots == 1){
    
    if (std::find(qubit_vec.begin(),qubit_vec.end(),state_vector.slot_to_qubit(0)) == qubit_vec.end())
        qubit_vec.push_back(state_vector.slot_to_qubit(0));
    if (std::find(qubit_vec.begin(),qubit_vec.end(),state_vector.slot_to_qubit(1)) == qubit_vec.end())
        qubit_vec.push_back(state_vector.slot_to_qubit(1));
    if (std::find(qubit_vec.begin(),qubit_vec.end(),state_vector.slot_to_qubit(2)) == qubit_vec.end())
        qubit_vec.push_back(state_vector.slot_to_qubit(2));
    if (std::find(qubit_vec.begin(),qubit_vec.end(),state_vector.slot_to_qubit(3)) == qubit_vec.end())
        qubit_vec.push_back(state_vector.slot_to_qubit(3));
    /*  for (int i = 0; i < state_vector.num_slots(); i++){
        if (std::find(qubit_vec.begin(),qubit_vec.end(),i) == qubit_vec.end()
            && qubit_vec.size() < num_local_slots)
            qubit_vec.push_back(i);
      } */
  }
  #endif
  std::vector<Qubit> ordered_qubits =
      state_vector.qubits_ordered_by_slots(qubit_vec);
  std::cout << "size of the ordered qubit " << ordered_qubits.size() << std::endl;
  std::cout << "contains qubit ";
  for (auto q : ordered_qubits)
  {
    std::cout << q << " ";
  }
  std::cout << std::endl;
  //StateVector inner_sv{ordered_qubits,{},0};

  std::vector<Qubit> outer_qubits = state_vector.remaining_qubits(
      std::set<Qubit>{ordered_qubits.begin(), ordered_qubits.end()});

  auto start_creation = std::chrono::high_resolution_clock::now();
  //std::cout << "num of threads " << n_threads << std::endl;
  StateVector * inner_sv = new StateVector[n_threads];
  #pragma omp parallel for num_threads(n_threads)
  for (int i = 0; i < n_threads; i++){
    inner_sv[i] = StateVector(ordered_qubits,{},2);
  }
  //inner_sv[0].get_qubit_slot_map();
  //StateVector inner_sv{ordered_qubits,{},0};
  auto creation_elapsed = std::chrono::high_resolution_clock::now() - start_creation;
  pure_creation_timer += std::chrono::duration_cast<std::chrono::microseconds>(creation_elapsed).count();

  auto pure_func_last = std::chrono::high_resolution_clock::now();
  std::vector<Slot> this_slot_gather = state_vector.get_slot_gather(ordered_qubits,inner_sv[0]);
  std::vector<Slot> this_slot_scatter = state_vector.get_slot_scatter(ordered_qubits,inner_sv[0]);
  std::sort(this_slot_gather.begin(), this_slot_gather.end());
    // unique slots
    //std::cerr << __PRETTY_FUNCTION__ << " " << __LINE__ << "\n";
  assert((std::set<Slot>{this_slot_gather.begin(), this_slot_gather.end()}.size()) == this_slot_gather.size());
  std::vector<LoopInfo> loop_info_gather = state_vector.loop_info_for_slots(this_slot_gather);
  //std::cout << "loop info size " <<loop_info_gather.size() << std::endl;
  std::sort(this_slot_scatter.begin(), this_slot_scatter.end());
    // unique slots
    //std::cerr << __PRETTY_FUNCTION__ << " " << __LINE__ << "\n";
  assert((std::set<Slot>{this_slot_scatter.begin(), this_slot_scatter.end()}.size()) == this_slot_scatter.size());
  std::vector<LoopInfo> loop_info_scatter = state_vector.loop_info_for_slots(this_slot_scatter);
  std::vector<size_t> count_gather(loop_info_gather.size());
  std::vector<size_t> skip_gather(loop_info_gather.size());
  std::vector<size_t> hi_gather(loop_info_gather.size());
  for (size_t i = 0; i < loop_info_gather.size(); i++) {
      std::tie(count_gather[i], skip_gather[i]) = loop_info_gather[i];
      hi_gather[i] = count_gather[i] * skip_gather[i];
  }
  std::vector<size_t> count_scatter(loop_info_scatter.size());
  std::vector<size_t> skip_scatter(loop_info_scatter.size());
  std::vector<size_t> hi_scatter(loop_info_scatter.size());
  for (size_t i = 0; i < loop_info_scatter.size(); i++) {
      std::tie(count_scatter[i], skip_scatter[i]) = loop_info_scatter[i];
      hi_scatter[i] = count_scatter[i] * skip_scatter[i];
  }
  /*----need to parse boundgate here */
  std::vector<GateBuffer> gates_buffer;
  for (auto& bg : bound_gates) {
    gates_buffer.push_back(bg.apply_to_buffer(inner_sv[0]));
  }
  std::cout << "before entering the qubit loop" << std::endl;
  state_vector.qubit_loop(
      [&](size_t loop_itr, size_t /*trip_count*/,int tid) {
        //auto start_func_overhead = std::chrono::high_resolution_clock::now();
        //auto func_elapsed = start_func_overhead - pure_func_last;
        //pure_func_overhead += std::chrono::duration_cast<std::chrono::microseconds>(func_elapsed).count();
        
        //Type a = inner_sv.prefetch_state();
        //assert(std::exp(std::conj(a)) == std::conj(std::exp(a))); 
        auto start_gather = std::chrono::high_resolution_clock::now();
        //state_vector.gather_qubits(loop_info_gather, count_gather,skip_gather,hi_gather,loop_itr, inner_sv[tid]);
        auto end_gather = std::chrono::high_resolution_clock::now();
        //pure_gather_timer += std::chrono::duration_cast<std::chrono::microseconds>(gather_elapsed).count();
        //std::cout << "gather takes " << std::chrono::duration_cast<std::chrono::microseconds>(gather_elapsed).count() << std::endl;
        //auto start_execute = std::chrono::high_resolution_clock::now();
        //omp_set_nested(true);
        //execute_on(inner_sv[tid], gates_buffer);
        //execute_on(inner_sv[tid], num_targets,&(target_slots[0]),control_offsets,loopinfo,count,skip,hi);
        //omp_set_nested(false);
        auto end_compute = std::chrono::high_resolution_clock::now();
        //pure_execute_timer += std::chrono::duration_cast<std::chrono::microseconds>(execute_elapsed).count();
        //std::cout << "execute on takes " << std::chrono::duration_cast<std::chrono::microseconds>(execute_elapsed).count() << std::endl;
        //auto start_scatter = std::chrono::high_resolution_clock::now();
        //state_vector.scatter_qubits(loop_info_scatter, count_scatter,skip_scatter,hi_scatter,loop_itr, inner_sv[tid]);
        auto end_scatter = std::chrono::high_resolution_clock::now();
        /*pure_gather_timers[tid*32] += std::chrono::duration_cast<std::chrono::microseconds>(end_gather-start_gather).count();
        pure_compute_timers[tid*32] += std::chrono::duration_cast<std::chrono::microseconds>(end_compute-end_gather).count();
        pure_scatter_timers[tid*32] += std::chrono::duration_cast<std::chrono::microseconds>(end_scatter-end_compute).count();*/
        //pure_scatter_timer += std::chrono::duration_cast<std::chrono::microseconds>(scatter_elapsed).count();
        //pure_func_last = std::chrono::high_resolution_clock::now();
        //std::cout << "scatter takes " << std::chrono::duration_cast<std::chrono::microseconds>(scatter_elapsed).count() << std::endl;
        //pure_gather_timer += loop_itr*loop_itr;
      },
      outer_qubits);
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  gather_timer += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  //std::cout << "pure gather time longest " <<  *std::max_element(pure_gather_timers.begin(),pure_gather_timers.end()) << std::endl;
  //std::cout << "pure compoute time longest " <<  *std::max_element(pure_compute_timers.begin(),pure_compute_timers.end()) << std::endl;
  //std::cout << "pure scatter time longest " <<  *std::max_element(pure_scatter_timers.begin(),pure_scatter_timers.end()) << std::endl;
  /*std::cout << "pure gather time average " <<  std::accumulate(pure_gather_timers.begin(),pure_gather_timers.end(),0)/128 << std::endl;
  std::cout << "pure compute time average " <<  std::accumulate(pure_compute_timers.begin(),pure_compute_timers.end(),0)/128 << std::endl;
  std::cout << "pure scatter time average " <<  std::accumulate(pure_scatter_timers.begin(),pure_scatter_timers.end(),0)/128 << std::endl;*/
  //std::cout << "pure scatter time " <<  pure_scatter_timer << std::endl;
  //std::cout << "pure execute time " <<  pure_execute_timer << std::endl;
  //std::cout << "pure func overhead time " <<  pure_func_overhead << std::endl;
  //std::cout << "loop N=1 total time " << slot_loop_timer << std::endl;
  //std::cout << "loop N=1 func total time " << inside_loop_func << std::endl;
   //std::cout << "gather time " <<  gather_timer << std::endl;
   //std::cout << "data movement " << mem_move_counter << std::endl;
   mem_move_counter = 0;
  //std::cout << "pure creation time " <<  pure_creation_timer << std::endl;
  /*for (int i = 0; i < n_threads; i++){
    inner_sv[i].free_state();
  }
  delete inner_sv;
  inner_sv = NULL;*/
}

void gather_and_execute_on_multi(StateVector& state_vector,
                                   std::vector<BoundGate>& bound_gates, std::vector<Qubit> top_level_qubits, size_t local_limit) {
  auto start = std::chrono::high_resolution_clock::now();
  std::set<Qubit> needed_qubits;
  for (const auto& bg : bound_gates) {
    needed_qubits.insert(bg.targets().begin(),
                         bg.targets().begin() + bg.num_targets());
    needed_qubits.insert(bg.controls().begin(),
                         bg.controls().begin() + bg.num_controls());
  }
    std::vector<Qubit> qubit_vec{needed_qubits.begin(), needed_qubits.end()};
    std::cout << "size of qubit_vec is " << qubit_vec.size() << std::endl;
  // adding fast slots
  #ifndef SVMPI
  if (opt_slots == 1){
    
    if (std::find(qubit_vec.begin(),qubit_vec.end(),state_vector.slot_to_qubit(0)) == qubit_vec.end())
        qubit_vec.push_back(state_vector.slot_to_qubit(0));
    if (std::find(qubit_vec.begin(),qubit_vec.end(),state_vector.slot_to_qubit(1)) == qubit_vec.end())
        qubit_vec.push_back(state_vector.slot_to_qubit(1));
    if (std::find(qubit_vec.begin(),qubit_vec.end(),state_vector.slot_to_qubit(2)) == qubit_vec.end())
        qubit_vec.push_back(state_vector.slot_to_qubit(2));
    if (std::find(qubit_vec.begin(),qubit_vec.end(),state_vector.slot_to_qubit(3)) == qubit_vec.end())
        qubit_vec.push_back(state_vector.slot_to_qubit(3));
    /*  for (int i = 0; i < state_vector.num_slots(); i++){
        if (std::find(qubit_vec.begin(),qubit_vec.end(),i) == qubit_vec.end()
            && qubit_vec.size() < num_local_slots)
            qubit_vec.push_back(i);
      } */
  }
  #else
    for (auto q:top_level_qubits){
        if (std::find(qubit_vec.begin(),qubit_vec.end(),q) == qubit_vec.end()
            && qubit_vec.size() < local_limit && std::find(top_level_qubits.begin(),top_level_qubits.end(),q) != top_level_qubits.end())
            qubit_vec.push_back(q);
    }
  #endif
  std::vector<Qubit> ordered_qubits =
      state_vector.qubits_ordered_by_slots(qubit_vec);
  std::cout << "size of the ordered qubit in g_e" << ordered_qubits.size() << std::endl;
  std::cout << "contains qubit ";
  for (auto q : ordered_qubits)
  {
    std::cout << q << " ";
  }
  std::cout << std::endl;
  //StateVector inner_sv{ordered_qubits,{},0};

  std::vector<Qubit> outer_qubits = state_vector.remaining_qubits(
      std::set<Qubit>{ordered_qubits.begin(), ordered_qubits.end()});

  auto start_creation = std::chrono::high_resolution_clock::now();
  //std::cout << "num of threads " << n_threads << std::endl;
  StateVector * inner_sv = new StateVector[n_threads];
  #pragma omp parallel for num_threads(n_threads)
  for (int i = 0; i < n_threads; i++){
    inner_sv[i] = StateVector(ordered_qubits,{},2);
  }
  //inner_sv[0].get_qubit_slot_map();
  //StateVector inner_sv{ordered_qubits,{},0};
  auto creation_elapsed = std::chrono::high_resolution_clock::now() - start_creation;
  pure_creation_timer += std::chrono::duration_cast<std::chrono::microseconds>(creation_elapsed).count();

  auto pure_func_last = std::chrono::high_resolution_clock::now();
  std::vector<Slot> this_slot_gather = state_vector.get_slot_gather(ordered_qubits,inner_sv[0]);
  std::vector<Slot> this_slot_scatter = state_vector.get_slot_scatter(ordered_qubits,inner_sv[0]);
  std::sort(this_slot_gather.begin(), this_slot_gather.end());
    // unique slots
    //std::cerr << __PRETTY_FUNCTION__ << " " << __LINE__ << "\n";
  assert((std::set<Slot>{this_slot_gather.begin(), this_slot_gather.end()}.size()) == this_slot_gather.size());
  std::vector<LoopInfo> loop_info_gather = state_vector.loop_info_for_slots(this_slot_gather);
  //std::cout << "loop info size " <<loop_info_gather.size() << std::endl;
  std::sort(this_slot_scatter.begin(), this_slot_scatter.end());
    // unique slots
    //std::cerr << __PRETTY_FUNCTION__ << " " << __LINE__ << "\n";
  assert((std::set<Slot>{this_slot_scatter.begin(), this_slot_scatter.end()}.size()) == this_slot_scatter.size());
  std::vector<LoopInfo> loop_info_scatter = state_vector.loop_info_for_slots(this_slot_scatter);
  std::vector<size_t> count_gather(loop_info_gather.size());
  std::vector<size_t> skip_gather(loop_info_gather.size());
  std::vector<size_t> hi_gather(loop_info_gather.size());
  for (size_t i = 0; i < loop_info_gather.size(); i++) {
      std::tie(count_gather[i], skip_gather[i]) = loop_info_gather[i];
      hi_gather[i] = count_gather[i] * skip_gather[i];
  }
  std::vector<size_t> count_scatter(loop_info_scatter.size());
  std::vector<size_t> skip_scatter(loop_info_scatter.size());
  std::vector<size_t> hi_scatter(loop_info_scatter.size());
  for (size_t i = 0; i < loop_info_scatter.size(); i++) {
      std::tie(count_scatter[i], skip_scatter[i]) = loop_info_scatter[i];
      hi_scatter[i] = count_scatter[i] * skip_scatter[i];
  }
  /*----need to parse boundgate here */
  std::vector<GateBuffer> gates_buffer;
  for (auto& bg : bound_gates) {
    gates_buffer.push_back(bg.apply_to_buffer(inner_sv[0]));
  }
  std::cout << "before entering the qubit loop" << std::endl;
  state_vector.qubit_loop(
      [&](size_t loop_itr, size_t /*trip_count*/,int tid) {
        //auto start_func_overhead = std::chrono::high_resolution_clock::now();
        //auto func_elapsed = start_func_overhead - pure_func_last;
        //pure_func_overhead += std::chrono::duration_cast<std::chrono::microseconds>(func_elapsed).count();
        
        //Type a = inner_sv.prefetch_state();
        //assert(std::exp(std::conj(a)) == std::conj(std::exp(a))); 
        auto start_gather = std::chrono::high_resolution_clock::now();
        state_vector.gather_qubits(loop_info_gather, count_gather,skip_gather,hi_gather,loop_itr, inner_sv[tid]);
        auto end_gather = std::chrono::high_resolution_clock::now();
        //pure_gather_timer += std::chrono::duration_cast<std::chrono::microseconds>(gather_elapsed).count();
        //std::cout << "gather takes " << std::chrono::duration_cast<std::chrono::microseconds>(gather_elapsed).count() << std::endl;
        //auto start_execute = std::chrono::high_resolution_clock::now();
        //omp_set_nested(true);
        execute_on(inner_sv[tid], gates_buffer);
        //execute_on(inner_sv[tid], num_targets,&(target_slots[0]),control_offsets,loopinfo,count,skip,hi);
        //omp_set_nested(false);
        auto end_compute = std::chrono::high_resolution_clock::now();
        //pure_execute_timer += std::chrono::duration_cast<std::chrono::microseconds>(execute_elapsed).count();
        //std::cout << "execute on takes " << std::chrono::duration_cast<std::chrono::microseconds>(execute_elapsed).count() << std::endl;
        //auto start_scatter = std::chrono::high_resolution_clock::now();
        state_vector.scatter_qubits(loop_info_scatter, count_scatter,skip_scatter,hi_scatter,loop_itr, inner_sv[tid]);
        auto end_scatter = std::chrono::high_resolution_clock::now();
        /*pure_gather_timers[tid*32] += std::chrono::duration_cast<std::chrono::microseconds>(end_gather-start_gather).count();
        pure_compute_timers[tid*32] += std::chrono::duration_cast<std::chrono::microseconds>(end_compute-end_gather).count();
        pure_scatter_timers[tid*32] += std::chrono::duration_cast<std::chrono::microseconds>(end_scatter-end_compute).count();*/
        //pure_scatter_timer += std::chrono::duration_cast<std::chrono::microseconds>(scatter_elapsed).count();
        //pure_func_last = std::chrono::high_resolution_clock::now();
        //std::cout << "scatter takes " << std::chrono::duration_cast<std::chrono::microseconds>(scatter_elapsed).count() << std::endl;
        //pure_gather_timer += loop_itr*loop_itr;
      },
      outer_qubits);
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  gather_timer += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  //std::cout << "pure gather time longest " <<  *std::max_element(pure_gather_timers.begin(),pure_gather_timers.end()) << std::endl;
  //std::cout << "pure compoute time longest " <<  *std::max_element(pure_compute_timers.begin(),pure_compute_timers.end()) << std::endl;
  //std::cout << "pure scatter time longest " <<  *std::max_element(pure_scatter_timers.begin(),pure_scatter_timers.end()) << std::endl;
  /*std::cout << "pure gather time average " <<  std::accumulate(pure_gather_timers.begin(),pure_gather_timers.end(),0)/128 << std::endl;
  std::cout << "pure compute time average " <<  std::accumulate(pure_compute_timers.begin(),pure_compute_timers.end(),0)/128 << std::endl;
  std::cout << "pure scatter time average " <<  std::accumulate(pure_scatter_timers.begin(),pure_scatter_timers.end(),0)/128 << std::endl;*/
  //std::cout << "pure scatter time " <<  pure_scatter_timer << std::endl;
  //std::cout << "pure execute time " <<  pure_execute_timer << std::endl;
  //std::cout << "pure func overhead time " <<  pure_func_overhead << std::endl;
  //std::cout << "loop N=1 total time " << slot_loop_timer << std::endl;
  //std::cout << "loop N=1 func total time " << inside_loop_func << std::endl;
   //std::cout << "gather time " <<  gather_timer << std::endl;
   //std::cout << "data movement " << mem_move_counter << std::endl;
   mem_move_counter = 0;
  //std::cout << "pure creation time " <<  pure_creation_timer << std::endl;
  /*for (int i = 0; i < n_threads; i++){
    inner_sv[i].free_state();
  }
  delete inner_sv;
  inner_sv = NULL;*/
}


inline void gather_noncontrol_and_execute_on(
    StateVector& state_vector,  std::vector<BoundGate>& bound_gates) {
  std::set<Qubit> needed_qubits;
  std::set<Qubit> needed_control_qubits;
  for (const auto& bg : bound_gates) {
    needed_qubits.insert(bg.targets().begin(),
                         bg.targets().begin() + bg.num_targets());
    needed_control_qubits.insert(bg.controls().begin(),
                                 bg.controls().begin() + bg.num_controls());
  }
  std::set<Qubit> control_only_qubits;
  std::set_difference(
      needed_control_qubits.begin(), needed_control_qubits.end(),
      needed_qubits.begin(), needed_qubits.end(),
      std::inserter(control_only_qubits, control_only_qubits.end()));

  std::vector<Qubit> qubit_vec{needed_qubits.begin(), needed_qubits.end()};
  std::vector<Qubit> ordered_qubits =
      state_vector.qubits_ordered_by_slots(qubit_vec);
  StateVector * inner_sv = (StateVector *)malloc(sizeof(StateVector)*n_threads);
  for (int i = 0; i < n_threads; i++){
    inner_sv[i] = StateVector(ordered_qubits,control_only_qubits,0);
  }

  std::set<Qubit> used_qubits{ordered_qubits.begin(), ordered_qubits.end()};
  used_qubits.insert(control_only_qubits.begin(), control_only_qubits.end());
  std::vector<Qubit> outer_qubits = state_vector.remaining_qubits(
      std::set<Qubit>{used_qubits.begin(), used_qubits.end()});
std::vector<Slot> this_slot_gather = state_vector.get_slot_gather(ordered_qubits,inner_sv[0]);
  std::vector<Slot> this_slot_scatter = state_vector.get_slot_scatter(ordered_qubits,inner_sv[0]);
  std::sort(this_slot_gather.begin(), this_slot_gather.end());
    // unique slots
    //std::cerr << __PRETTY_FUNCTION__ << " " << __LINE__ << "\n";
  assert((std::set<Slot>{this_slot_gather.begin(), this_slot_gather.end()}.size()) == this_slot_gather.size());
  std::vector<LoopInfo> loop_info_gather = state_vector.loop_info_for_slots(this_slot_gather);
  std::sort(this_slot_scatter.begin(), this_slot_scatter.end());
    // unique slots
    //std::cerr << __PRETTY_FUNCTION__ << " " << __LINE__ << "\n";
  assert((std::set<Slot>{this_slot_scatter.begin(), this_slot_scatter.end()}.size()) == this_slot_scatter.size());
  std::vector<LoopInfo> loop_info_scatter = state_vector.loop_info_for_slots(this_slot_scatter);
  std::vector<size_t> count_gather(loop_info_gather.size());
  std::vector<size_t> skip_gather(loop_info_gather.size());
  std::vector<size_t> hi_gather(loop_info_gather.size());
  for (size_t i = 0; i < loop_info_gather.size(); i++) {
      std::tie(count_gather[i], skip_gather[i]) = loop_info_gather[i];
      hi_gather[i] = count_gather[i] * skip_gather[i];
  }
  std::vector<size_t> count_scatter(loop_info_scatter.size());
  std::vector<size_t> skip_scatter(loop_info_scatter.size());
  std::vector<size_t> hi_scatter(loop_info_scatter.size());
  for (size_t i = 0; i < loop_info_scatter.size(); i++) {
      std::tie(count_scatter[i], skip_scatter[i]) = loop_info_scatter[i];
      hi_scatter[i] = count_scatter[i] * skip_scatter[i];
  }
  std::vector<GateBuffer> gates_buffer;
  for (auto& bg : bound_gates) {
    gates_buffer.push_back(bg.apply_to_buffer(inner_sv[0]));
  }
  state_vector.qubit_loop(
      [&](size_t loop_itr, size_t /*trip_count*/, int tid) {
        //StateVector inner_sv{ordered_qubits, control_only_qubits,0};
        state_vector.gather_qubits(loop_info_gather, count_gather,skip_gather,hi_gather, loop_itr, inner_sv[tid]);
        execute_on(inner_sv[tid], gates_buffer);
        state_vector.scatter_qubits(loop_info_scatter, count_scatter, skip_scatter, hi_scatter, loop_itr, inner_sv[tid]);
      },
      outer_qubits);
}

#ifdef SVMPI
  void gather_and_execute_on_mpi(int num_total_qubits, int num_local_qubits,
                                   std::vector<std::vector<BoundGate>>& parts) {
    int comm_rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    double prep_time =0, comm_time = 0, compute_time = 0, end_to_end_time = 0;
    double max_prep_time =0, max_comm_time = 0, max_compute_time = 0, max_end_to_end_time = 0;
    double min_prep_time =0, min_comm_time = 0, min_compute_time = 0, min_end_to_end_time = 0;
    double avg_prep_time =0, avg_comm_time = 0, avg_compute_time = 0, avg_end_to_end_time = 0;
    prep_time = MPI_Wtime();
    end_to_end_time = MPI_Wtime();
    // take the first part and create the buf_in and buf_out state vector
    std::vector<BoundGate> first_part = parts[0];
    std::set<Qubit> needed_qubits;
    for (const auto& bg : first_part) {
      needed_qubits.insert(bg.targets().begin(),
                           bg.targets().begin() + bg.num_targets());
      needed_qubits.insert(bg.controls().begin(),
                           bg.controls().begin() + bg.num_controls());
    }
    
    std::vector<Qubit> qubit_vec{needed_qubits.begin(), needed_qubits.end()};
    // adding fast slots
    if (opt_slots == 1){
      for (int i = 0; i < num_total_qubits; i++){
        if (std::find(qubit_vec.begin(),qubit_vec.end(),i) == qubit_vec.end()
            && qubit_vec.size() < num_local_qubits)
            qubit_vec.push_back(i);
      }
    }
    std::vector<Qubit> ordered_qubits{qubit_vec.begin(),qubit_vec.end()};
    std::sort(ordered_qubits.begin(), ordered_qubits.end());
    if (comm_rank == 0)
    {
      std::cout << "size of the ordered qubit " << ordered_qubits.size() << std::endl;
      std::cout << "contains qubit ";
      for (auto q : ordered_qubits)
      {
        std::cout << q << " ";
      }
      std::cout << std::endl;
    }
    std::vector<Qubit> proc_qubits;
    for (Qubit q = 0; q < num_total_qubits; ++q) {
        if (std::find(ordered_qubits.begin(), ordered_qubits.end(),
                                        q) == ordered_qubits.end()) {
          proc_qubits.push_back(q);
        }
    }
    std::sort(proc_qubits.begin(), proc_qubits.end());
    ordered_qubits.insert(ordered_qubits.end(),proc_qubits.begin(),proc_qubits.end());
    if (comm_rank == 0){
      std::cout << "after merging ";
      for (auto q : ordered_qubits){
        std::cout << q << " ";
      }
    std::cout << std::endl;
    }
    //handle all the parts here iteratively
  
    //prepare two SvSimMPI objects in and out, each take the total number of qubits
    SvSimMPI svsim_mpi_in = SvSimMPI(MPI_COMM_WORLD,ordered_qubits);
    svsim_mpi_in.allocate();
    SvSimMPI svsim_mpi_out = SvSimMPI(MPI_COMM_WORLD,ordered_qubits);
    svsim_mpi_out.allocate();
    SvSimMPI *svsim_mpi_in_ptr = &svsim_mpi_in;
    SvSimMPI *svsim_mpi_out_ptr = &svsim_mpi_out;
    int part_count = 0;
    prep_time = MPI_Wtime() - prep_time;
    std::vector<double> compute_part;
    for (auto & part : parts){
       std::vector<GateBuffer> gates_buffer;
      if (part_count != 0){
        std::set<Qubit> needed_qubits_in_part;
        for (auto& bg : part) {
          needed_qubits_in_part.insert(bg.targets().begin(),
                           bg.targets().begin() + bg.num_targets());
          needed_qubits_in_part.insert(bg.controls().begin(),
                           bg.controls().begin() + bg.num_controls());
        }
        std::stringstream ss_needs;
        if (comm_rank == 0){
          
          for (auto q : needed_qubits_in_part)
          {
            ss_needs << q << " ";
          }
          std::cout << "needed_qubits_in_part " << ss_needs.str() << std::endl;
        }
        // idenfity needed qubits in part : local qubits
        std::vector<Qubit> qubit_vec_in_part{needed_qubits_in_part.begin(), needed_qubits_in_part.end()};
        // adding fast slots
        if (opt_slots == 1){
          for (int i = 0; i < num_total_qubits; i++){
            if (std::find(qubit_vec_in_part.begin(),qubit_vec_in_part.end(),i) == qubit_vec_in_part.end()
                && qubit_vec_in_part.size() < num_local_qubits)
                qubit_vec_in_part.push_back(i);
          }
        }
        // order them
        std::vector<Qubit> ordered_qubits_in_part{qubit_vec_in_part.begin(),qubit_vec_in_part.end()};
        std::sort(ordered_qubits_in_part.begin(), ordered_qubits_in_part.end());
        // identify the proc qubits and put them in a set
        std::set<Qubit> outer_qubits_in_part;
        for (Qubit q = 0; q < num_total_qubits; ++q) {
          if (std::find(ordered_qubits_in_part.begin(), ordered_qubits_in_part.end(),
            q) == ordered_qubits_in_part.end()) {
            outer_qubits_in_part.insert(q);
          }
        }

        std::stringstream ss_proc;
        if (comm_rank == 0){
          for (auto it = outer_qubits_in_part.begin(); it !=
                             outer_qubits_in_part.end(); ++it)
            ss_proc << ' ' << *it;
          std::cout << "proc qubits " << ss_proc.str() << std::endl;
        }
        //print_buf(sv_current, num_local_els, MPI_COMM_WORLD) << std::flush;
        //gather_slots(sv_current->state(), sv_rank->state(), num_proc_slots + num_local_slots, outer_qubits_in_part,
        //           MPI_COMM_WORLD);
        //auto start_gather = std::chrono::high_resolution_clock::now();
        auto tmp_time = MPI_Wtime();
        svsim_mpi_in_ptr->gather_qubits(*svsim_mpi_out_ptr,outer_qubits_in_part);
        //auto end_gather = std::chrono::high_resolution_clock::now();
        //mpi_communication_timer += std::chrono::duration_cast<std::chrono::microseconds>(end_gather-start_gather).count();
        comm_time += MPI_Wtime() - tmp_time;
      }
      
      for (auto& bg : part) {
        gates_buffer.push_back(bg.apply_to_buffer(svsim_mpi_out_ptr->local_state_vector()));
      }
      // test for one gate 
      //auto bg = part[0];
      //gates_buffer.push_back(bg.apply_to_buffer(svsim_mpi_out_ptr->local_state_vector()));
      //auto start_compute = std::chrono::high_resolution_clock::now();
      auto tmp_time_2 = MPI_Wtime();
      execute_on(svsim_mpi_out_ptr->local_state_vector(),gates_buffer);
      //auto end_compute = std::chrono::high_resolution_clock::now();
      //mpi_compute_timer += std::chrono::duration_cast<std::chrono::microseconds>(end_compute-start_compute).count();
      auto compute_in_part = MPI_Wtime() - tmp_time_2;
      compute_time += compute_in_part;
      compute_part.push_back(compute_in_part);
      std::swap(svsim_mpi_in_ptr, svsim_mpi_out_ptr);
      //std::cout << "maps" << std::endl;
      //svsim_mpi_in_ptr->get_qubit_slot_map();
      //svsim_mpi_out_ptr->get_qubit_slot_map();
      part_count ++;
    }
    end_to_end_time = MPI_Wtime() - end_to_end_time;
    // MPI timing reduction
    // prep
    MPI_Reduce(&prep_time, &max_prep_time, 1, MPI_DOUBLE,MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&prep_time, &min_prep_time, 1, MPI_DOUBLE, MPI_MIN, 0,MPI_COMM_WORLD);
    MPI_Reduce(&prep_time, &avg_prep_time, 1, MPI_DOUBLE, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(&comm_time, &max_comm_time, 1, MPI_DOUBLE,MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_time, &min_comm_time, 1, MPI_DOUBLE, MPI_MIN, 0,MPI_COMM_WORLD);
    MPI_Reduce(&comm_time, &avg_comm_time, 1, MPI_DOUBLE, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(&compute_time, &max_compute_time, 1, MPI_DOUBLE,MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&compute_time, &min_compute_time, 1, MPI_DOUBLE, MPI_MIN, 0,MPI_COMM_WORLD);
    MPI_Reduce(&compute_time, &avg_compute_time, 1, MPI_DOUBLE, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(&end_to_end_time, &max_end_to_end_time, 1, MPI_DOUBLE,MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&end_to_end_time, &min_end_to_end_time, 1, MPI_DOUBLE, MPI_MIN, 0,MPI_COMM_WORLD);
    MPI_Reduce(&end_to_end_time, &avg_end_to_end_time, 1, MPI_DOUBLE, MPI_SUM, 0,MPI_COMM_WORLD);
    std::vector<double> max_compute_part;
    std::vector<double> min_compute_part;
    std::vector<double> sum_compute_part;
    double max,min,sum;
    for (auto& c_part : compute_part){
      MPI_Reduce(&c_part, &max, 1, MPI_DOUBLE,MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&c_part, &min, 1, MPI_DOUBLE,MPI_MIN, 0, MPI_COMM_WORLD);
      MPI_Reduce(&c_part, &sum, 1, MPI_DOUBLE,MPI_SUM, 0, MPI_COMM_WORLD);
      max_compute_part.push_back(max);
      min_compute_part.push_back(min);
      sum_compute_part.push_back(sum);
    }
    if (comm_rank == 0){
      std::cout << " MPI communiation time max: " << max_comm_time << " min: " << min_comm_time << " avg: " << avg_comm_time/comm_size << std::endl;
      std::cout << " MPI compute time  max: " << max_compute_time << " min: " << min_compute_time << " avg: " << avg_compute_time/comm_size << std::endl;
       std::cout << " MPI preperation time max: " << max_prep_time << " min: " << min_prep_time << " avg: " << avg_prep_time/comm_size << std::endl;
      std::cout << " MPI end-to-end time  max: " << max_end_to_end_time << " min: " << min_end_to_end_time << " avg: " << avg_end_to_end_time/comm_size << std::endl;
      for (size_t i = 0; i < compute_part.size(); i++){
        std::cout << " MPI compute for part " << i << " max: " << max_compute_part[i] << " min: " << min_compute_part[i] << " avg: " << sum_compute_part[i]/comm_size << std::endl;
      }
    }
    //std::cout << "MPI communication time " << comm_time << " Rank "<< comm_rank << std::endl;
  }

  void gather_and_execute_multilevel_on_mpi(int num_total_qubits, int num_local_qubits,
                                   std::vector<std::vector<std::vector<BoundGate>>>& multi_parts)
  {
    int comm_rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    double prep_time =0, comm_time = 0, compute_time = 0, end_to_end_time = 0;
    double max_prep_time =0, max_comm_time = 0, max_compute_time = 0, max_end_to_end_time = 0;
    double min_prep_time =0, min_comm_time = 0, min_compute_time = 0, min_end_to_end_time = 0;
    double avg_prep_time =0, avg_comm_time = 0, avg_compute_time = 0, avg_end_to_end_time = 0;
    prep_time = MPI_Wtime();
    end_to_end_time = MPI_Wtime();
    std::vector<std::vector<BoundGate>> part_0_lists = multi_parts[0];
    // take the first part and create the buf_in and buf_out state vector
    std::vector<BoundGate> first_part = part_0_lists[0];
    std::set<Qubit> needed_qubits;
    std::vector<Qubit> top_qubit_for_multi;
    for (const auto& bg : first_part) {
      needed_qubits.insert(bg.targets().begin(),
                           bg.targets().begin() + bg.num_targets());
      needed_qubits.insert(bg.controls().begin(),
                           bg.controls().begin() + bg.num_controls());
    }
    
    std::vector<Qubit> qubit_vec{needed_qubits.begin(), needed_qubits.end()};
    // adding fast slots
    if (opt_slots == 1){
      for (int i = 0; i < num_total_qubits; i++){
        if (std::find(qubit_vec.begin(),qubit_vec.end(),i) == qubit_vec.end()
            && qubit_vec.size() < num_local_qubits)
            qubit_vec.push_back(i);
      }
    }
    std::vector<Qubit> ordered_qubits{qubit_vec.begin(),qubit_vec.end()};
    std::sort(ordered_qubits.begin(), ordered_qubits.end());
    top_qubit_for_multi = ordered_qubits;
    if (comm_rank == 0)
    {
      std::cout << "size of the first ordered qubit " << ordered_qubits.size() << std::endl;
      std::cout << "contains qubit ";
      /*for (auto q : ordered_qubits)
      {
        std::cout << q << " ";
      }
      std::cout << std::endl;*/
    }
    std::vector<Qubit> proc_qubits;
    for (Qubit q = 0; q < num_total_qubits; ++q) {
        if (std::find(ordered_qubits.begin(), ordered_qubits.end(),
                                        q) == ordered_qubits.end()) {
          proc_qubits.push_back(q);
        }
    }
    std::sort(proc_qubits.begin(), proc_qubits.end());
    ordered_qubits.insert(ordered_qubits.end(),proc_qubits.begin(),proc_qubits.end());
    if (comm_rank == 0){
      std::cout << "after merging ";
      for (auto q : ordered_qubits){
        std::cout << q << " ";
      }
    std::cout << std::endl;
    }
    //handle all the parts here iteratively
  
    //prepare two SvSimMPI objects in and out, each take the total number of qubits
    SvSimMPI svsim_mpi_in = SvSimMPI(MPI_COMM_WORLD,ordered_qubits);
    svsim_mpi_in.allocate();
    SvSimMPI svsim_mpi_out = SvSimMPI(MPI_COMM_WORLD,ordered_qubits);
    svsim_mpi_out.allocate();
    SvSimMPI *svsim_mpi_in_ptr = &svsim_mpi_in;
    SvSimMPI *svsim_mpi_out_ptr = &svsim_mpi_out;
    int part_count = 0;
    prep_time = MPI_Wtime() - prep_time;
    int top_count = 0;
    for (auto & top_part : multi_parts){
        if (top_count != 0){
          std::set<Qubit> needed_qubits_in_part;
          std::vector<BoundGate> part = top_part[0];
          // the first element of the top_part contains all the gates in this part.
          for (auto& bg : part) {
            needed_qubits_in_part.insert(bg.targets().begin(),
                           bg.targets().begin() + bg.num_targets());
            needed_qubits_in_part.insert(bg.controls().begin(),
                           bg.controls().begin() + bg.num_controls());
          }
          if (comm_rank == 0){
            std::cout << "needed_qubits_in_part ";
            for (auto q : needed_qubits_in_part)
            {
              std::cout << q << " ";
            }
            std::cout << std::endl;
          }
          // idenfity needed qubits in part : local qubits
          std::vector<Qubit> qubit_vec_in_part{needed_qubits_in_part.begin(), needed_qubits_in_part.end()};
          // adding fast slots: there may not be enough qubits to reach the number of local qubits
          if (opt_slots == 1){
            for (int i = 0; i < num_total_qubits; i++){
              if (std::find(qubit_vec_in_part.begin(),qubit_vec_in_part.end(),i) == qubit_vec_in_part.end()
                  && qubit_vec_in_part.size() < num_local_qubits)
                  qubit_vec_in_part.push_back(i);
            }
          }
          // order them
          std::vector<Qubit> ordered_qubits_in_part{qubit_vec_in_part.begin(),qubit_vec_in_part.end()};
          std::sort(ordered_qubits_in_part.begin(), ordered_qubits_in_part.end());
          top_qubit_for_multi = ordered_qubits_in_part;
          // identify the proc qubits and put them in a set
          std::set<Qubit> outer_qubits_in_part;
          for (Qubit q = 0; q < num_total_qubits; ++q) {
            if (std::find(ordered_qubits_in_part.begin(), ordered_qubits_in_part.end(),
              q) == ordered_qubits_in_part.end()) {
              outer_qubits_in_part.insert(q);
            }
          }
          if (comm_rank == 0){
            std::cout << "proc qubits ";
            for (auto it = outer_qubits_in_part.begin(); it !=
                             outer_qubits_in_part.end(); ++it)
              std::cout << ' ' << *it;
            std::cout << std::endl;
          }
        //print_buf(sv_current, num_local_els, MPI_COMM_WORLD) << std::flush;
        //gather_slots(sv_current->state(), sv_rank->state(), num_proc_slots + num_local_slots, outer_qubits_in_part,
        //           MPI_COMM_WORLD);
        //auto start_gather = std::chrono::high_resolution_clock::now();
          auto tmp_time = MPI_Wtime();
          svsim_mpi_in_ptr->gather_qubits(*svsim_mpi_out_ptr,outer_qubits_in_part);
        //auto end_gather = std::chrono::high_resolution_clock::now();
        //mpi_communication_timer += std::chrono::duration_cast<std::chrono::microseconds>(end_gather-start_gather).count();
          comm_time += MPI_Wtime() - tmp_time;
        }
        top_count ++;
        for (int i = 1; i < top_part.size(); i++){
          std::vector<BoundGate>& second_level_part = top_part[i];
          auto tmp_time_2 = MPI_Wtime();
          //gather_and_execute_on(svsim_mpi_out_ptr->local_state_vector(),second_level_part);
          gather_and_execute_on_multi(svsim_mpi_out_ptr->local_state_vector(),second_level_part,top_qubit_for_multi,16);
          compute_time += MPI_Wtime() - tmp_time_2;
        }
        //auto end_compute = std::chrono::high_resolution_clock::now();
        //mpi_compute_timer += std::chrono::duration_cast<std::chrono::microseconds>(end_compute-start_compute).count();
      
        std::swap(svsim_mpi_in_ptr, svsim_mpi_out_ptr);
      //std::cout << "maps" << std::endl;
      //svsim_mpi_in_ptr->get_qubit_slot_map();
      //svsim_mpi_out_ptr->get_qubit_slot_map();
    }
    std::cout << "number of parts " << top_count << std::endl; 
    end_to_end_time = MPI_Wtime() - end_to_end_time;
    // MPI timing reduction
    // prep
    MPI_Reduce(&prep_time, &max_prep_time, 1, MPI_DOUBLE,MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&prep_time, &min_prep_time, 1, MPI_DOUBLE, MPI_MIN, 0,MPI_COMM_WORLD);
    MPI_Reduce(&prep_time, &avg_prep_time, 1, MPI_DOUBLE, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(&comm_time, &max_comm_time, 1, MPI_DOUBLE,MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_time, &min_comm_time, 1, MPI_DOUBLE, MPI_MIN, 0,MPI_COMM_WORLD);
    MPI_Reduce(&comm_time, &avg_comm_time, 1, MPI_DOUBLE, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(&compute_time, &max_compute_time, 1, MPI_DOUBLE,MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&compute_time, &min_compute_time, 1, MPI_DOUBLE, MPI_MIN, 0,MPI_COMM_WORLD);
    MPI_Reduce(&compute_time, &avg_compute_time, 1, MPI_DOUBLE, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(&end_to_end_time, &max_end_to_end_time, 1, MPI_DOUBLE,MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&end_to_end_time, &min_end_to_end_time, 1, MPI_DOUBLE, MPI_MIN, 0,MPI_COMM_WORLD);
    MPI_Reduce(&end_to_end_time, &avg_end_to_end_time, 1, MPI_DOUBLE, MPI_SUM, 0,MPI_COMM_WORLD);
    if (comm_rank == 0){
      std::cout << " Multi MPI communiation time max: " << max_comm_time << " min: " << min_comm_time << " avg: " << avg_comm_time/comm_size << std::endl;
      std::cout << " Multi MPI compute time  max: " << max_compute_time << " min: " << min_compute_time << " avg: " << avg_compute_time/comm_size << std::endl;
       std::cout << "Multi MPI preperation time max: " << max_prep_time << " min: " << min_prep_time << " avg: " << avg_prep_time/comm_size << std::endl;
      std::cout << " Multi MPI end-to-end time  max: " << max_end_to_end_time << " min: " << min_end_to_end_time << " avg: " << avg_end_to_end_time/comm_size << std::endl;
    }
  }                  
  #endif // MPI
}; 


#endif // EXECUTE_HPP_
