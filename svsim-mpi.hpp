#include <mpi.h>

#include <set>
#include <vector>
#include <cmath>
#include <cassert>
#include "state_vector.hpp"
#include "mpi_redistributer.hpp"

namespace SvSim {

class SvSimMPI {
 public:
  SvSimMPI() : allocated_{false}, comm_{MPI_COMM_SELF} { 
      comm_size_ = 1;
  }

  SvSimMPI(const SvSimMPI&) = default;

  /**
   * @brief Construct a new SvSimMPI object. qubits[0] will be the fastest
   * varying.
   *
   * @param comm
   * @param qubits
   */
  SvSimMPI(MPI_Comm comm, std::vector<Qubit>& qubits)
      : allocated_{false}, comm_{comm} {
    // validate
    assert(std::set(qubits.begin(), qubits.end()).size() == qubits.size());
    MPI_Comm_size(comm_, &comm_size_);
    int slot = 0;
    for(auto q : qubits) {
        qubit_to_slot_[q] = slot;
        slot_to_qubit_[slot] = q;
        slot++;
    }
    int num_proc_qubit = std::log2(comm_size_);
    int num_local_qubit = slot - num_proc_qubit;
    for (size_t i = 0; i < slot; i++){
        if (i < num_local_qubit)
          local_qubits_.push_back(qubits[i]);
        else
          proc_qubits_.push_back(qubits[i]);
    }
  }

  SvSimMPI(MPI_Comm comm, int num_qubits)
      : allocated_{false}, comm_{comm} {
    // validate
    assert(num_qubits>0);
    MPI_Comm_size(comm_, &comm_size_);
    int slot = 0;
    for (int q=0; q<num_qubits; q++) {
      qubit_to_slot_[q] = slot;
      slot_to_qubit_[slot] = q;
      slot++;
    }
    int num_proc_qubit = std::log2(comm_size_);
    int num_local_qubit = num_qubits - num_proc_qubit;
    for (size_t i = 0; i < slot; i++){
        if (i < num_local_qubit)
          local_qubits_.push_back(slot_to_qubit_[i]);
        else
          proc_qubits_.push_back(slot_to_qubit_[i]);
    }
  }

  MPI_Comm comm() const {
    assert(allocated_);
    return comm_;
  }

  void get_qubit_slot_map(){
      for (auto q: qubit_to_slot_){
      std::cout << q.first  << ":" << q.second << " ";
    }
    std::cout << std::endl;
    for (auto s: slot_to_qubit_){
      std::cout << s.first << ":" << s.second << " ";
    }
  }

  void allocate() {
    if (allocated_) return;
    std::stringstream ss_qubits;
    for (auto q : local_qubits_)
     ss_qubits << q << " ";
    //std::cout << "local qubits for allocation" << ss_qubits.str() << std::endl;
    sv_ = new StateVector(local_qubits_,{},0);
   
    allocated_ = true;
  }

  void deallocate() {
    if (!allocated_) return;
    delete sv_;
    sv_ = nullptr;
    allocated_ = false;
  }

  bool allocated() const {
      return allocated_;
  }

  StateVector& local_state_vector() {
    assert(allocated_);
    return *sv_;
  }

  int num_proc_qubits() { return std::log2(comm_size_); }

  int num_total_qubits() { return qubit_to_slot_.size(); }

  int num_local_qubits() { return num_total_qubits() - num_proc_qubits(); }

    /**
     * @brief 
     * 
     * @param sv_out 
     * @param new_proc_qubits new_proc_qubits[0] is fastest varying
     */
  void gather_qubits(SvSimMPI& sv_out, const std::set<Qubit>& new_proc_qubits) {
    assert(allocated_);
    // check new proc qubits are valid in size and numbers
    assert(new_proc_qubits.size() == num_proc_qubits());
    for (auto npq : new_proc_qubits) {
      // is either a local or a proc qubit
      assert(qubit_to_slot_.count(npq));
    }
    // compute new qubit order
    //give by: p | q | r | s
    //p: local qubits in both current and new state vector, starting with fastest slot in current state vector
    //q: local qubits in new state vector, 
    //r: proc qubits in current local sv, starting with fastest
    //s: new_proc_qubits[0:]
    std::vector<Qubit> new_qubit_order;
    // p then q qubits -- current local then remote qubits that are local in new order
    for(int i=0; i<num_total_qubits(); i++) {
        if(!new_proc_qubits.count(slot_to_qubit_[i])) {
            new_qubit_order.push_back(slot_to_qubit_[i]);
        }
    }
    // r then s qubits -- current local then remote qubits that are remote in new order
    for (int i = 0; i<num_total_qubits(); i++) {
      if (new_proc_qubits.count(slot_to_qubit_[i])) {
        new_qubit_order.push_back(slot_to_qubit_[i]);
      }
    }
   // std::cout << "inside gather slot, new qubit order " ;
  //  for (auto q: new_qubit_order)
    //  std::cout << q << " ";
  //  std::cout << std::endl;
    // construct sv_out
    if (sv_out.allocated()) {
      sv_out.deallocate();
    }
    sv_out = SvSimMPI(comm_, new_qubit_order);
    sv_out.allocate();

    std::set<int> new_proc_slots;
    for (auto npq : new_proc_qubits) {
      new_proc_slots.insert(qubit_to_slot_[npq]);
    }

    // call gather slots into sv_out.local_state_vector()
    gather_slots(local_state_vector().state(),
                 sv_out.local_state_vector().state(), num_total_qubits(),
                 new_proc_slots, comm_);
    /*gather_slots_alltoall(local_state_vector().state(),
                 sv_out.local_state_vector().state(), num_total_qubits(),
                 new_proc_slots, comm_);*/
  }

  /*SvSimMPI gather_qubits(const std::vector<Qubit>& new_proc_qubits) {
    assert(allocated_);
    SvSimMPI sv_out;
    gather_qubits(sv_out, new_proc_qubits);
    return sv_out;
  }*/

    /*int num_total_qubits() const {
        assert(allocated_);
        return local_qubits_.size() + proc_qubits_.size();
    }*/

 private:
  bool allocated_;
  StateVector *sv_;
  MPI_Comm comm_;
  int comm_size_;
  std::map<Qubit,Slot> qubit_to_slot_;
  std::map<Slot,Qubit> slot_to_qubit_;
  std::vector<Qubit> local_qubits_;
  std::vector<Qubit> proc_qubits_;
}; // class SvSimMPI

}; // namespace SvSim
