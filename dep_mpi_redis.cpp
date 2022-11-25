#ifndef MPI_REDISTRIBUTER_HPP_
#define MPI_REDISTRIBUTER_HPP_
#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <unordered_set>
#include <set>
#include <vector>
#include "types.hpp"
#define ENABLE_TEST 1

namespace mpi_redistributer {

/**
 * @brief MPI redistribution for a power-of-2 distributed data structure.
 *
 * The data is assumed to be of total 2^(p+l) with 2^p MPI ranks, each with 2^l
 * elements. An element is located by a (p+l)-bit vector based on p-bit MPI rank
 * and l-bit intra-MPI-rank offset. In the input, the p bits are considered the
 * most significant bits (MSB). That is, all elements in rank 0 are ordered
 * before any in rank 1 and so on. Redistribution makes a different set of bits
 * to be the MPI rank (and thus MSB bits).
 *
 * Specifically, the input is addressed as [a_p, ... a_1, a_l, ... a_1]. Given a
 * set of new proc ranks q, the new order would be: the bits in q, the bits in p
 * not in q, the bits in l but not in q.
 *
 * The user code needs to track which bits are MSB at any point in time.
 *
 */

class MPIRedistributer {
 public:
  /**
   * @brief Construct a new MPIRedistributer object.
   *
   * The input distribution is assumed to be MPI rank slots followed by local
   * slots. The @param new_proc_slots are made the MPI rank slots, in descending
   * order.
   *
   * @param element_type MPI datatype of individual element (assumed to be a
   * primitive type)
   * @param num_total_slots total number of slots (mpi rank and intra-mpi slots)
   * @param num_proc_slots number of mpi rank slots
   * @param new_proc_slots New mpi ranks.
   *
   * @pre new_proc_slots.size() == num_proc_slots
   * @pre num_total_slots  > 0
   * @pre 0 < num_proc_slots < num_total_slots
   */
  MPIRedistributer(MPI_Datatype element_type, int num_total_slots,
                   int num_proc_slots, const std::set<int>& new_proc_slots)
      : element_type_{element_type},
        num_total_slots_{num_total_slots},
        num_proc_slots_{num_proc_slots},
        new_proc_slots_{new_proc_slots} {
    assert(num_total_slots_ > 0);
    assert(num_proc_slots_ < num_total_slots_);
    assert(num_proc_slots_ == new_proc_slots_.size());

    int num_local_slots = num_total_slots_ - num_proc_slots_;

    create_mpi_datatype();
    // compute recvcount
    recvcount_ = 1;
    for (auto dim : send_shape_) {
      recvcount_ *= dim;
    }
  }

  ~MPIRedistributer() { MPI_Type_free(&send_type_); }

  /**
   * @brief Communication plan for MPI rank
   *
   */
  struct CommunicationPlan {
    MPI_Datatype send_type;  ///< MPI data type to use for sending
    size_t recv_count;  ///< Number of elements of basic type to receive per MPI
                        ///< recv
    std::vector<size_t>
        send_displacement;  ///< Send displacement in buffer for each send
    std::vector<int> send_to_proc;    ///< Destination rank for each MPI send
    std::vector<int> recv_from_proc;  ///< Source rank for each MPI recv
  };

  /**
   * @brief Construct a communication plan for the given MPI rank
   * 
   * @param proc MPI rank for which communication plan needs to be prepared
   * @return CommunicationPlan Returned communication plan object
   */
  CommunicationPlan communication_plan(int proc) const {
    CommunicationPlan comm_plan;
    comm_plan.send_type = send_type_;
    comm_plan.recv_count = recvcount_;
    //@todo: compute proc ids and displs

    //              send proc calculation
    int num_local_slots = num_total_slots_ - num_proc_slots_;
    int num_send_bits = 0;
    int proc_offset = 0;
    int ctr = 0;
    for (auto slot : new_proc_slots_) {
      if (slot < num_local_slots) {
        num_send_bits += 1;
      } else {
        // assume new_proc_slots_ is sorted
        proc_offset += ((proc >> (slot - num_local_slots)) & 1) << ctr;
        ctr += 1;
      }
    }
    proc_offset <<= num_send_bits;
    for (int i = 0; i < (1ul << num_send_bits); i++) {
      comm_plan.send_to_proc.push_back(proc_offset + i);
    }
    //              send displacement calculation
    comm_plan.send_displacement.push_back(0);
    ctr = 0;
    for (auto slot : new_proc_slots_) {
      if (slot < num_local_slots) {
        int sz = comm_plan.send_displacement.size();
        for (int i = 0; i < sz; i++) {
          comm_plan.send_displacement.push_back(comm_plan.send_displacement[i] +
                                                (1ul << slot));
        }
      }
    }
    //              recv proc calculation
    ctr = 0;
    proc_offset = 0;
    comm_plan.recv_from_proc.push_back(0);
    {
      int i = 0;
      auto it = new_proc_slots_.begin();
      while (it != new_proc_slots_.end()) {
        if (*it >= num_local_slots) {
          proc_offset += ((proc >> i) & 1) << (*it - num_local_slots);
        }
        it++;
        i++;
      }
    }
    for (int slot = num_local_slots; slot < num_total_slots_; slot++) {
      if (!new_proc_slots_.count(slot)) {
        int sz = comm_plan.recv_from_proc.size();
        for (int i = 0; i < sz; i++) {
          comm_plan.recv_from_proc.push_back(comm_plan.recv_from_proc[i] +
                                             (1ul << (slot - num_local_slots)));
        }
      }
    }
    for (auto& rp : comm_plan.recv_from_proc) {
      rp += proc_offset;
    }
    return comm_plan;
  }

 private:
  /**
   * @brief Create a mpi datatype object for the communication to be performed.
   *
   */
  void create_mpi_datatype() {
    std::vector<int> array_of_sizes;
    std::vector<int> array_of_subsizes;
    int order = MPI_ORDER_C;

    int num_local_slots = num_total_slots_ - num_proc_slots_;
    int s = 0;
    while (s < num_local_slots) {
      int start = 0;
      while (s < num_local_slots && new_proc_slots_.count(s)) {
        s++;
      }
      if (s != start) {
        array_of_sizes.push_back(1 << (s - start));
        array_of_subsizes.push_back(1);
      }
      start = s;
      while (s < num_local_slots && !new_proc_slots_.count(s)) {
        s++;
      }
      if (s != start) {
        array_of_sizes.push_back(1 << (s - start));
        array_of_subsizes.push_back(1 << (s - start));
      }
    }
    std::reverse(array_of_sizes.begin(), array_of_sizes.end());
    std::reverse(array_of_subsizes.begin(), array_of_subsizes.end());
    std::vector<int> array_of_starts(array_of_sizes.size(), 0);
    send_shape_ = array_of_subsizes;

    // std::cout<<"send_shape.size() :"<<send_shape_.size()<<"\n";
    // for(int i=0; i<send_shape_.size(); i++) {
    //   std::cout<<send_shape_[i]<<"\n";
    // }

    int ndims = array_of_sizes.size();
    MPI_Type_create_subarray(ndims, array_of_sizes.data(),
                             array_of_subsizes.data(), array_of_starts.data(),
                             order, element_type_, &send_type_);
    MPI_Type_commit(&send_type_);
  }

  MPI_Datatype element_type_;
  int num_total_slots_;
  int num_proc_slots_;
  std::set<int> new_proc_slots_;

  std::vector<int> send_shape_;
  size_t recvcount_;
  MPI_Datatype send_type_;
}; // class MPIRedistributer

/**
 * @brief MPI_Datatype for a given C/C++ type
 *
 * @tparam T native C/C++ type
 * @return MPI_Datatype corresponding MPI type
 */
template <typename T>
MPI_Datatype mpi_data_type() {
  return MPI_INT;
}

template <>
MPI_Datatype mpi_data_type<double>() {
  return MPI_DOUBLE;
}
/**
 * @brief gather slots in input state vector buffer to output state vector
 * buffer. redistributes the given slots to be "innermost" in the slot order
 *
 * @param buf_in input buffer
 * @param buf_out output buffer
 * @param num_slots num_slots in both buffers (need to be same)
 * @param slots slots to be gathered
 */
template<typename Type>
void gather_slots(const Type* buf_in, Type* buf_out, int num_total_slots,
                  const std::set<int>& new_proc_slots, MPI_Comm comm) {
  int comm_rank, comm_size;
  MPI_Comm_rank(comm, &comm_rank);
  MPI_Comm_size(comm, &comm_size);
  assert(comm_size == (1ul << new_proc_slots.size()));

  int num_proc_slots = std::log2(comm_size);
  assert(num_proc_slots == new_proc_slots.size());
  assert(comm_size ==
         (1ul << num_proc_slots));  // only power of 2 comm sizes for now
  MPI_Datatype element_type = mpi_data_type<Type>();
  MPIRedistributer mpir(element_type, num_total_slots, num_proc_slots,
                        new_proc_slots);
  MPIRedistributer::CommunicationPlan comm_plan =
      mpir.communication_plan(comm_rank);

  // std::cout<<"recvcount="<<mpir.recvcount_<<"\n";

  std::vector<MPI_Request> reqs(comm_plan.recv_from_proc.size(),
                                MPI_REQUEST_NULL);
  for (int i = 0; i < comm_plan.recv_from_proc.size(); i++) {
    MPI_Irecv(buf_out + i * comm_plan.recv_count, comm_plan.recv_count,
              element_type, comm_plan.recv_from_proc[i], 0, comm, &reqs[i]);
  }
  for (int i = 0; i < comm_plan.send_to_proc.size(); i++) {
    // std::cout<<comm_rank<<" : send at off = "<<
    // comm_plan.send_displacement[i]<<std::endl;
    MPI_Send(buf_in + comm_plan.send_displacement[i], 1, comm_plan.send_type,
             comm_plan.send_to_proc[i], 0, comm);
  }
  for (auto& req : reqs) {
    MPI_Status status;
    MPI_Wait(&req, &status);
  }
} // gather_slots

}  // namespace mpi_redistributer



//------------------------------------------------------------------------------
//
//    Code below is for testing. Should be disabled when used in application
//
//------------------------------------------------------------------------------

#if ENABLE_TEST

/**
 * @brief allocate a buffer to hold @param num_slots slots
 *
 * @param num_slots num of qubits slots to hold in this buffer
 * @return Type* allocated buffer
 */
template <typename Type>
Type* alloc(int num_slots) {
  Type* ret = new Type[1ul << num_slots];
  for (size_t i = 0; i < (1ul << num_slots); i++) {
    ret[i] = 0;
  }
  return ret;
}

/**
 * @brief deallocate buffer
 *
 * @param buf qubit slot buffer to deallocate
 */
template<typename Type>
void dealloc(Type* buf) { delete[] buf; }

template <typename T>
std::ostream& print_buf(T* buf, size_t num_local_els, MPI_Comm comm) {
  int comm_rank, comm_size;
  MPI_Comm_rank(comm, &comm_rank);
  MPI_Comm_size(comm, &comm_size);

  auto& os = std::cout;
  for (int i = 0; i < comm_size; i++) {
    if (i == comm_rank) {
      os << i << ": ";
      for (int i = 0; i < num_local_els; i++) {
        os << buf[i] << "\t";
      }
      os << std::endl;
    }
    MPI_Barrier(comm);
  }
  return os;
}



int main(int argc, char* argv[]) {
  using namespace mpi_redistributer;
  using namespace SvSim;
  int comm_rank, comm_size;
  double start, end;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Barrier(MPI_COMM_WORLD);
  
  int num_local_slots = 26;

  //--------------------------------------
  int num_local_els = 1ul << num_local_slots;
  Type* buf = alloc<Type>(num_local_slots);
  for (int i = 0; i < num_local_els; i++) {
    buf[i] = comm_rank * num_local_els + i;
  }
  MPI_Barrier(MPI_COMM_WORLD);

  Type* buf2 = alloc<Type>(num_local_slots);
  int num_proc_slots = std::log2(comm_size);
  std::vector<std::set<int>> out_list;
  std::set<int> new_proc_slots{24,25,27,28};
  std::set<int> new_proc_slots2{16,18,20,26};
  //std::set<int> new_proc_slots{8,9,10,11};
  //std::set<int> new_proc_slots2{7,8,10,11};
  out_list.push_back(new_proc_slots);
  out_list.push_back(new_proc_slots2);
  MPI_Barrier(MPI_COMM_WORLD);
  //for(int i=0; i<std::log2(comm_size); i++) {
  //  new_proc_slots.insert(i);
  //}
  // assuming there are two parts
  start = MPI_Wtime();
  for(int i = 0; i < 2; i ++){
    gather_slots(buf, buf2, num_proc_slots + num_local_slots, out_list[i],
               MPI_COMM_WORLD);
    
    //if (comm_rank == 0) {
    //  std::cout << "Output after redistribution " << std::endl;
    //}
    //print_buf(buf2, num_local_els, MPI_COMM_WORLD) << std::flush;
    MPI_Barrier(MPI_COMM_WORLD);
    std::swap(buf,buf2);
    MPI_Barrier(MPI_COMM_WORLD);
  }
  end = MPI_Wtime();
  if (comm_rank == 0){
    std::cout << "Runtime = " << end-start << std::endl;
  }
  dealloc(buf2);
  //-----------------------------------
  dealloc(buf);
  MPI_Finalize();
  return 0;
}

#endif
#endif
