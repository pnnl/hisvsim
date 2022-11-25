//------------------------------------------------------------------------------
// Copyright (C) 2019 Intel Corporation 
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//------------------------------------------------------------------------------

/// @file getting_started_with_IQS.cpp
/// Tutorial on the basic use of Intel Quantum Simulator (IQS).

// Include the header file with the declaration of all classes and methods of IQS.
#include "../include/qureg.hpp"

/////////////////////////////////////////////////////////////////////////////////////////

// Start of the main program (C++ language).
int main(int argc, char **argv)
{
#ifndef INTELQS_HAS_MPI
  std::cout << "\nThis introductory code is thought to be run with MPI.\n"
            << "To do so, please set the option '-DIqsMPI=ON' when calling CMake.\n\n"
            << "However the code will execute also without MPI.\n\n";
#endif


/////////////////////////////////////////////////////////////////////////////////////////
// Setting the MPI environment
/////////////////////////////////////////////////////////////////////////////////////////

  // Create the MPI environment, passing the same argument to all the ranks.
  iqs::mpi::Environment env(argc, argv);
  // IQS is structured so that only 2^k ranks are used to store and manipulate
  // the quantum state. In case the number of ranks differ from a power of 2,
  // all ranks in excess are effectively excluded from the computations and called
  // 'dummy'. The dummy ranks should be terminated.
  if (env.IsUsefulRank() == false) return 0;
  // IQS has functions that simplify some MPI instructions. However, it is important
  // to keep trace of the current rank.
  int tot_num_ranks = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &tot_num_ranks);

  int myid = env.GetStateRank();
  double end_to_end_time = 0, max_end_to_end_time = 0, min_end_to_end_time = 0, avg_end_to_end_time = 0;
  end_to_end_time = MPI_Wtime();
  // NOTE: Above, we asked for the 'state rank', meaning that we are considering the MPI
  // ranks that are used to store and manipulate a single quantum state. This difference
  // is unnecessary when simulating ideal quantum circuits, but once noise is introduced
  // via the noise-gate approach the situation may differ.

/////////////////////////////////////////////////////////////////////////////////////////
// Initialize the state of the quantum register
/////////////////////////////////////////////////////////////////////////////////////////
/* IQS stores a full representation of the quantum state in the computational basis.
 * In practice, the quantum state of N qubits is represented as a complex vector with
 * 2^N components.
 *
 * Each component corresponds to the probability amplitude of a specific computational
 * basis state:
 *     ψ(k)=⟨k|ψ⟩
 * with the index k corresponding to the N-bit integer in decimal representation, and
 * k∈{0,1,2,…,2N−1}.
 */
/////////////////////////////////////////////////////////////////////////////////////////

  // Allocate memory for the quantum register's state and initialize it to |0000>.
  // This can be achieved by using the codeword "base".
  int num_qubits = 31;
  iqs::QubitRegister<ComplexDP> psi (num_qubits);
  std::size_t index = 0;
  // The state can be initialized to a random state. To allow such initialization,
  // we need to declare a (pseudo) random number generator...

/////////////////////////////////////////////////////////////////////////////////////////
// Display the quantum state
/////////////////////////////////////////////////////////////////////////////////////////
/* It is important to be able to access and visualize the quantum state.
 * IQS allows to access the single components of the state or to print a comprehensive
 * description.
 * What index is associated to state |1011⟩? In decimal representation one has:
 *     1011 → 1×2^0 + 0×2^1 + 1×2^2 + 1×2^3 = 1+4+8 = 13
 * therefore it corresponds to the computational basis state with index 13.
 *
 * NOTE: contrary to what is adopted in decimal notation, our binary representation
 * must be read from left to right (from least significant to most significant bit).
 */
/////////////////////////////////////////////////////////////////////////////////////////

  // Initialize the state to |1000>.
  // The index of |1000> in decimal representation is 1.

  // Prepare the state |+-01>.
  psi.Initialize("base", index);
psi.ApplyPauliX(30);
psi.ApplyHadamard(0);
psi.ApplyHadamard(1);
psi.ApplyHadamard(2);
psi.ApplyHadamard(3);
psi.ApplyHadamard(4);
psi.ApplyHadamard(5);
psi.ApplyHadamard(6);
psi.ApplyHadamard(7);
psi.ApplyHadamard(8);
psi.ApplyHadamard(9);
psi.ApplyHadamard(10);
psi.ApplyHadamard(11);
psi.ApplyHadamard(12);
psi.ApplyHadamard(13);
psi.ApplyHadamard(14);
psi.ApplyHadamard(15);
psi.ApplyHadamard(16);
psi.ApplyHadamard(17);
psi.ApplyHadamard(18);
psi.ApplyHadamard(19);
psi.ApplyHadamard(20);
psi.ApplyHadamard(21);
psi.ApplyHadamard(22);
psi.ApplyHadamard(23);
psi.ApplyHadamard(24);
psi.ApplyHadamard(25);
psi.ApplyHadamard(26);
psi.ApplyHadamard(27);
psi.ApplyHadamard(28);
psi.ApplyHadamard(29);
psi.ApplyHadamard(30);
psi.ApplyPauliX(4);
psi.ApplyPauliX(7);
psi.ApplyPauliX(15);
psi.ApplyPauliX(16);
psi.ApplyPauliX(18);
psi.ApplyPauliX(19);
psi.ApplyPauliX(20);
psi.ApplyPauliX(23);
psi.ApplyPauliX(27);
psi.ApplyPauliX(28);
psi.ApplyHadamard(2);
psi.ApplyHadamard(3);
psi.ApplyHadamard(5);
psi.ApplyHadamard(6);
psi.ApplyHadamard(8);
psi.ApplyHadamard(9);
psi.ApplyHadamard(10);
psi.ApplyHadamard(11);
psi.ApplyHadamard(12);
psi.ApplyHadamard(13);
psi.ApplyHadamard(14);
psi.ApplyHadamard(17);
psi.ApplyHadamard(21);
psi.ApplyHadamard(22);
psi.ApplyHadamard(24);
psi.ApplyHadamard(25);
psi.ApplyHadamard(26);
psi.ApplyHadamard(29);
psi.ApplyToffoli(0,1,30);
psi.ApplyPauliX(4);
psi.ApplyPauliX(7);
psi.ApplyPauliX(15);
psi.ApplyPauliX(16);
psi.ApplyPauliX(18);
psi.ApplyPauliX(19);
psi.ApplyPauliX(20);
psi.ApplyPauliX(23);
psi.ApplyPauliX(27);
psi.ApplyPauliX(28);
psi.ApplyPauliX(2);
psi.ApplyPauliX(3);
psi.ApplyPauliX(5);
psi.ApplyPauliX(6);
psi.ApplyPauliX(8);
psi.ApplyPauliX(9);
psi.ApplyPauliX(10);
psi.ApplyPauliX(11);
psi.ApplyPauliX(12);
psi.ApplyPauliX(13);
psi.ApplyPauliX(14);
psi.ApplyPauliX(17);
psi.ApplyPauliX(21);
psi.ApplyPauliX(22);
psi.ApplyPauliX(24);
psi.ApplyPauliX(25);
psi.ApplyPauliX(26);
psi.ApplyPauliX(29);
psi.ApplyHadamard(0);
psi.ApplyHadamard(1);
psi.ApplyHadamard(4);
psi.ApplyHadamard(7);
psi.ApplyHadamard(15);
psi.ApplyHadamard(16);
psi.ApplyHadamard(18);
psi.ApplyHadamard(19);
psi.ApplyHadamard(20);
psi.ApplyHadamard(23);
psi.ApplyHadamard(27);
psi.ApplyHadamard(28);
psi.ApplyPauliX(2);
psi.ApplyPauliX(3);
psi.ApplyPauliX(5);
psi.ApplyPauliX(6);
psi.ApplyPauliX(8);
psi.ApplyPauliX(9);
psi.ApplyPauliX(10);
psi.ApplyPauliX(11);
psi.ApplyPauliX(12);
psi.ApplyPauliX(13);
psi.ApplyPauliX(14);
psi.ApplyPauliX(17);
psi.ApplyPauliX(21);
psi.ApplyPauliX(22);
psi.ApplyPauliX(24);
psi.ApplyPauliX(25);
psi.ApplyPauliX(26);
psi.ApplyPauliX(29);
psi.ApplyPauliX(0);
psi.ApplyPauliX(1);
psi.ApplyPauliX(4);
psi.ApplyPauliX(7);
psi.ApplyPauliX(15);
psi.ApplyPauliX(16);
psi.ApplyPauliX(18);
psi.ApplyPauliX(19);
psi.ApplyPauliX(20);
psi.ApplyPauliX(23);
psi.ApplyPauliX(27);
psi.ApplyPauliX(28);
psi.ApplyHadamard(2);
psi.ApplyHadamard(3);
psi.ApplyHadamard(5);
psi.ApplyHadamard(6);
psi.ApplyHadamard(8);
psi.ApplyHadamard(9);
psi.ApplyHadamard(10);
psi.ApplyHadamard(11);
psi.ApplyHadamard(12);
psi.ApplyHadamard(13);
psi.ApplyHadamard(14);
psi.ApplyHadamard(17);
psi.ApplyHadamard(21);
psi.ApplyHadamard(22);
psi.ApplyHadamard(24);
psi.ApplyHadamard(25);
psi.ApplyHadamard(26);
psi.ApplyHadamard(29);
psi.ApplyHadamard(1);
psi.ApplyPauliX(4);
psi.ApplyPauliX(7);
psi.ApplyPauliX(15);
psi.ApplyPauliX(16);
psi.ApplyPauliX(18);
psi.ApplyPauliX(19);
psi.ApplyPauliX(20);
psi.ApplyPauliX(23);
psi.ApplyPauliX(27);
psi.ApplyPauliX(28);
psi.ApplyCPauliX(0,1);
psi.ApplyHadamard(4);
psi.ApplyHadamard(7);
psi.ApplyHadamard(15);
psi.ApplyHadamard(16);
psi.ApplyHadamard(18);
psi.ApplyHadamard(19);
psi.ApplyHadamard(20);
psi.ApplyHadamard(23);
psi.ApplyHadamard(27);
psi.ApplyHadamard(28);
psi.ApplyHadamard(1);
psi.ApplyPauliX(0);
psi.ApplyPauliX(1);
psi.ApplyHadamard(0);
psi.ApplyHadamard(1);
    

  end_to_end_time = MPI_Wtime() - end_to_end_time;
  MPI_Reduce(&end_to_end_time, &max_end_to_end_time, 1, MPI_DOUBLE,MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&end_to_end_time, &min_end_to_end_time, 1, MPI_DOUBLE, MPI_MIN, 0,MPI_COMM_WORLD);
  MPI_Reduce(&end_to_end_time, &avg_end_to_end_time, 1, MPI_DOUBLE, MPI_SUM, 0,MPI_COMM_WORLD);
  if (myid == 0){
      std::cout << " MPI end-to-end time  max: " << max_end_to_end_time << " min: " << min_end_to_end_time << " avg: " << avg_end_to_end_time/tot_num_ranks << std::endl;
    }

  // The Pauli string given by:  X_0 . id_1 . Z_2 . Z_3
  // Such observable is defined by the position of the non-trivial Pauli matrices:
  return 0;
}
