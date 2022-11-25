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
psi.ApplyHadamard(0);
psi.ApplyRotationY(1,2.2722018);
psi.ApplyRotationZ(1,2.946554);
psi.ApplyRotationY(2,1.6664782);
psi.ApplyRotationZ(2,0.75798724);
psi.ApplyRotationY(3,6.1702624);
psi.ApplyRotationZ(3,4.431635);
psi.ApplyRotationY(4,2.1202105);
psi.ApplyRotationZ(4,1.4456975);
psi.ApplyRotationY(5,5.2889287);
psi.ApplyRotationZ(5,3.8998312);
psi.ApplyRotationY(6,1.7827599);
psi.ApplyRotationZ(6,0.63218444);
psi.ApplyRotationY(7,5.1051847);
psi.ApplyRotationZ(7,5.1521567);
psi.ApplyRotationY(8,3.4621905);
psi.ApplyRotationZ(8,5.4486691);
psi.ApplyRotationY(9,5.5182397);
psi.ApplyRotationZ(9,5.4450935);
psi.ApplyRotationY(10,1.4311348);
psi.ApplyRotationZ(10,1.8110405);
psi.ApplyRotationY(11,1.4475826);
psi.ApplyRotationZ(11,4.1296311);
psi.ApplyRotationY(12,5.427023);
psi.ApplyRotationZ(12,1.3386555);
psi.ApplyRotationY(13,2.7936794);
psi.ApplyRotationZ(13,4.5079869);
psi.ApplyRotationY(14,1.1303507);
psi.ApplyRotationZ(14,1.0047356);
psi.ApplyRotationY(15,2.8550601);
psi.ApplyRotationZ(15,5.8735908);
psi.ApplyRotationX(1,1.5707963267948966);
psi.ApplyRotationX(2,1.5707963267948966);
psi.ApplyCPauliX(1,2);
psi.ApplyRotationZ(2,1.3777652);
psi.ApplyCPauliX(1,2);
psi.ApplyRotationX(1,-1.5707963267948966);
psi.ApplyRotationX(2,-1.5707963267948966);
psi.ApplyCPauliX(1,2);
psi.ApplyRotationZ(2,1.868311);
psi.ApplyCPauliX(1,2);
psi.ApplyRotationX(2,1.5707963267948966);
psi.ApplyRotationX(3,1.5707963267948966);
psi.ApplyCPauliX(2,3);
psi.ApplyRotationZ(3,3.5674688);
psi.ApplyCPauliX(2,3);
psi.ApplyRotationX(2,-1.5707963267948966);
psi.ApplyRotationX(3,-1.5707963267948966);
psi.ApplyCPauliX(2,3);
psi.ApplyRotationZ(3,6.1748609);
psi.ApplyCPauliX(2,3);
psi.ApplyRotationX(3,1.5707963267948966);
psi.ApplyRotationX(4,1.5707963267948966);
psi.ApplyCPauliX(3,4);
psi.ApplyRotationZ(4,5.2212638);
psi.ApplyCPauliX(3,4);
psi.ApplyRotationX(3,-1.5707963267948966);
psi.ApplyRotationX(4,-1.5707963267948966);
psi.ApplyCPauliX(3,4);
psi.ApplyRotationZ(4,2.0999227);
psi.ApplyCPauliX(3,4);
psi.ApplyRotationX(4,1.5707963267948966);
psi.ApplyRotationX(5,1.5707963267948966);
psi.ApplyCPauliX(4,5);
psi.ApplyRotationZ(5,3.0021756);
psi.ApplyCPauliX(4,5);
psi.ApplyRotationX(4,-1.5707963267948966);
psi.ApplyRotationX(5,-1.5707963267948966);
psi.ApplyCPauliX(4,5);
psi.ApplyRotationZ(5,2.6500381);
psi.ApplyCPauliX(4,5);
psi.ApplyRotationX(5,1.5707963267948966);
psi.ApplyRotationX(6,1.5707963267948966);
psi.ApplyCPauliX(5,6);
psi.ApplyRotationZ(6,1.9417036);
psi.ApplyCPauliX(5,6);
psi.ApplyRotationX(5,-1.5707963267948966);
psi.ApplyRotationX(6,-1.5707963267948966);
psi.ApplyCPauliX(5,6);
psi.ApplyRotationZ(6,1.4210576);
psi.ApplyCPauliX(5,6);
psi.ApplyRotationX(6,1.5707963267948966);
psi.ApplyRotationX(7,1.5707963267948966);
psi.ApplyCPauliX(6,7);
psi.ApplyRotationZ(7,2.6428403);
psi.ApplyCPauliX(6,7);
psi.ApplyRotationX(6,-1.5707963267948966);
psi.ApplyRotationX(7,-1.5707963267948966);
psi.ApplyCPauliX(6,7);
psi.ApplyRotationZ(7,1.5672463);
psi.ApplyCPauliX(6,7);
psi.ApplyRotationX(7,1.5707963267948966);
psi.ApplyRotationX(8,1.5707963267948966);
psi.ApplyCPauliX(7,8);
psi.ApplyRotationZ(8,1.5485984);
psi.ApplyCPauliX(7,8);
psi.ApplyRotationX(7,-1.5707963267948966);
psi.ApplyRotationX(8,-1.5707963267948966);
psi.ApplyCPauliX(7,8);
psi.ApplyRotationZ(8,1.5304351);
psi.ApplyCPauliX(7,8);
psi.ApplyRotationX(8,1.5707963267948966);
psi.ApplyRotationX(9,1.5707963267948966);
psi.ApplyCPauliX(8,9);
psi.ApplyRotationZ(9,5.1342529);
psi.ApplyCPauliX(8,9);
psi.ApplyRotationX(8,-1.5707963267948966);
psi.ApplyRotationX(9,-1.5707963267948966);
psi.ApplyCPauliX(8,9);
psi.ApplyRotationZ(9,1.4695349);
psi.ApplyCPauliX(8,9);
psi.ApplyRotationX(9,1.5707963267948966);
psi.ApplyRotationX(10,1.5707963267948966);
psi.ApplyCPauliX(9,10);
psi.ApplyRotationZ(10,4.1114547);
psi.ApplyCPauliX(9,10);
psi.ApplyRotationX(9,-1.5707963267948966);
psi.ApplyRotationX(10,-1.5707963267948966);
psi.ApplyCPauliX(9,10);
psi.ApplyRotationZ(10,4.7313494);
psi.ApplyCPauliX(9,10);
psi.ApplyRotationX(10,1.5707963267948966);
psi.ApplyRotationX(11,1.5707963267948966);
psi.ApplyCPauliX(10,11);
psi.ApplyRotationZ(11,1.2984641);
psi.ApplyCPauliX(10,11);
psi.ApplyRotationX(10,-1.5707963267948966);
psi.ApplyRotationX(11,-1.5707963267948966);
psi.ApplyCPauliX(10,11);
psi.ApplyRotationZ(11,3.5758165);
psi.ApplyCPauliX(10,11);
psi.ApplyRotationX(11,1.5707963267948966);
psi.ApplyRotationX(12,1.5707963267948966);
psi.ApplyCPauliX(11,12);
psi.ApplyRotationZ(12,1.5208288);
psi.ApplyCPauliX(11,12);
psi.ApplyRotationX(11,-1.5707963267948966);
psi.ApplyRotationX(12,-1.5707963267948966);
psi.ApplyCPauliX(11,12);
psi.ApplyRotationZ(12,5.2702693);
psi.ApplyCPauliX(11,12);
psi.ApplyRotationX(12,1.5707963267948966);
psi.ApplyRotationX(13,1.5707963267948966);
psi.ApplyCPauliX(12,13);
psi.ApplyRotationZ(13,0.77874391);
psi.ApplyCPauliX(12,13);
psi.ApplyRotationX(12,-1.5707963267948966);
psi.ApplyRotationX(13,-1.5707963267948966);
psi.ApplyCPauliX(12,13);
psi.ApplyRotationZ(13,0.40170729);
psi.ApplyCPauliX(12,13);
psi.ApplyRotationX(13,1.5707963267948966);
psi.ApplyRotationX(14,1.5707963267948966);
psi.ApplyCPauliX(13,14);
psi.ApplyRotationZ(14,3.9742007);
psi.ApplyCPauliX(13,14);
psi.ApplyRotationX(13,-1.5707963267948966);
psi.ApplyRotationX(14,-1.5707963267948966);
psi.ApplyCPauliX(13,14);
psi.ApplyRotationZ(14,3.8123714);
psi.ApplyCPauliX(13,14);
psi.ApplyRotationX(14,1.5707963267948966);
psi.ApplyRotationX(15,1.5707963267948966);
psi.ApplyCPauliX(14,15);
psi.ApplyRotationZ(15,0.85338524);
psi.ApplyCPauliX(14,15);
psi.ApplyRotationX(14,-1.5707963267948966);
psi.ApplyRotationX(15,-1.5707963267948966);
psi.ApplyCPauliX(14,15);
psi.ApplyRotationZ(15,2.5013908);
psi.ApplyCPauliX(14,15);
psi.ApplyRotationX(1,1.5707963267948966);
psi.ApplyRotationX(2,1.5707963267948966);
psi.ApplyCPauliX(1,2);
psi.ApplyRotationZ(2,0.095459523);
psi.ApplyCPauliX(1,2);
psi.ApplyRotationX(1,-1.5707963267948966);
psi.ApplyRotationX(2,-1.5707963267948966);
psi.ApplyCPauliX(1,2);
psi.ApplyRotationZ(2,5.364397);
psi.ApplyCPauliX(1,2);
psi.ApplyRotationX(2,1.5707963267948966);
psi.ApplyRotationX(3,1.5707963267948966);
psi.ApplyCPauliX(2,3);
psi.ApplyRotationZ(3,3.5723083);
psi.ApplyCPauliX(2,3);
psi.ApplyRotationX(2,-1.5707963267948966);
psi.ApplyRotationX(3,-1.5707963267948966);
psi.ApplyCPauliX(2,3);
psi.ApplyRotationZ(3,4.9577729);
psi.ApplyCPauliX(2,3);
psi.ApplyRotationX(3,1.5707963267948966);
psi.ApplyRotationX(4,1.5707963267948966);
psi.ApplyCPauliX(3,4);
psi.ApplyRotationZ(4,4.5135882);
psi.ApplyCPauliX(3,4);
psi.ApplyRotationX(3,-1.5707963267948966);
psi.ApplyRotationX(4,-1.5707963267948966);
psi.ApplyCPauliX(3,4);
psi.ApplyRotationZ(4,3.396199);
psi.ApplyCPauliX(3,4);
psi.ApplyRotationX(4,1.5707963267948966);
psi.ApplyRotationX(5,1.5707963267948966);
psi.ApplyCPauliX(4,5);
psi.ApplyRotationZ(5,4.5989047);
psi.ApplyCPauliX(4,5);
psi.ApplyRotationX(4,-1.5707963267948966);
psi.ApplyRotationX(5,-1.5707963267948966);
psi.ApplyCPauliX(4,5);
psi.ApplyRotationZ(5,0.98457203);
psi.ApplyCPauliX(4,5);
psi.ApplyRotationX(5,1.5707963267948966);
psi.ApplyRotationX(6,1.5707963267948966);
psi.ApplyCPauliX(5,6);
psi.ApplyRotationZ(6,2.8489591);
psi.ApplyCPauliX(5,6);
psi.ApplyRotationX(5,-1.5707963267948966);
psi.ApplyRotationX(6,-1.5707963267948966);
psi.ApplyCPauliX(5,6);
psi.ApplyRotationZ(6,3.3154415);
psi.ApplyCPauliX(5,6);
psi.ApplyRotationX(6,1.5707963267948966);
psi.ApplyRotationX(7,1.5707963267948966);
psi.ApplyCPauliX(6,7);
psi.ApplyRotationZ(7,4.536065);
psi.ApplyCPauliX(6,7);
psi.ApplyRotationX(6,-1.5707963267948966);
psi.ApplyRotationX(7,-1.5707963267948966);
psi.ApplyCPauliX(6,7);
psi.ApplyRotationZ(7,4.3995337);
psi.ApplyCPauliX(6,7);
psi.ApplyRotationX(7,1.5707963267948966);
psi.ApplyRotationX(8,1.5707963267948966);
psi.ApplyCPauliX(7,8);
psi.ApplyRotationZ(8,1.5762544);
psi.ApplyCPauliX(7,8);
psi.ApplyRotationX(7,-1.5707963267948966);
psi.ApplyRotationX(8,-1.5707963267948966);
psi.ApplyCPauliX(7,8);
psi.ApplyRotationZ(8,0.80807723);
psi.ApplyCPauliX(7,8);
psi.ApplyRotationX(8,1.5707963267948966);
psi.ApplyRotationX(9,1.5707963267948966);
psi.ApplyCPauliX(8,9);
psi.ApplyRotationZ(9,1.8260727);
psi.ApplyCPauliX(8,9);
psi.ApplyRotationX(8,-1.5707963267948966);
psi.ApplyRotationX(9,-1.5707963267948966);
psi.ApplyCPauliX(8,9);
psi.ApplyRotationZ(9,5.8272508);
psi.ApplyCPauliX(8,9);
psi.ApplyRotationX(9,1.5707963267948966);
psi.ApplyRotationX(10,1.5707963267948966);
psi.ApplyCPauliX(9,10);
psi.ApplyRotationZ(10,5.0571281);
psi.ApplyCPauliX(9,10);
psi.ApplyRotationX(9,-1.5707963267948966);
psi.ApplyRotationX(10,-1.5707963267948966);
psi.ApplyCPauliX(9,10);
psi.ApplyRotationZ(10,5.5719428);
psi.ApplyCPauliX(9,10);
psi.ApplyRotationX(10,1.5707963267948966);
psi.ApplyRotationX(11,1.5707963267948966);
psi.ApplyCPauliX(10,11);
psi.ApplyRotationZ(11,3.0627477);
psi.ApplyCPauliX(10,11);
psi.ApplyRotationX(10,-1.5707963267948966);
psi.ApplyRotationX(11,-1.5707963267948966);
psi.ApplyCPauliX(10,11);
psi.ApplyRotationZ(11,5.4194142);
psi.ApplyCPauliX(10,11);
psi.ApplyRotationX(11,1.5707963267948966);
psi.ApplyRotationX(12,1.5707963267948966);
psi.ApplyCPauliX(11,12);
psi.ApplyRotationZ(12,4.0336232);
psi.ApplyCPauliX(11,12);
psi.ApplyRotationX(11,-1.5707963267948966);
psi.ApplyRotationX(12,-1.5707963267948966);
psi.ApplyCPauliX(11,12);
psi.ApplyRotationZ(12,0.71621777);
psi.ApplyCPauliX(11,12);
psi.ApplyRotationX(12,1.5707963267948966);
psi.ApplyRotationX(13,1.5707963267948966);
psi.ApplyCPauliX(12,13);
psi.ApplyRotationZ(13,3.3558793);
psi.ApplyCPauliX(12,13);
psi.ApplyRotationX(12,-1.5707963267948966);
psi.ApplyRotationX(13,-1.5707963267948966);
psi.ApplyCPauliX(12,13);
psi.ApplyRotationZ(13,0.58545119);
psi.ApplyCPauliX(12,13);
psi.ApplyRotationX(13,1.5707963267948966);
psi.ApplyRotationX(14,1.5707963267948966);
psi.ApplyCPauliX(13,14);
psi.ApplyRotationZ(14,6.1311826);
psi.ApplyCPauliX(13,14);
psi.ApplyRotationX(13,-1.5707963267948966);
psi.ApplyRotationX(14,-1.5707963267948966);
psi.ApplyCPauliX(13,14);
psi.ApplyRotationZ(14,3.3637331);
psi.ApplyCPauliX(13,14);
psi.ApplyRotationX(14,1.5707963267948966);
psi.ApplyRotationX(15,1.5707963267948966);
psi.ApplyCPauliX(14,15);
psi.ApplyRotationZ(15,5.2892379);
psi.ApplyCPauliX(14,15);
psi.ApplyRotationX(14,-1.5707963267948966);
psi.ApplyRotationX(15,-1.5707963267948966);
psi.ApplyCPauliX(14,15);
psi.ApplyRotationZ(15,4.0941267);
psi.ApplyCPauliX(14,15);
psi.ApplyRotationY(16,-0.10062302);
psi.ApplyRotationZ(16,-1.3893113);
psi.ApplyRotationY(17,0.71168662);
psi.ApplyRotationZ(17,-2.0348454);
psi.ApplyRotationY(18,3.118147);
psi.ApplyRotationZ(18,-2.9259188);
psi.ApplyRotationY(19,-0.91422507);
psi.ApplyRotationZ(19,-0.37558816);
psi.ApplyRotationY(20,0.29787287);
psi.ApplyRotationZ(20,-2.3392824);
psi.ApplyRotationY(21,-0.9488212);
psi.ApplyRotationZ(21,0.084427821);
psi.ApplyRotationY(22,-0.23176578);
psi.ApplyRotationZ(22,2.1492547);
psi.ApplyRotationY(23,2.0223901);
psi.ApplyRotationZ(23,2.5291934);
psi.ApplyRotationY(24,1.0959072);
psi.ApplyRotationZ(24,2.0036062);
psi.ApplyRotationY(25,2.3662701);
psi.ApplyRotationZ(25,-0.14481619);
psi.ApplyRotationY(26,-2.4070959);
psi.ApplyRotationZ(26,-2.9426629);
psi.ApplyRotationY(27,-2.0393492);
psi.ApplyRotationZ(27,1.3806717);
psi.ApplyRotationY(28,-0.10186216);
psi.ApplyRotationZ(28,0.091733181);
psi.ApplyRotationY(29,-1.7765786);
psi.ApplyRotationZ(29,1.3794109);
psi.ApplyRotationY(30,-0.17487479);
psi.ApplyRotationZ(30,0.96949875);
psi.ApplyCPauliX(16,1);
psi.ApplyToffoli(0,1,16);
psi.ApplyCPauliX(16,1);
psi.ApplyCPauliX(17,2);
psi.ApplyToffoli(0,2,17);
psi.ApplyCPauliX(17,2);
psi.ApplyCPauliX(18,3);
psi.ApplyToffoli(0,3,18);
psi.ApplyCPauliX(18,3);
psi.ApplyCPauliX(19,4);
psi.ApplyToffoli(0,4,19);
psi.ApplyCPauliX(19,4);
psi.ApplyCPauliX(20,5);
psi.ApplyToffoli(0,5,20);
psi.ApplyCPauliX(20,5);
psi.ApplyCPauliX(21,6);
psi.ApplyToffoli(0,6,21);
psi.ApplyCPauliX(21,6);
psi.ApplyCPauliX(22,7);
psi.ApplyToffoli(0,7,22);
psi.ApplyCPauliX(22,7);
psi.ApplyCPauliX(23,8);
psi.ApplyToffoli(0,8,23);
psi.ApplyCPauliX(23,8);
psi.ApplyCPauliX(24,9);
psi.ApplyToffoli(0,9,24);
psi.ApplyCPauliX(24,9);
psi.ApplyCPauliX(25,10);
psi.ApplyToffoli(0,10,25);
psi.ApplyCPauliX(25,10);
psi.ApplyCPauliX(26,11);
psi.ApplyToffoli(0,11,26);
psi.ApplyCPauliX(26,11);
psi.ApplyCPauliX(27,12);
psi.ApplyToffoli(0,12,27);
psi.ApplyCPauliX(27,12);
psi.ApplyCPauliX(28,13);
psi.ApplyToffoli(0,13,28);
psi.ApplyCPauliX(28,13);
psi.ApplyCPauliX(29,14);
psi.ApplyToffoli(0,14,29);
psi.ApplyCPauliX(29,14);
psi.ApplyCPauliX(30,15);
psi.ApplyToffoli(0,15,30);
psi.ApplyCPauliX(30,15);
psi.ApplyHadamard(0);
    

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
