<h1>HISVSIM</h1>

Early but promising results in quantum computing have been enabled by the concurrent development of quantum algorithms, devices, and materials. Classical simulation of quantum programs has enabled the design and analysis of algorithms and implementation strategies targeting current and anticipated quantum device architectures. In this work, we present a graph-based approach to achieving efficient quantum circuit simulation. Our approach involves partitioning the graph representation of a given quantum circuit into acyclic sub-graphs/circuits that exhibit better data locality. Simulation of each sub-circuit is organized hierarchically, with the iterative construction and simulation of smaller state vectors, improving overall performance. Also, this partitioning reduces the number of passes through data, improving the total computation time. We present three partitioning strategies and observe that acyclic graph partitioning typically results in the best time-to-solution. In contrast, other strategies reduce the partitioning time at the expense of potentially increased simulation times. Experimental evaluation demonstrates the effectiveness of our approach.

The simulator consists of two components: The circuit partitioning module and the hierarchical state vector simulator. The simulator accepts acyclic partitions of an input circuit and the circuit partitioning module partitions an input circuit, given as in dot format, into acyclic blocks.


to build the wrapper and statevector simulator together

prerequisites:

    - pybind11
    - python3
    - networkx

To build the whole framework inside this directory:

    $ c++ -O3 -funroll-loops -march=native -fomit-frame-pointer -fopenmp -Wall -shared -std=c++17 -lnuma -fPIC `python3 -m pybind11 --includes` py_wrapper.cpp -o svsimulator_py_wrapper`python3-config --extension-suffix`

    For MPI
    $ mpic++ -fopenmp -O3 -funroll-loops -march=native -fomit-frame-pointer -Wall -shared -ffast-math -std=c++17 -lnuma -fPIC `python3 -m pybind11 --includes` py_wrapper.cpp -DSVMPI -o svsimulator_py_wrapper`python3-config --extension-suffix`

This will generate a .so file, such as:
   
    svsimulator_py_wrapper.cpython-37m-x86_64-linux-gnu.so

On MacOS:

    (assume we have brew and pip installed)
    1. install openmp support 
    $ brew install libomp

    2. install pybind11
    $ python3 -m pip install pybind11

    3. compile the shared python module
    $ c++ -O3 -Xpreprocessor -fopenmp -Wall -shared -std=c++17 -lomp -fPIC -undefined dynamic_lookup `python3 -m pybind11 --includes` py_wrapper.cpp -o svsimulator_py_wrapper`python3-config --extension-suffix`    

Then you can copy the qasm_assembler_standalone.py and the .so file to any place to run it. 


to run the simulator

    python qasm_assembler_standalone.py --dot --app bv_n26 --initn 8 --n 8 --alloc 0 (this will generate the dot files and execute the circuit without any parts information)

to run with the parts file

    python qasm_assembler_standalone.py --parts --app bv_n26 --n 8 --alloc 0 --partfile PART_0.txt (assuming the dot files are arealy generated) --slots

    python qasm_assembler_standalone.py --app bv_n19 --initn 8 --n 1 --alloc 0 --parts --partfile bv_n19_part_smart --slots

or, to run the whole process in one shot:

    python qasm_assembler_standalone.py --dot --parts --app bv_n26 --init 8 --n 8 --alloc 0 --partfile PART_0.txt --slots (assuming you already put the the code that generates the parts file inside the qasm_assembler_standalone.py at the indicated location)

run with MPI


`mpirun -n 16 python3 qasm_assembler_standalone.py --app bv_n30 --n 1 --initn 1 --slots --alloc 0 --parts --partfile bv_n30_part_smart --mpi --nlocal 26
`

run with multi-level MPI

 `mpirun -n 16 python3 qasm_assembler_standalone.py --app bv_n30 --n 1 --initn 1 --slots --alloc 0 --parts --partfile directory_to_all_parts_file --mpi --multi --nlocal 26`
 

#Note

Currently the qasm_assembler_standalone.py is set up to run with the QASMBench. So the QASMBench should exist in the same directoy of this python file, and the .so of course.  


This work is a collaboration between Pacific Northwest National Laboratory (PNNL) and TDALab.

================================<br>
If you use HiSVSIM, please cite:

Bo Fang, M. Yusuf Özkaya, Ang Li, Ümit V. Çatalyürek, Sriram Krishnamoorthy,
"Efficient Hierarchical State Vector Simulation of Quantum Circuits via Acyclic Graph Partitioning",
IEEE CLUSTER, 2022  Best paper award.

arXiv: https://arxiv.org/pdf/2205.06973.pdf


