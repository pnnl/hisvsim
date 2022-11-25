#include <stdio.h>
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "state_vector.hpp"
#include "execute.hpp"


namespace py = pybind11;
using namespace SvSim;
using namespace BasicGates;

PYBIND11_MODULE(svsimulator_py_wrapper, m) 
{
    /*
    py::class_<X>(m,"X")
        .def(py::init<>());
    
    py::class_<Y>(m,"Y")
        .def(py::init<>());

    py::class_<Z>(m,"Z")
        .def(py::init<>());
    
    py::class_<I>(m,"I")
        .def(py::init<>());
    
    py::class_<H>(m,"H")
        .def(py::init<>());

    py::class_<S>(m,"S")
        .def(py::init<>());

    py::class_<T>(m,"T")
        .def(py::init<>());

    py::class_<RX>(m,"RX")
        .def(py::init<Type>());
    
    py::class_<RY>(m,"RY")
        .def(py::init<Type>());
    
    py::class_<RZ>(m,"RZ")
        .def(py::init<Type>());
    
    py::class_<RI>(m,"RI")
        .def(py::init<Type>());

    py::class_<R1>(m,"R1")
        .def(py::init<Type>());

    py::class_<RZFrac>(m,"RZFrac")
        .def(py::init<unsigned, unsigned>());

    py::class_<RXFrac>(m,"RXFrac")
        .def(py::init<unsigned, unsigned>());

    py::class_<RYFrac>(m,"RYFrac")
        .def(py::init<unsigned, unsigned>());

    py::class_<RIFrac>(m,"RIFrac")
        .def(py::init<unsigned, unsigned>());

    py::class_<R1Frac>(m,"R1Frac")
        .def(py::init<unsigned, unsigned>());

    py::class_<SWAP>(m,"SWAP")
        .def(py::init<>());
    
    py::class_<C1>(m,"C1")
        .def(py::init<Type, Type, Type, Type>());

    py::class_<C2>(m,"C2")
        .def(py::init<Type, Type, Type, Type, Type, Type, Type, Type, Type, Type, Type, Type, Type, Type, Type, Type>());

    py::class_<U2>(m,"U2")
        .def(py::init<Type, Type>());

    py::class_<U3>(m,"U3")
        .def(py::init<Type, Type, Type>());

    py::class_<RXX>(m,"RXX")
        .def(py::init<Type>());
    
    py::class_<RYY>(m,"RYY")
        .def(py::init<Type>());

    py::class_<RZZ>(m,"RZZ")
        .def(py::init<Type>());

    
    py::class_<SDG>(m,"SDG")
        .def(py::init<>());

    py::class_<TDG>(m,"TDG")
        .def(py::init<>());
        */

    py::class_<AggregateGate_0>(m,"AggregateGate_0")
        .def(py::init<const std::string &>())
        ;

    py::class_<AggregateGate_1>(m,"AggregateGate_1")
        .def(py::init<const std::string &, Type>())
        ;

    py::class_<AggregateGate_2>(m,"AggregateGate_2")
        .def(py::init<const std::string &, Type, Type>())
        ;

    py::class_<AggregateGate_3>(m,"AggregateGate_3")
        .def(py::init<const std::string &, Type, Type, Type>())
        ;
    
    py::class_<AggregateGate_C1>(m,"AggregateGate_C1")
        .def(py::init<const std::string &, Type, Type, Type, Type>())
        ;
    
    py::class_<AggregateGate_C2>(m,"AggregateGate_C2")
        .def(py::init<const std::string &, Type, Type, Type, Type,
                                           Type, Type, Type, Type,
                                           Type, Type, Type, Type,
                                           Type, Type, Type, Type>())
        ;

    py::class_<AggregateGate_Frac>(m,"AggregateGate_Frac")
        .def(py::init<const std::string &, unsigned, unsigned>())
        ;
    
    py::class_<AggregateGateFactory>(m,"AggregateGateFactory")
        .def(py::init<AggregateGate_0 &>())
        .def(py::init<AggregateGate_1 &>())
        .def(py::init<AggregateGate_2 &>())
        .def(py::init<AggregateGate_3 &>())
        .def(py::init<AggregateGate_Frac &>())
        .def(py::init<AggregateGate_C1 &>())
        .def(py::init<AggregateGate_C2 &>())
        ;

    py::class_<StateVector>(m,"StateVector")
        .def(py::init<Qubit,int>())
        .def("qubits",&StateVector::qubits)
        .def("measure",&StateVector::measure)
        .def("state_equal",&StateVector::state_equal)
        .def("dump_state",&StateVector::dump_state_new)
        .def(py::self == py::self)
        ;

    py::class_<BoundGate>(m,"BoundGate")
        .def(py::init<AggregateGateFactory &, const std::vector<Qubit> &, const std::vector<Qubit> &>())
        .def("apply",&BoundGate::apply)
        ;
    
    m.def("gather_and_execute_on", &gather_and_execute_on);
    
    #ifdef SVMPI
        m.def("gather_and_execute_on_mpi", &gather_and_execute_on_mpi);
        m.def("gather_and_execute_multilevel_on_mpi",&gather_and_execute_multilevel_on_mpi);
    #endif

    m.def("obtain_apply_time",&obtain_accumulate_timer);

    m.def("obtain_gate_counter",&obtain_gate_counter);

    m.def("obtain_move_counter",&obtain_move_counter);

    m.def("obtain_gather_time",&obtain_gater_timer);

    m.def("set_num_threads",&set_num_threads);

    m.def("set_init_num_threads",&set_init_num_threads);

    m.def("set_opt_slots",&set_opt_slots);
    
}


