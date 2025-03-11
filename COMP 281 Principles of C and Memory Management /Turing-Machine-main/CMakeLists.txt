cmake_minimum_required(VERSION 3.21)
project(COMP282)

set(CMAKE_CXX_STANDARD 11)


add_executable(COMP282
        Main.cpp
        DenseTuringMachine.cpp
        DenseTuringMachine.h
        TuringMachine.h
        TuringMachineState.cpp
        TuringMachineState.h
        TuringTape.cpp
        TuringTape.h
        MenuSystem.cpp
        MenuSystem.h
        SparseTuringMachine.h
        SparseTuringMachine.cpp

        )
