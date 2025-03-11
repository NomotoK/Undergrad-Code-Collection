//
// Created by Xie Hailin on 2023/5/1.
//

#include "SparseTuringMachine.h"

SparseTuringMachine::SparseTuringMachine() {
    // The constructor does not need to do anything in this case
}

TuringMachineState* SparseTuringMachine::find(int x, int y) {
    for (auto& s : states) {
        if (s.getCurrentState() == x && s.getCurrentContent() == y) {
            return &s;
        }
    }
    return nullptr;
    //Find the state with the given current state and current content
}

void SparseTuringMachine::add(TuringMachineState &s) {
    states.push_back(s);
}

std::vector<TuringMachineState>* SparseTuringMachine::getAll() {
    return &states;
}
