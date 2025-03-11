//
// Created by Xie Hailin on 2023/4/28.
//

#include "DenseTuringMachine.h"
#include <limits>


DenseTuringMachine::DenseTuringMachine(int x, int y) {//when x or y is -1, it means that the value is not set
    if (x == -1) {
        max_x = 1024;
    }
    else {
        max_x = x;
    }

    if (y == -1) {
        max_y = 1024;
    }
    else {
        max_y = y;
    }

    matrix.resize(max_x+1);
    for (int i = 0; i <= max_x; i++) {
        matrix[i].resize(max_y+1, TuringMachineState(-1, -1, -1, -1, " "));
    }
}
//find the state in the matrix, if not found, return nullptr
TuringMachineState* DenseTuringMachine::find(int x, int y) {
    if (x > max_x || y > max_y){
        return nullptr;
    }
    else if (matrix[x][y].getCurrentState() == -1) {
        return nullptr;
    }
    else {
        return &matrix[x][y];
    }
}
//add the state to the matrix
void DenseTuringMachine::add(TuringMachineState& s) {
    int x = s.getCurrentState();
    int y = s.getCurrentContent();
    if (x >= 0 && x <= max_x && y >= 0 && y <= max_y) {
        matrix[x][y] = s;
        all_states.push_back(s);
    }
}
//get all the states in the matrix
std::vector<TuringMachineState>* DenseTuringMachine::getAll() {
    return &all_states;
}
