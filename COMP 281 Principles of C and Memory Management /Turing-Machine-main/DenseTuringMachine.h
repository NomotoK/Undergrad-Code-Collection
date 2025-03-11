#ifndef DENSETURINGMACHINE_H
#define DENSETURINGMACHINE_H

#include "TuringMachine.h"
#include "TuringMachineState.h"
#include <vector>

class DenseTuringMachine : public TuringMachine {
public:
    DenseTuringMachine(int x, int y);
    TuringMachineState* find(int x, int y);
    void add(TuringMachineState& s);
    std::vector<TuringMachineState>* getAll();

private:
    std::vector<std::vector<TuringMachineState>> matrix;//use a matrix to store the states
    std::vector<TuringMachineState> all_states;//use a vector to store all the states
    int max_x;
    int max_y;
};

#endif // DENSETURINGMACHINE_H
