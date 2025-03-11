// TuringMachineState.h

#ifndef TURINGMACHINESTATE_H_
#define TURINGMACHINESTATE_H_

#include <iostream>
#include <string>


class TuringMachineState {
public:
    TuringMachineState(int currentState, int currentContent, int nextState,
                       int nextContent, std::string moveDirection);

    TuringMachineState();

    int getCurrentState() const;
    int getCurrentContent() const;
    int getNextState() const;
    int getNextContent() const;
    std::string getMoveDirection() const;

    bool operator<(const TuringMachineState& other) const;
    bool operator>(const TuringMachineState& other) const;
    bool operator==(const TuringMachineState& other) const;

    friend std::ostream& operator<<(std::ostream& os, const TuringMachineState& state);
    friend std::istream& operator>>(std::istream& is, TuringMachineState& state);

private:
    int currentState_;
    int currentContent_;
    int nextState_;
    int nextContent_;
    std::string moveDirection_;
};


#endif // TURING_MACHINE_STATE_H