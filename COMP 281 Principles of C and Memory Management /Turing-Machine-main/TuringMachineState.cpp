// TuringMachineState.cpp

#include "TuringMachineState.h"

TuringMachineState::TuringMachineState(int currentState, int currentContent,
                                       int nextState, int nextContent,
                                       std::string moveDirection)
        : currentState_(currentState),
          currentContent_(currentContent),
          nextState_(nextState),
          nextContent_(nextContent),
          moveDirection_(moveDirection) {}
          // The constructor, which initializes all the private variables

int TuringMachineState::getCurrentState() const {
    return currentState_;
}

int TuringMachineState::getCurrentContent() const {
    return currentContent_;
}

int TuringMachineState::getNextState() const {
    return nextState_;
}

int TuringMachineState::getNextContent() const {
    return nextContent_;
}

std::string TuringMachineState::getMoveDirection() const {
    return moveDirection_;
}// The getters for the private variables

bool TuringMachineState::operator<(const TuringMachineState& other) const {
    if (currentState_ != other.currentState_) {
        return currentState_ < other.currentState_;
    }
    return currentContent_ < other.currentContent_;
}

bool TuringMachineState::operator>(const TuringMachineState& other) const {
    if (currentState_ != other.currentState_) {
        return currentState_ > other.currentState_;
    }
    return currentContent_ > other.currentContent_;
}

bool TuringMachineState::operator==(const TuringMachineState& other) const {
    return currentState_ == other.currentState_ && currentContent_ == other.currentContent_;
}
// The comparison operators for the TuringMachineState class
std::ostream& operator<<(std::ostream& os, const TuringMachineState& state) {
    os << state.getCurrentState() << " "
       << state.getCurrentContent() << " "
       << state.getNextState() << " "
       << state.getNextContent() << " "
       << state.getMoveDirection();
    return os;
}
// The output operator for the TuringMachineState class
std::istream& operator>>(std::istream& is, TuringMachineState& state) {
    int currentState, currentContent, nextState, nextContent;
    std::string moveDirection;
    is >> currentState >> currentContent >> nextState >> nextContent >> moveDirection;
    state = TuringMachineState(currentState, currentContent, nextState, nextContent, moveDirection);
    return is;
}

