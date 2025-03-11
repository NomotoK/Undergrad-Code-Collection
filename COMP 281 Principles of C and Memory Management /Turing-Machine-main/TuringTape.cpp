#include "TuringTape.h"
#include <limits>

TuringTape::TuringTape(int n) : tape(), currentPosition(0), highestPosition(0) {
    if (n != -1) {// -1 means no size specified
        tape.resize(n, 0);
    } else {
        tape.resize(1024, 0);
    }
}

bool TuringTape::moveRight() {// return false if the tape is full
    currentPosition++;
    if (currentPosition > tape.size() - 1) {
        tape.push_back(0);
    }
    if (currentPosition > highestPosition) {// update highest position
        highestPosition = currentPosition;
    }
    return true;
}

bool TuringTape::moveLeft() {// return false if the tape is empty
    currentPosition--;
    if (currentPosition < 0) {
        return false;
    }
    return true;
}

int TuringTape::getContent() {// return 0 if the tape is empty
    if (currentPosition >= 0 && currentPosition < tape.size()) {
        return tape[currentPosition];
    }
    else {
        return 0;
    }
}

void TuringTape::setContent(int c) {
    if (currentPosition >= 0 && currentPosition < tape.size()) {
        tape[currentPosition] = c;
    }
}

int TuringTape::getPosition() {
    return currentPosition;
}

int TuringTape::getHighestPosition() {
    return highestPosition;
}

std::ostream& operator<<(std::ostream& out, const TuringTape& s) {// print the tape
    for (int i = 0; i <= s.highestPosition; ++i) {
        out << s.tape[i];
    }
    return out;
}
