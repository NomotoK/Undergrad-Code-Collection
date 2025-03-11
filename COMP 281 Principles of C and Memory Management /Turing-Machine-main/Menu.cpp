#include <iostream>
#include "MenuSystem.h"
#include "DenseTuringMachine.h"
#include "TuringTape.h"

using namespace std;

int main() {
    int tapeLength = 0;
    cout << "How long should the tape be? ";
    cin >> tapeLength;

    TuringTape tape(tapeLength);
    int currentState = 1;

    MenuSystem menu;

    while (true) {
        menu.menu();
        cout << "Enter Option: ";

        string input;
        getline(cin, input);

        if (input == "1") {
            int maxState, maxContent;
            cout << "What is the maximum state and what is the maximum content? ";
            cin >> maxState >> maxContent;

            DenseTuringMachine dtm(maxState, maxContent);
            tape.setContent(0);
            currentState = 1;
        } else if (input == "2") {
            // Implement create sparse Turing machine later
        } else if (input == "3") {
            int stateToAdd;
            cout << "What state do you wish to add? ";
            cin >> stateToAdd;

            TuringMachineState tms(stateToAdd, 0, 0, 0, "");
            DenseTuringMachine dtm(0, 0);
        } else if (input == "4") {
            // Implement compact Turing machine later
        } else if (input == "5") {
            int numSteps;
            cout << "How many steps do you wish to execute? ";
            cin >> numSteps;

            for (int i = 0; i < numSteps; i++) {
                if (tape.getPosition() < 0 || tape.getPosition() >= tape.getLength()) {
                    cout << "In step " << i << ", the position is " << tape.getPosition() << ", but that is outside the tape." << endl;
                    break;
                }

                int currentStateIndex = dtm.getIndexOfState(currentState, tape.getContent());
                if (currentStateIndex == -1) {
                    cout << "In step " << i << ", there is no Turing machine state with state " << currentState << " and content " << tape.getContent() << "." << endl;
                    break;
                }

                TuringMachineState currentTMS = dtm.getState(currentStateIndex);
                currentState = currentTMS.getNextState();
                tape.setContent(currentTMS.getNextContent());

                if (currentTMS.getMoveDirection() == TuringMachineState::Move::LEFT) {
                    tape.moveLeft();
                } else {
                    tape.moveRight();
                }
            }
        } else if (input == "6") {
            cout << "The current state is " << currentState << ". The current position is " << tape.getPosition() << ".\nThe content of the tape is " << tape.getContent() << ".\nThe states of the Turing machine is ";
            for (int i = 0; i < dtm.getNumStates(); i++) {
                cout << dtm.getState(i).getState() << " ";
            }
            cout << endl;
        } else if (input == "q" || input == "Q") {
            break;
        } else {
            cout << "Invalid input. Please try again." << endl;
        }
    }

    return 0;
}
