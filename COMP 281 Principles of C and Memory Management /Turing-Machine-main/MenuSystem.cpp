#include "DenseTuringMachine.h"
#include "MenuSystem.h"
#include "TuringTape.h"
#include "TuringMachineState.h"
#include "TuringMachine.h"
#include "SparseTuringMachine.h"
#include <iostream>
#include <string>
#include <memory>
#include <map>
#include <limits>

void MenuSystem::menu() {
    std::string input;//user input
    int tapeLength;
    int stepCount = 0;

    while (true){//check if input is a number
        try{
            tapeLength = std::stoi(input);
            break;
        } catch (std::invalid_argument& e) {
            std::cout << "How long should the tape be?\n";
            std::cin >> input;
        }
    }

    if (tapeLength == -1) {//if input is -1, tape length is 1024
        tapeLength = 1024;
    }

    TuringTape tape(tapeLength);//create tape
    int currentState = 0;
    std::unique_ptr<TuringMachine> turingMachine;//create Turing machine

    while (true) {//menu
        std::cout << "1. Create dense Turing machine\n"
                     "2. Create sparse Turing machine\n"
                     "3. Add state to Turing machine\n"
                     "4. Compact Turing machine\n"
                     "5. Execute Turing machine\n"
                     "6. Output current information\n"
                     "Write q or Q to quit\n"
                     "Enter Option \n";
        std::string option;
        std::cin >> option;//get user input

        if (option == "q" || option == "Q") {//quit
            break;

        } else if (option == "1") {//create dense Turing machine
            std::string inputMaxState, inputMaxContent;
            int maxState, maxContent;

            while (true){//check if input is a number
                try{
                    maxState = std::stoi(inputMaxState);
                    maxContent = std::stoi(inputMaxContent);
                    break;
                } catch (std::invalid_argument& e) {
                    std::cout << "What is the maximum state and what is the maximum content?\n";
                    std::cin >> inputMaxState>> inputMaxContent;
                }
            }
            turingMachine.reset(new DenseTuringMachine(maxState, maxContent));

        } else if (option == "2") {//create sparse Turing machine
            SparseTuringMachine *sparseTuringMachine = new SparseTuringMachine();
            turingMachine.reset(sparseTuringMachine);


        } else if (option == "3") {//add state to Turing machine
            TuringMachineState state(0, 0, 0, 0, "");
            std::cout << "What state do you wish to add?\n";
            std::cin >> state;
            turingMachine->add(state);


        } else if (option == "4") {//compact Turing machine
            std::map <int,int> stateMap;//map to store states
            std::map <int,int> contentMap;
            std::set<int>allStates;
            std::set<int>allContents;
            int stateCount = 0;//count of states
            int contentCount = 0;



            for(auto& s : *turingMachine->getAll()) {//get all states and contents

                int currentState = s.getCurrentState();
                int currentContent = s.getCurrentContent();
                int nextState = s.getNextState();
                int nextContent = s.getNextContent();
                //update state and content count

                allStates.insert(currentState);
                allStates.insert(nextState);
                allContents.insert(currentContent);
                allContents.insert(nextContent);
                //update state and content map
            }

               for (const auto& s : allStates) {//map states and contents
                    stateMap[s] = stateCount++;
                }

                for (const auto& s : allContents) {//map states and contents
                    contentMap[s] = contentCount++;
               }


                DenseTuringMachine newTuringMachine(stateCount-1, contentCount-1);//create new Turing machine

                for (auto& s : *turingMachine->getAll()) {
                    int currentState = stateMap[s.getCurrentState()];
                    int currentContent = contentMap[s.getCurrentContent()];
                    int nextState = stateMap[s.getNextState()];
                    int nextContent = contentMap[s.getNextContent()];
                    std::string moveDirection = s.getMoveDirection();

                    TuringMachineState newState(currentState, currentContent, nextState, nextContent, moveDirection);
                    newTuringMachine.add(newState);//copy states to new Turing machine

                }
                turingMachine.reset(new DenseTuringMachine(stateCount-1, contentCount-1));
                for (auto& s : *newTuringMachine.getAll()) {
                    turingMachine->add(s);//add states to Turing machine
                }


        } else if (option == "5") {//execute Turing machine
            int steps;
            std::string stepsInput;
            while (true){//check if input is a number
                try{
                    steps = std::stoi(stepsInput);
                    if(steps < 0){
                        steps = abs(steps);
                    }
                    break;
                } catch (std::invalid_argument& e) {
                    std::cout << "How many steps do you wish to execute?\n";
                    std::cin >> stepsInput;
                }
            }

            for (int i = 0; i < steps; ++i) {//execute Turing machine
                stepCount++;
                if (tape.getPosition() > tape.getHighestPosition() || tape.getPosition() < 0) {
                    std::cout << "In step " << stepCount << ", the position is " << tape.getPosition()
                              << ", but that is outside the tape." << std::endl;
                    break;
                }
                int content = tape.getContent();
                TuringMachineState *state = turingMachine->find(currentState, content);
                if (state == nullptr) {//check if state exists
                    std::cout << "In step " << i << ", there is no Turing machine state with state " << currentState
                    << " and content " << content << std::endl;
                    break;

                }

                currentState = state->getNextState();
                tape.setContent(state->getNextContent());//update tape content

                if (state->getMoveDirection() == "->") {
                    tape.moveRight();
                } else if (state->getMoveDirection() == "<-"){
                    tape.moveLeft();
                }//move tape

            }


        } else if (option == "6") {//output current information
            std::cout << "The current state is " << currentState << ".\n";
            std::cout << "The current position is " << tape.getPosition() << ".\n";
            std::cout << "The content of the tape is " << tape << ".\n";
            std::cout << "The Turing machine has states: ";
            for (const auto &state : *turingMachine->getAll()) {//output states
                std::cout << "<" << state << ">" << " ";
            }
            std::cout << std::endl;

        } else {
            std::cout << "Invalid option. Please try again.\n";
        }
    }
}