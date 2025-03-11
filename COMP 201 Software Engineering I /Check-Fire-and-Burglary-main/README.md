# Check-Fire-and-Burglary
COMP201 CW1.2

## Card handling
All the code you will implement is done as part of files `Authenticator.java` and
`Card.java`, you will need to modify both these files. It is very important you do
NOT modify the public interface of these Java classes.
Look at the comments in the source files for information on what has to be
done. Everywhere there is a `TO DO` comment, please complete the code as
requested.

## checkFireCode
The behaviour should be as follows, when calling `checkFireCode()`. If the card
is locked, it will return `CARD_LOCKED_FIRE`. If the code is correct, the method
returns the OK status. If the code is invalid the method returns
`INVALID_FIRE_CODE`. If the code is incorrect, the method returns
`BAD_FIRE_CODE`. Getting the code wrong 3 times will lock the card and return
`CARD_LOCKED_FIRE`.

## checkBurglaryCode
The behaviour should be as follows, when calling `checkBurglaryCode()`. If the
card is locked, it will return `CARD_LOCKED_BURGLAR_ALARM`. If the code
is correct, the method returns the OK status. If the code is invalid the method
returns `INVALID_BURGLARY_CODE`. If the code is incorrect, the method
returns `BAD_BURGLARY_CODE` . Getting the code wrong 3 times will lock
the card and return `CARD_LOCKED_BURGLAR_ALARM`.

For this functionality you will have to had behaviour to the card simulator which
allows it to count the number of wrong code counts and save them persistently.
