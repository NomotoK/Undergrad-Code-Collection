package uk.ac.liv.comp201;

import static uk.ac.liv.comp201.ResponseCode.*;

public class Authenticator {
	private Card card;
	private String passcodeFire;
	private String passcodeBurglary;


	public Authenticator(Card card) {
		this.card=card;
	}
	
	public ResponseCode checkFireCode(String passCodeFire) throws CardException{
		passCodeFire = passcodeFire;
		ResponseCode returnValue=OK;
		int status = card.getCardStatus();
		int remainChances = 3-card.getWrongFirePassword();
		int wrongCode = 0;

		if (status!=2&&status!=3) {
			if (passCodeFire.length() < 10 || passCodeFire.length() > 14) {
				return INVALID_FIRE_CODE;
			}

			for (int idx = 0; idx < passCodeFire.length(); idx++) {
				if (!Character.isAlphabetic(passCodeFire.charAt(idx)) && !Character.isDigit(passCodeFire.charAt(idx))) {
					return INVALID_FIRE_CODE;
				}
			}
		}

		if (status == 2) {
			returnValue = CARD_LOCKED;
		}
		if (status == 3){
//			returnValue = CARD_STATUS_BAD;
			throw new CardException(returnValue = CARD_STATUS_BAD);
		}

		else if (!card.getCardFireCode().equals(passCodeFire)) {
			card.setWrongFirePassword(card.getWrongFirePassword() + 1);
			if (card.getWrongFirePassword() == 3) {
				card.setCardStatus(2);
				card.setWrongFirePassword(0);
				returnValue = CARD_LOCKED;
			}
			else {
				returnValue = BAD_FIRE_CODE;
			}
		}
		if (card.getCardFireCode().equals(passCodeFire)&&status!=2&&status!=3) {
			card.setWrongFirePassword(0);
		}
		// TO DO
		// 1. Add code to validate fire code
		// 2. Code to check fire code is correct for card
		// 3. Code to return appropriate response
		// 4. Add code to lock card, if wrong fire code is
		// entered wrong 3 times in sequence, lockout works
		// independently for the two codes

		return(returnValue);
	}
	
	public ResponseCode checkBurglaryCode(String passCodeBurglary) throws CardException {
		passCodeBurglary = passcodeBurglary;
		ResponseCode returnValue=OK;
		int status = card.getCardStatus();

		if (status!=2&&status!=3) {
			if (passCodeBurglary.length() < 8 || passCodeBurglary.length() > 10) {
				return INVALID_BURGLARY_CODE;
			}


			for (int idx = 0; idx < passCodeBurglary.length(); idx++) {
				if (!Character.isDigit(passCodeBurglary.charAt(idx))) {
					return INVALID_BURGLARY_CODE;
				}
			}
		}

		if (status == 2) {
			returnValue = CARD_LOCKED;
		}
		if (status == 3){
//			returnValue = CARD_STATUS_BAD;
			throw new CardException(returnValue = CARD_STATUS_BAD);
		}
		else if (!card.getCardBurlaryCode().equals(passCodeBurglary)){
			card.setWrongBurglaryPassword(card.getWrongBurglaryPassword() + 1);

			if(card.getWrongBurglaryPassword() == 3){
				card.setCardStatus(2);
				card.setWrongBurglaryPassword(0);
				returnValue = CARD_LOCKED;
			}else{
				returnValue = BAD_BURGLARY_CODE;
			}
			if(card.getCardBurlaryCode().equals(passcodeFire)&&status!=2&&status!=3) {
				card.setWrongBurglaryPassword(0);
			}
		}

		// TO DO
		// 1. Add code to validate burglary code
		// 2. Code to check burglary code is correct for card
		// 3. Code to return appropriate response
		// 4. Add code to lock card, if wrong code is
		// entered wrong 3 times in sequence, lockout works
		// independently for the two codes


		return(returnValue);
	}


	

}
