const unsigned char PRPIN[15] = {A0,A1,A2,A3,A4,A5,A6,A7};  //압력센서 핀

int prValue1,  //압력센서값 저장하는 변수 선언
    prValue2,
    prValue3,
    prValue4,
    prValue5,
    prValue6,
    prValue7,
    prValue8;
    

void setup() {
  Serial.begin(9600);
}

void loop() {
  prValue1 = analogRead(PRPIN[0]); //순시 센서값 읽어들이기
  prValue2 = analogRead(PRPIN[1]);
  prValue3 = analogRead(PRPIN[2]);
  prValue4 = analogRead(PRPIN[3]);
  prValue5 = analogRead(PRPIN[4]);
  prValue6 = analogRead(PRPIN[5]);
  prValue7 = analogRead(PRPIN[6]);
  prValue8 = analogRead(PRPIN[7]);
  
  delay(210);
  Serial.print('A');     //순시 센서값 보내기
  Serial.println(prValue1);
  Serial.print('B');
  Serial.println(prValue2);
  Serial.print('C');
  Serial.println(prValue3);
  Serial.print('D');
  Serial.println(prValue4);
  
  if((prValue5 > 50 && prValue7 > 50)||(prValue7 > 50 && prValue8 > 50)||(prValue5 > 50 && prValue8 > 50)){//바른자세
    Serial.println('E');     
  }
  else if(prValue8 > 50 && prValue7 <= 50 && prValue6 <= 50 && prValue5 <= 50){//앞 자세
    Serial.println('F'); 
  }
  else if(prValue5> 50 && prValue7 <= 50 && prValue8 <= 50){//뒷 자세
    Serial.println('G');  
  }
  else if(((prValue1 > 0)||//아래 센서만 들어올 때
  (prValue2 > 0)||
  (prValue3 > 0)||
  (prValue4 > 0))&&
  ((prValue5 <= 50)&&
  (prValue6 <= 50)&&
  (prValue7 <= 50)&&
  (prValue8 <= 50))){
    Serial.println('I'); 
  }
  else{
    Serial.println('H');//그 외
  }
}
