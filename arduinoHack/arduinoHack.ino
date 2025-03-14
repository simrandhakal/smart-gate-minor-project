#include <Servo.h>

const int ledPin = 13;   // LED pin
const int servoPin = 9;  // Servo control pin

Servo myservo;  // Create servo object to control a servo

void setup() {
  Serial.begin(9600);  // Set the baud rate to match the Python script
  pinMode(ledPin, OUTPUT);
  myservo.attach(servoPin);
  myservo.write(0);  // Attach the servo to the specified pin
}

int pos; 

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();  // Read the command from serial

    if (command == 's') {
      digitalWrite(ledPin, HIGH);  // Turn on the LED
      Serial.println("LED turned on");
      pos = 0;
      Serial.println(pos);
      myservo.write(pos);  // tell servo to go to position in variable 'pos'
      delay(2000); 
      pos = 180;
      Serial.println(pos);
      myservo.write(pos);  // tell servo to go to position in variable 'pos'
      delay(2000);
      pos = 0;
      Serial.println(pos);
      myservo.write(pos);
      delay(2000);
       
      

//      for (pos = 0; pos <= 90; pos += 1) {  // goes from 0 degrees to 180 degrees
//        // in steps of 1 degree
//        Serial.println(pos);
//        myservo.write(pos);  // tell servo to go to position in variable 'pos'
//        delay(20);           // waits 15 ms for the servo to reach the position
//      }
//      delay(2000);
//      for (pos = 90; pos >= 0; pos -= 1) {  // goes from 180 degrees to 0 degrees
//        myservo.write(pos);                  // tell servo to go to position in variable 'pos'
//        Serial.println(pos);
//        delay(20);                           // waits 15 ms for the servo to reach the position
//      }
      command = 't';
    }
  }
}
