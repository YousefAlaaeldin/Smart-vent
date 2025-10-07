#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <DHT.h>

#define DHTPIN 4
#define DHTTYPE DHT22
DHT dht(DHTPIN, DHTTYPE);

LiquidCrystal_I2C lcd(0x27, 16, 2);

#define IN3 8
#define IN4 9
#define ENB 10

#define POT_PIN A0
#define MQ_PIN A1
int mqThreshold = 400;  // Adjust this value

#define ALERT_LED 11     // LED for gas alert

const unsigned long LED_PULSE_MS = 300;
unsigned long lastRxMillis = 0;
bool ledOn = false;

const uint8_t VOICE_SPEED_PWM[6] = { 0, 100, 140, 175, 210, 255 };
int8_t voiceLevel = 0;
bool stopOverride = false;
bool invertDirFlag = false;

int lastDirection = 0;
unsigned int receivedValue = 0;

void setOutputs(int direction, int pwm) {
  if (direction != lastDirection) {
    analogWrite(ENB, 0);
    delay(1000);
    if (direction > 0) {
      digitalWrite(IN3, HIGH);
      digitalWrite(IN4, LOW);
    } else if (direction < 0) {
      digitalWrite(IN3, LOW);
      digitalWrite(IN4, HIGH);
    } else {
      digitalWrite(IN3, LOW);
      digitalWrite(IN4, LOW);
    }
    lastDirection = direction;
  }
  analogWrite(ENB, pwm);
}

void setup() {
  Serial.begin(9600);
  Wire.begin();
  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.setCursor(0,0); lcd.print("Smart Vent Init");
  delay(1200);
  dht.begin();
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENB, OUTPUT);
  analogWrite(ENB, 0);
  pinMode(MQ_PIN, INPUT);
  pinMode(ALERT_LED, OUTPUT);
  digitalWrite(ALERT_LED, LOW);
}

void loop() {
  if (Serial.available() >= 2) {
    byte highByte = Serial.read();
    byte lowByte  = Serial.read();
    receivedValue = (highByte << 8) | lowByte;
    lastRxMillis = millis();
    if (receivedValue == 0xA190) {
      voiceLevel = 3; stopOverride = false;
    } else if (receivedValue == 0xA191) {
      if (voiceLevel < 5) voiceLevel++;
      stopOverride = false;
    } else if (receivedValue == 0xA192) {
      voiceLevel = 2; stopOverride = false;
    } else if (receivedValue == 0xA193) {
      voiceLevel = 3; stopOverride = false;
    } else if (receivedValue == 0xA194) {
      voiceLevel = 4; stopOverride = false;
    } else if (receivedValue == 0xA195) {
      voiceLevel = 0; stopOverride = true;
    } else if (receivedValue == 0xA196) {
      invertDirFlag = !invertDirFlag;
    }
    receivedValue = 0;
  }


  int potRaw = analogRead(POT_PIN);
  int potPWM = map(potRaw, 0, 1023, 0, 255);
  if (potPWM < 10) potPWM = 0;

  float setTemp = map(potRaw, 0, 1023, 20, 40);
  float currentTemp = dht.readTemperature();
  if (isnan(currentTemp)) {
    lcd.clear();
    lcd.setCursor(0,0); lcd.print("DHT22 ERROR");
    delay(1000);
    return;
  }

  int autoDir = 0;
  if (currentTemp > setTemp + 0.2) autoDir = +1;
  else if (currentTemp < setTemp - 0.2) autoDir = -1;
  else autoDir = 0;

  int finalDir = autoDir;
  if (invertDirFlag && finalDir != 0) finalDir = -finalDir;

  int voicePWM = VOICE_SPEED_PWM[voiceLevel];
  int effectivePWM = 0;
  if (stopOverride) {
    effectivePWM = 0;
  } else {
    effectivePWM = (potPWM > voicePWM) ? potPWM : voicePWM;
  }
  if (finalDir == 0) effectivePWM = 0;

  setOutputs(finalDir, effectivePWM);

  // --- Check MQ sensor ---
  int mqValue = analogRead(MQ_PIN);

  if (mqValue > mqThreshold) {
    digitalWrite(ALERT_LED, HIGH);  // LED ON if gas detected
  } else {
    digitalWrite(ALERT_LED, LOW);   // LED OFF otherwise
  }

  // --- LCD Display ---
  lcd.clear();
  lcd.setCursor(0,0);
  lcd.print("Set:");
  lcd.print(setTemp, 1); lcd.print((char)223); lcd.print("C ");
  lcd.print("PWM:");
  lcd.print(effectivePWM);

  lcd.setCursor(0,1);
  lcd.print("Now:");
  lcd.print(currentTemp, 1); lcd.print((char)223); lcd.print("C");
  delay(500);
}
