#include <Arduino.h>

// ===================== PINS =====================
static const int SENSOR_A0 = A0;
static const int SENSOR_A1 = A1;
static const int SENSOR_A2 = A2;
static const int SENSOR_A3 = A3;
static const int SENSOR_D6 = D6;

// ===================== TIMING =====================
static const unsigned long PRINT_INTERVAL_MS = 100;
unsigned long lastPrint = 0;

// Convert 12-bit ADC (0-4095) to 10-bit (0-1023)
int to1023(int raw) {
  return (raw * 1023) / 4095;
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(SENSOR_A0, INPUT);
  pinMode(SENSOR_A1, INPUT);
  pinMode(SENSOR_A2, INPUT);
  pinMode(SENSOR_A3, INPUT);
  pinMode(SENSOR_D6, INPUT);

  analogReadResolution(12); // ESP32 ADC native-style range: 0-4095

  Serial.println();
  Serial.println("=== Sensor Test ===");
  Serial.println("Format: A0,A1,A2,A3,D6");
  Serial.println("A0-A3 = 0-1023, D6 = 0 or 1");
}

void loop() {
  unsigned long now = millis();
  if (now - lastPrint >= PRINT_INTERVAL_MS) {
    lastPrint = now;

    int a0 = to1023(analogRead(SENSOR_A0));
    int a1 = to1023(analogRead(SENSOR_A1));
    int a2 = to1023(analogRead(SENSOR_A2));
    int a3 = to1023(analogRead(SENSOR_A3));
    int d6 = digitalRead(SENSOR_D6);

    Serial.print(a0);
    Serial.print(",");
    Serial.print(a1);
    Serial.print(",");
    Serial.print(a2);
    Serial.print(",");
    Serial.print(a3);
    Serial.print(",");
    Serial.println(d6);
  }
}