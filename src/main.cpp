#include <Arduino.h>

// ===================== PIN CONFIG =====================
static const uint8_t HALL_ANALOG_PINS[4] = {A0, A1, A2, A3};
static const uint8_t HALL_DIGITAL_PIN = D6;

// ===================== TIMING =========================
static const uint32_t SAMPLE_INTERVAL_MS = 10;   // 100 Hz output
static const uint8_t ANALOG_SAMPLES = 4;         // simple averaging

int readAnalogAveraged(uint8_t pin, uint8_t samples)
{ 
    uint32_t total = 0;
    for (uint8_t i = 0; i < samples; i++)
    {
        total += analogRead(pin);
    }
    return total / samples;
}

void setup()
{
    Serial.begin(115200);
    delay(1500);

    analogReadResolution(12); // ESP32 ADC range: 0..4095

    for (uint8_t i = 0; i < 4; i++)
    {
        pinMode(HALL_ANALOG_PINS[i], INPUT);
    }

    pinMode(HALL_DIGITAL_PIN, INPUT);

    Serial.println("m1,m2,m3,m4,m5");
}

void loop()
{
    static uint32_t lastSampleMs = 0;
    uint32_t now = millis();

    if (now - lastSampleMs >= SAMPLE_INTERVAL_MS)
    {
        lastSampleMs = now;

        int m1 = readAnalogAveraged(HALL_ANALOG_PINS[0], ANALOG_SAMPLES);
        int m2 = readAnalogAveraged(HALL_ANALOG_PINS[1], ANALOG_SAMPLES);
        int m3 = readAnalogAveraged(HALL_ANALOG_PINS[2], ANALOG_SAMPLES);
        int m4 = readAnalogAveraged(HALL_ANALOG_PINS[3], ANALOG_SAMPLES);
        int m5 = digitalRead(HALL_DIGITAL_PIN);   // 0 or 1

        Serial.print(m1);
        Serial.print(",");
        Serial.print(m2);
        Serial.print(",");
        Serial.print(m3);
        Serial.print(",");
        Serial.print(m4);
        Serial.print(",");
        Serial.println(m5);
    }
}