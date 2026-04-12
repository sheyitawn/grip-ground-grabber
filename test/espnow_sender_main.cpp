#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>

// ===================== USER CONFIG =====================
static const uint8_t ESPNOW_CHANNEL = 1;
static const bool PRINT_DEBUG_TO_SERIAL = true;

// ===================== LED CONFIG =====================
#ifndef LED_BUILTIN
#define LED_BUILTIN 21
#endif
static const uint8_t STATUS_LED_PIN = LED_BUILTIN;
static const bool LED_ACTIVE_HIGH = true;
static const uint32_t TX_LED_ON_MS = 10;
static uint32_t ledOffAtMs = 0;

// ===================== SENSOR CONFIG =====================
static const uint8_t ANALOG_HALL_PINS[4] = {A0, A1, A2, A3};
static const uint8_t DIGITAL_HALL_PIN = D6;

static const uint8_t ANALOG_COUNT = 4;
static const uint8_t FINGER_COUNT = 5;

static const uint32_t BASELINE_MS = 1500;
static const uint16_t SAMPLE_HZ = 200;
static const uint32_t SAMPLE_PERIOD_US = 1000000UL / SAMPLE_HZ;

// ESP-NOW send rate. 50-100 Hz is usually plenty for glove data.
static const uint16_t TX_HZ = 80;
static const uint32_t TX_PERIOD_US = 1000000UL / TX_HZ;

static const float SENSOR_ALPHA = 0.35f;
static const int MAG_DEADBAND = 1;
static const int DIGITAL_ACTIVE_VALUE = 1000;
static const int DIGITAL_INACTIVE_VALUE = 0;

// ===================== PACKET =====================
struct __attribute__((packed)) HallPacket {
  uint32_t seq;
  uint32_t sender_ms;
  uint16_t sample_hz;
  uint16_t tx_hz;
  int16_t values[FINGER_COUNT];
};

// ===================== GLOBALS =====================
static int baseline[ANALOG_COUNT] = {0, 0, 0, 0};
static int latestMag[FINGER_COUNT] = {0, 0, 0, 0, 0};
static float smoothMag[ANALOG_COUNT] = {0, 0, 0, 0};

static uint32_t lastSampleUs = 0;
static uint32_t lastTxUs = 0;
static uint32_t txSeq = 0;
static volatile bool lastSendOk = false;

// Broadcast peer so the receiver can be any ESP32 listening on the same channel.
static const uint8_t BROADCAST_ADDR[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

static inline void setStatusLed(bool on) {
  digitalWrite(STATUS_LED_PIN, (on == LED_ACTIVE_HIGH) ? HIGH : LOW);
}

void onEspNowSent(const uint8_t* mac_addr, esp_now_send_status_t status) {
  (void)mac_addr;
  lastSendOk = (status == ESP_NOW_SEND_SUCCESS);
}

void printMacAddress() {
  uint8_t mac[6] = {0};
  WiFi.macAddress(mac);
  Serial.printf("Sender MAC: %02X:%02X:%02X:%02X:%02X:%02X\n",
                mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
}

void calibrateBaseline() {
  long sums[ANALOG_COUNT] = {0, 0, 0, 0};
  uint32_t count = 0;
  uint32_t startMs = millis();

  if (PRINT_DEBUG_TO_SERIAL) {
    Serial.println("Calibrating baseline... keep sensors in neutral position.");
  }

  while (millis() - startMs < BASELINE_MS) {
    for (uint8_t i = 0; i < ANALOG_COUNT; i++) {
      sums[i] += analogRead(ANALOG_HALL_PINS[i]);
    }
    count++;
    delay(2);
  }

  if (count == 0) count = 1;
  for (uint8_t i = 0; i < ANALOG_COUNT; i++) {
    baseline[i] = (int)(sums[i] / (long)count);
    smoothMag[i] = 0.0f;
  }

  if (PRINT_DEBUG_TO_SERIAL) {
    Serial.printf("Baselines: %d, %d, %d, %d\n",
                  baseline[0], baseline[1], baseline[2], baseline[3]);
  }
}

void initEspNow() {
  WiFi.mode(WIFI_STA);
  WiFi.disconnect(true, true);
  delay(100);

  esp_err_t err = esp_wifi_set_channel(ESPNOW_CHANNEL, WIFI_SECOND_CHAN_NONE);
  if (err != ESP_OK) {
    Serial.printf("esp_wifi_set_channel failed: %d\n", (int)err);
  }

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW init failed");
    while (true) {
      setStatusLed(true);
      delay(100);
      setStatusLed(false);
      delay(100);
    }
  }

  esp_now_register_send_cb(onEspNowSent);

  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, BROADCAST_ADDR, 6);
  peerInfo.channel = ESPNOW_CHANNEL;
  peerInfo.encrypt = false;

  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Failed to add broadcast peer");
    while (true) {
      setStatusLed(true);
      delay(250);
      setStatusLed(false);
      delay(250);
    }
  }
}

void sampleHallSensors() {
  const uint32_t nowUs = micros();
  if ((uint32_t)(nowUs - lastSampleUs) < SAMPLE_PERIOD_US) return;
  lastSampleUs = nowUs;

  for (uint8_t i = 0; i < ANALOG_COUNT; i++) {
    int raw = analogRead(ANALOG_HALL_PINS[i]);
    int mag = abs(raw - baseline[i]);
    if (mag <= MAG_DEADBAND) mag = 0;

    smoothMag[i] = SENSOR_ALPHA * mag + (1.0f - SENSOR_ALPHA) * smoothMag[i];
    latestMag[i] = (int)roundf(smoothMag[i]);
  }

  latestMag[4] = (digitalRead(DIGITAL_HALL_PIN) == LOW)
                   ? DIGITAL_ACTIVE_VALUE
                   : DIGITAL_INACTIVE_VALUE;
}

void sendPacketIfDue() {
  const uint32_t nowUs = micros();
  if ((uint32_t)(nowUs - lastTxUs) < TX_PERIOD_US) return;
  lastTxUs = nowUs;

  HallPacket packet = {};
  packet.seq = txSeq++;
  packet.sender_ms = millis();
  packet.sample_hz = SAMPLE_HZ;
  packet.tx_hz = TX_HZ;
  for (uint8_t i = 0; i < FINGER_COUNT; i++) {
    packet.values[i] = (int16_t)latestMag[i];
  }

  esp_err_t err = esp_now_send(BROADCAST_ADDR, reinterpret_cast<const uint8_t*>(&packet), sizeof(packet));
  lastSendOk = (err == ESP_OK);

  setStatusLed(true);
  ledOffAtMs = millis() + TX_LED_ON_MS;

  if (PRINT_DEBUG_TO_SERIAL && (packet.seq % TX_HZ == 0)) {
    Serial.printf("TX seq=%lu ok=%d values=%d,%d,%d,%d,%d\n",
                  (unsigned long)packet.seq,
                  lastSendOk ? 1 : 0,
                  packet.values[0], packet.values[1], packet.values[2], packet.values[3], packet.values[4]);
  }
}

void setup() {
  Serial.begin(115200);
  delay(400);

  pinMode(STATUS_LED_PIN, OUTPUT);
  setStatusLed(false);

  for (uint8_t i = 0; i < ANALOG_COUNT; i++) {
    pinMode(ANALOG_HALL_PINS[i], INPUT);
  }
  pinMode(DIGITAL_HALL_PIN, INPUT_PULLUP);

  analogReadResolution(12);
  calibrateBaseline();
  initEspNow();
  printMacAddress();

  Serial.printf("ESP-NOW sender ready on channel %u at %u Hz\n", ESPNOW_CHANNEL, TX_HZ);
}

void loop() {
  sampleHallSensors();
  sendPacketIfDue();

  if (ledOffAtMs != 0 && millis() >= ledOffAtMs) {
    ledOffAtMs = 0;
    setStatusLed(false);
  }
}
