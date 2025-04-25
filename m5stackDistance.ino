/**
 * @file ContinuousWithDetails.ino
 * @author SeanKwok (shaoxiang@m5stack.com)
 * @brief M5Unit ToF4M Continuous Read Data With Details Demo.
 * @version 0.1
 * @date 2023-11-23
 *
 *
 * @Hardwares: M5Unit ToF4M
 * @Platform Version: Arduino M5Stack Board Manager v2.0.7
 * @Dependent Library:
 * VL53L1X: https://github.com/pololu/vl53l1x-arduino
 * M5Unified: https://github.com/m5stack/M5Unified
 */

#include <FastLED.h>

// How many leds in your strip?
#define NUM_LEDS 61

// For led chips like WS2812, which have a data line, ground, and power, you just
// need to define DATA_PIN.  For led chipsets that are SPI based (four wires - data, clock,
// ground, and power), like the LPD8806 define both DATA_PIN and CLOCK_PIN
// Clock pin only needed for SPI based chipsets when not using hardware SPI
#define DATA_PIN 8
// Define the array of leds
CRGB leds[NUM_LEDS];

#include "M5Unified.h"
#include <Wire.h>
#include <VL53L1X.h>
#include <math.h>

#include <WiFi.h>
#include <M5Unified.h>
#include <HTTPClient.h>

#include <ESPAsyncWebServer.h>

const char* ssid = "Rogers1";
const char* password = "4168265982";

AsyncWebServer server(80);
String receivedCommand = "wait";

VL53L1X sensor;

void setup() {
  Serial.begin(115200);
  FastLED.addLeds<WS2812, DATA_PIN, GRB>(leds, NUM_LEDS);  // GRB ordering is typical
  M5.begin();
  // M5.Ex_I2C.begin(21, 22, 400000);
  M5.Ex_I2C.begin();
  sensor.setBus(&Wire);
  sensor.setTimeout(500);
  if (!sensor.init()) {
    Serial.println("Failed to detect and initialize sensor!");
    while (1)
      ;
  }

  // Use long distance mode and allow up to 50000 us (50 ms) for a
  // measurement. You can change these settings to adjust the performance of
  // the sensor, but the minimum timing budget is 20 ms for short distance
  // mode and 33 ms for medium and long distance modes. See the VL53L1X
  // datasheet for more information on range and timing limits.
  sensor.setDistanceMode(VL53L1X::Long);
  sensor.setMeasurementTimingBudget(50000);

  // Start continuous readings at a rate of one measurement every 50 ms (the
  // inter-measurement period). This period should be at least as long as the
  // timing budget.
  sensor.startContinuous(50);

  // connecting to the wifi to send the distance measure
  M5.begin();
  Serial.begin(115200);
  Serial.println("Connecting to WiFi...");

  WiFi.begin(ssid, password);
  Serial.println("Connecting to WiFi...");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nConnected to WiFi!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  // Define POST endpoint
  server.on("/command", HTTP_POST, [](AsyncWebServerRequest *request){
    // Send a response AFTER the body has been handled
    request->send(200, "text/plain", "POST received.");
    receivedCommand = "done";
  });

  
  server.begin();
}

void loop() {

    sensor.read();

    Serial.print("range: ");
    int distance = sensor.ranging_data.range_mm;
    Serial.print(distance);

    int num_leds = round(distance / 20.0)+4;
    Serial.print("led_num: ");
    Serial.print(num_leds);

    for (int i = 0; i < 47; i++) {
        leds[i] = CRGB::Black;
    }
    FastLED.show();

    if (distance >= 0 && distance <= 75) {
        leds[6] = CRGB::Yellow;
    }
    if (distance >= 830) {
        Serial.print("No book selected");
    }
    else{
        leds[num_leds] = CRGB::Yellow;
    }
    // Turn the LED on, then pause
    
    FastLED.show();

    Serial.println();
    if (distance < 830){
        if (WiFi.status() == WL_CONNECTED) {
            HTTPClient http;
            http.begin("http://10.0.0.42:8080/data");
            http.addHeader("Content-Type", "application/json");

            String distance_str = String(distance);
            int httpResponseCode = http.POST(distance_str);
            Serial.print("HTTP Response code: ");
            Serial.println(httpResponseCode);
            http.end();
            while(receivedCommand == "wait"){
                delay(1000);
            }
            receivedCommand = "wait";
        }
    }
    delay(1000); // Wait a bit before repeating
}
