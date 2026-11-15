# AI Space Guardian — Dataset Summary

> All data is **synthetic**. Not real NASA data. Not medically validated.

**Generated:** 2026-11-15T00:15:37.771397 UTC  
**Simulation:** 1 day(s), interval 5s  
**Random seed:** 42

## Record counts
- Astronaut sensor records: **69,120**
- Spacecraft environment records: **17,280**
- Labelled alerts / anomalies: **28**
- Device rows: **21**

## Records per astronaut
- `AST-001`: 17,280
- `AST-002`: 17,280
- `AST-003`: 17,280
- `AST-004`: 17,280

## Missing readings per sensor
- `heart_rate_bpm`: 319 missing
- `spo2_percent`: 319 missing
- `radiation_dose_rate_usv_h`: 211 missing
- `sleep_quality_score`: 317 missing
- `movement_indicator`: 120 missing

## Sensor-status distribution
- `ok`: 68,133
- `disconnected`: 636
- `malfunction`: 211
- `missing`: 120
- `corrupted`: 20

## Scenario-label distribution
- `sleeping`: 22,869
- `working_inside_spacecraft`: 21,066
- `resting`: 10,022
- `exercising`: 7,195
- `walking`: 7,138
- `sensor_disconnection`: 319
- `equipment_malfunction`: 211
- `unusual_sensor_reading`: 160
- `missing_measurement`: 120
- `corrupted_reading`: 20

## Alerts per astronaut
- `AST-001`: 7 labelled events
- `AST-002`: 7 labelled events
- `AST-003`: 7 labelled events
- `AST-004`: 7 labelled events

## Time coverage
- Start: 2031-03-01 00:00:00
- End:   2031-03-01 23:59:55
- Duration: 0 days 23:59:55

## Files
- `astronaut_alert_scenarios.csv` (4,029 bytes)
- `astronaut_sensor_data.csv` (31,161,526 bytes)
- `astronaut_sensor_data.json` (68,082,708 bytes)
- `astronauts.csv` (725 bytes)
- `data_dictionary.md` (3,860 bytes)
- `devices.csv` (2,371 bytes)
- `spacecraft_environment_data.csv` (3,093,831 bytes)
- `visualizations/` (12 files)