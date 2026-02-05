---
name: KPI Calculation Procedures
version: "1.0"
description: Standard KPI calculation procedures for pharma commercial analytics
triggers:
  - calculate TRx
  - calculate NRx
  - market share
  - conversion rate
  - KPI calculation
  - prescription metrics
agents:
  - gap_analyzer
  - causal_impact
categories:
  - pharma-commercial
---

# KPI Calculation Procedures

Standard procedures for calculating pharmaceutical commercial KPIs.

## TRx (Total Prescriptions)

Total prescriptions dispensed for a given brand or therapeutic area.

### Calculation

TRx = New prescriptions (NRx) + Refill prescriptions (RRx)

### Data Sources

- IQVIA claims data
- Symphony Health data

## NRx (New Prescriptions)

New prescriptions written by HCPs.

### Calculation

NRx = First-fill prescriptions in the measurement period

## Market Share

Brand share of total prescriptions in a therapeutic category.

### Calculation

Market Share = Brand TRx / Category TRx * 100

## Conversion Rate

Rate at which NRx converts to ongoing therapy.

### Calculation

Conversion Rate = Patients with >= 2 fills / Patients with >= 1 fill

## PDC (Proportion of Days Covered)

Adherence metric measuring prescription coverage.

### Calculation

PDC = Days with medication supply / Days in measurement period
