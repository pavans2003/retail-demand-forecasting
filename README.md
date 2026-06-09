# Adaptive Retail Demand Forecasting & Inventory Planning

An adaptive ML framework for retail demand forecasting and inventory decision support, evaluated on the M5 Walmart dataset (FOODS_1 category, CA_1 store).

## Framework Overview
1. **Demand Analysis** — STL decomposition, ACF, ADF stationarity test, ADI-CV² classification
2. **Adaptive Model Routing** — model selected per SKU based on demand type (smooth → RF/LightGBM, intermittent → Croston/SBA/TSB)
3. **Rolling Forecast Evaluation** — evaluated under identical conditions across all models
4. **Inventory Planning** — safety stock, reorder point, ABC-XYZ classification

## Results Summary
| SKU | Best Model | MASE | Reliability | ABC-XYZ |
|-----|-----------|------|-------------|---------|
| FOODS_1_001 | Random Forest | 0.82 | High | AY |
| FOODS_1_004 | LightGBM | 1.81 | Low | BZ |
| FOODS_1_005 | Random Forest | 0.89 | High | AZ |
| FOODS_1_008 | Croston | 2.01 | Low | CZ |

## Tech Stack
Python · LightGBM · Scikit-Learn · statsmodels (ARIMA/STL) · Pandas · NumPy

## Reference
Project Report — NIT Calicut, Dept. of Mechanical Engineering, May 2026  
Guides: Dr. Sajan T John & Dr. Sanghamitra Das

## Key Design Decisions
- ADI-CV² framework routes each SKU to demand-appropriate models instead of one-size-fits-all
- MASE used as primary metric — scale-independent, works across intermittent and smooth demand
- Safety stock formula accounts for forecast reliability: higher MASE → stricter service level → larger buffer
