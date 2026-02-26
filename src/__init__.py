"""
CB-TimeSense: Automatic Circuit Breaker Closing Time Estimation
for Synchronism-Check Automatic Reclosure (АПВ с Улавливанием Синхронизма)

Modules:
    download      - Dataset download from Figshare
    comtrade_parser - COMTRADE file parsing (IEEE C37.111)
    cb_grouper    - Circuit breaker identification and grouping
    closing_detector - Closing time detection from signals
    models        - Prediction models (Kalman, EWMA, Median, Regression)
    evaluation    - Metrics and evaluation framework
"""
