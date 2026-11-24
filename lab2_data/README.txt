ISyE 6767 Lab 2 data -- ERCOT, delivery days 2022-12-01 to 2025-12-31 (all times Central Prevailing Time)

prices/rpt.00013060...DAMLZHBSPP_YYYY.xlsx
    ERCOT NP4-180-ER, Historical DAM Load Zone and Hub Prices ($/MWh). One sheet per month,
    columns: Delivery Date, Hour Ending ("01:00".."24:00"), Repeated Hour Flag (Y = second copy of the
    fall-back hour), Settlement Point, Settlement Point Price.
prices/rpt.00013061...RTMLZHBSPP_YYYY.xlsx
    ERCOT NP6-785-ER, Historical RTM Load Zone and Hub Prices ($/MWh), 15-minute intervals. One sheet per month,
    columns: Delivery Date, Delivery Hour, Delivery Interval (1-4), Repeated Hour Flag, Settlement Point Name,
    Settlement Point Type, Settlement Point Price.
load_forecast/YYYYMMDD.csv     ERCOT NP3-561-CD  Seven-Day Load Forecast by Weather Zone (MW)
wind_forecast/YYYYMMDD.csv     ERCOT NP4-732-CD  Wind Power Production - Hourly Averaged Actual and Forecasted Values (MW)
solar_forecast/YYYYMMDD.csv    ERCOT NP4-737-CD  Solar Power Production - Hourly Averaged Actual and Forecasted Values (MW)
    YYYYMMDD = delivery day D. The file is the last one ERCOT posted before 09:00 Central on D-1
    (the Project 2 decision time), unchanged. Posting times: forecast_posting_times.csv.
