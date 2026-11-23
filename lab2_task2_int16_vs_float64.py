import pandas as pd
import time

city_temperature = pd.read_csv("city_temperature.csv")
#print(city_temperature.shape)
#city_temperature.head()
#print(city_temperature['State'].isna().sum())
#print(city_temperature['AvgTemperature'][:10].astype('int16') * 10)
city_temperature['Temp'] = city_temperature['AvgTemperature'].astype('int16') * 10

start_time_int16 = time.perf_counter()
city_temperature['Temp'].mean()
end_time_int16 = time.perf_counter()
exec_time_int16 = end_time_int16 - start_time_int16

start_time_float64 = time.perf_counter()
city_temperature['AvgTemperature'].mean()
end_time_float64 = time.perf_counter()
exec_time_float64 = end_time_float64 - start_time_float64

print(f"int16:   {exec_time_int16:.10f}\nfloat64: {exec_time_float64:.10f}")
