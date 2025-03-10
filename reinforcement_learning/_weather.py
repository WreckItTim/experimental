import airsim



# Rain: 0,
# Roadwetness: 1,
# Snow: 2,
# RoadSnow: 3,
# MapleLeaf: 4,
# RoadLeaf: 5,
# Dust: 6,
# Fog: 7

client = airsim.MultirotorClient()
print(client.confirmConnection())
client.reset()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()
client.simEnableWeather(True)

for i in range(8):
    client.simSetWeatherParameter(i, 0)

weather = 'rain'
degree = 1

if weather == 'rain':
    client.simEnableWeather(True)
    client.simSetWeatherParameter(0, degree)
    client.simEnableWeather(True)
    client.simSetWeatherParameter(1, degree)

if weather == 'snow':
    client.simSetWeatherParameter(2, degree)
    client.simSetWeatherParameter(3, degree)

if weather == 'leaf':
    client.simSetWeatherParameter(4, degree)
    client.simSetWeatherParameter(5, degree)

if weather == 'dust':
    client.simSetWeatherParameter(6, degree)

if weather == 'fog':
    client.simSetWeatherParameter(7, degree)

print('done')
