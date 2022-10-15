import xml.etree.ElementTree as et
import pandas as pd

# cols = ["data_timestep", "lane_id", "lane_queueing_length", "lane_queueing_length_experimental", "lane_queueing_time"]
cols = ["vehicles", "teleports", "safety", "persons", "vehicleTripStatistics", "pedestrianStatistics", "rideStatistics", "transportStatistics"]
rows = []

xmlparse = et.parse('out.xml')
root = xmlparse.getroot()
for i in root:
    vehicles = i.find("vehicles").text
    teleports = i.find("teleports").text
    safety = i.find("safety").text
    persons = i.find("persons").text
    vehicleTripStatistics = i.find("vehicleTripStatistics").text
    pedestrianStatistics = i.find("pedestrianStatistics").text
    rideStatistics = i.find("rideStatistics").text
    transportStatistics = i.find("transportStatistics").text

    rows.append({
        "vehicles": vehicles,
        "teleports": teleports,
        "safety": safety,
        "persons": persons,
        "vehicleTripStatistics": vehicleTripStatistics,
        "pedestrianStatistics": pedestrianStatistics,
        "rideStatistics": rideStatistics,
        "transportStatistics": transportStatistics
    })
    
    df = pd.DataFrame(rows, columns=cols)

    df.to_csv('out.csv')
    