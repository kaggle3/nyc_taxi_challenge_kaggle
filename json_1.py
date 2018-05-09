#author:sushma
import json,urllib,requests
import pandas as pd

data =requests.get('http://www.nyc.gov/html/dot/downloads/misc/speed_limit_manhattan.json')
d=data.json()
#print(d)
locations=[]
properties=[]
final=[]
for feature in d['features']:
    #pd.DataFrame(feature['geometry']['type'])

    locations.append((feature['geometry']['coordinates']))
    properties.append(feature['properties'])
#for type in d['type']:
 #   properties.append()
print(len(locations))
print(len(properties))
for ind,element in enumerate(locations):
    for item in element:
        row=[item[0],item[1],properties[ind]['postvz_sg'],properties[ind]['street'],properties[ind]['postvz_sl']]
        final.append(row)
df=pd.DataFrame(final,columns=['longitude','latitude','postvz_sg','street','speed_limit'])
df=df.replace('YES',1)
df=df.replace('NO',0)
print(df.info())
df.to_csv('speed_limit.csv')
