import pandas as pd
import utm

data = pd.read_csv("https://www.fs.fed.us/pnw/starkey/data/tables/Starkey_OR_Main_Telemetry_1993-1996_Data.txt")
data_habitat = pd.read_csv("https://www.fs.fed.us/pnw/starkey/data/tables/Starkey_OR_Main_Habitat_1993-1996_Data.txt")

data_merged = data.merge(data_habitat, left_on='UTMGrid', right_on='UTMGrid')

def convert_utm_lat(row):
    return utm.to_latlon(row[1], row[2], 11, 'T')[0]


def convert_utm_long(row):
    return utm.to_latlon(row[1], row[2], 11, 'T')[1]

def format_date(row):
    data = str(row[6])
    return data[6:8]+'/'+data[4:6]+'/'+data[0:4]+' '+str(row[5])

def rename_animal(row):
    if(str(row[10]).strip() == 'C'):
        return 'cattle'
    if(str(row[10]).strip() == 'E'):
        return 'elk'
    if(str(row[10]).strip() == 'D'):
        return 'deer'
    return row[10]

data_merged['lat'] = data_merged.apply(lambda row : convert_utm_lat(row), axis=1)
data_merged['long'] = data_merged.apply(lambda row : convert_utm_long(row), axis=1)
data_merged['time'] = data_merged.apply(lambda row : format_date(row), axis=1)
data_merged[' Species'] = data_merged.apply(lambda row : rename_animal(row), axis=1) 
#data_merged = data_merged.sort_values(by=[' Id'])

data_merged.rename(columns={'long':'lon',' Id':'tid', ' Species':'target', ' Elev':'altitude'}, inplace=True)
columns_to_use  = ['lat', 'lon', 'altitude', 'time', 'target', 'tid']
data_merged.to_csv('animal.csv', columns = columns_to_use, index = False)
