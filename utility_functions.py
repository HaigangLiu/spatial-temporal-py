import numpy as np
import pandas as pd
import os, json
from datetime import date, timedelta
# from shapely.geometry import Point
# from shapely.prepared import prep
import geopandas as gpd
import pandas as pd
import rasterio

def coordinates_converter(lat_lon_df, R = 3959):
    """
    Asssuming that earth is a perfect sphere.
    convert lon, lat coordinates of a point to a 3-D vector.
    The radius of earth is 3959
    """
    if isinstance(lat_lon_df, pd.DataFrame):
        try:
            lon_r = np.radians(lat_lon_df['LONGITUDE'])
            lat_r = np.radians(lat_lon_df['LATITUDE'])
        except KeyError:
            print('Need LONGITUDE and LATITUDE columns')
            return None

        x =  R * np.cos(lat_r) * np.cos(lon_r)
        y = R * np.cos(lat_r) * np.sin(lon_r)
        z = R * np.sin(lat_r)

        output = pd.DataFrame(np.array(list(zip(x, y, z))))
        output.columns = ['x', 'y', 'z']
        return output
    else:
        raise TypeError('the only accepted input type is pandas dataframe')
        return None

def generate_in_between_dates(start_date, end_date):
    '''
    for a start date and end date, generate the dates in between
    both start and end dates will be included in the final output
    args:
        start_date (string): Must follow 'xxxx-xx-xx' order: (year-month-day)
        end_date (string): Must follow 'xxxx-xx-xx' order: (year-month-day)
    '''
    s_year, s_month, s_day = start_date.split('-') #s for start
    e_year, e_month, e_day = end_date.split('-') #e for end

    start_date_formatted = date(int(s_year), int(s_month), int(s_day))
    end_date_formatted = date(int(e_year), int(e_month), int(e_day))
    delta = end_date_formatted - start_date_formatted

    list_of_dates = []
    for i in range(delta.days + 1):
        date_ = str(start_date_formatted + timedelta(i))
        list_of_dates.append(date_)
    return list_of_dates

def get_state_range(state_name):
    '''
    find maximum and miminum of latitude and longitude for a given state
    args:
        state_name: The name of the state. Can be full name or acrynom
            e.g.: either SC or South Carolina works.
    return:
        a dictionary of state range. The keys are given as follows:
            name, min_lat, min_lng, max_lat, max_lng
    '''
    jsonfile = open('./state_shapes/state_boundaries.json')
    data = json.load(jsonfile)

    if len(state_name) != 2:
        acrynoms = data.keys()
        full_names = [data[acrynom]['name'] for acrynom in acrynoms]
        lookup_table = {full_name: acrynom for acrynom, full_name in zip(acrynoms, full_names)}
        try:
            state_name = [word.capitalize() for word in state_name.split(' ')]
            state_name = lookup_table[' '.join(state_name)]
        except KeyError:
            print('Check the spelling of state name')
            return None
    return data[state_name]

def get_state_contours(state_name='South Carolina', use_cache=True):
    '''
    find the polygon of a given state in the united states,
    also generate the dataframe file with all available locations for a certain state

    method:
    first use sc_lat_lon_dict to narrow down the lookup range.
    Then apply a more detailed lookup by within() function from shapely.

    Args:
        state_name (string): the name of the state to look up in the NWS database
        Can be full name or acrynom. e.g.: either SC or South Carolina works.
        use_cache (boolean)): will use the cached csv file to speed up the computation if possible
    '''
    state_information = get_state_range(state_name) #use full name thereafter
    state_name = state_information['name']

    shape_for_states = gpd.read_file('./state_shapes/cb_2017_us_state_500k.shp')
    state_contour = shape_for_states[shape_for_states.NAME == state_name]

    if len(state_contour):
        state_contour = state_contour['geometry'].values[0]
    else:
        print(f'There is no contour information for {state_name}')
        return None, None

    if use_cache:
        try:
            file_name = os.path.join('./state_shapes/', state_name + '.csv')
            state_locations = pd.read_csv(file_name)
            print(f'loaded cached file for {state_name}')
            return state_locations, state_contour

        except FileNotFoundError:
            print('cached file not found. start downloading from national weather service site.')

    nws_locations = gpd.read_file('./all_locations/nws_precip_allpoint/nws_precip_allpoint.shp')

    m1 = nws_locations.LAT > state_information["min_lat"]
    m2 = nws_locations.LAT < state_information['max_lat']
    m3 = nws_locations.LON > state_information['min_lng']
    m4 = nws_locations.LON < state_information['max_lng']
    pre_screened_points = nws_locations[m1 & m2 & m3 & m4]

    state_contour_ = prep(state_contour)
    domain = []
    for idx, row in pre_screened_points.iterrows():
        lon, lat = row[['LON', 'LAT']]
        if state_contour_.contains(Point([lon, lat])):
            domain.append(True)
        else:
            domain.append(False)

    in_domain_points = pre_screened_points[domain]
    if len(in_domain_points) > 0:
        state_locations = in_domain_points[['LAT', 'LON']].round(4)
    else:
        print(f'no observational locations found in {state_name}')
        return None, None
    return state_locations, state_contour


def find_elevation(dataframe_copy, key, lat='LATITUDE', lon='LONGITUDE'):

    source_1 = './data/shape_file/elevation_info/srtm_20_05/srtm_20_05.tif'
    source_2 = './data/shape_file/elevation_info/srtm_20_06/srtm_20_06.tif'
    source_3 = './data/shape_file/elevation_info/srtm_21_05/srtm_21_05.tif'
    source_4 = './data/shape_file/elevation_info/srtm_21_06/srtm_21_06.tif'

    lats_and_lons = dataframe_copy.groupby([key]).first()[[lat,lon]]
    source_collection = [source_1, source_2, source_3, source_4]

    for idx, lat_and_lon in lats_and_lons.iterrows():

        lat_and_lon = list(lat_and_lon)
        mask = dataframe_copy[key] == idx

        for source in source_collection:
            with rasterio.open(source) as f:
                try:
                    vals = f.sample([lat_and_lon])
                    for val in vals:
                        if val >= 0: #lowest point of sc is sea level
                            dataframe_copy.loc[mask, 'ELEVATION'] = val[0]
                except IndexError:
                    continue
                else:
                    break
    return dataframe_copy

if __name__ == '__main__':

    jsonfile = open('./state_shapes/state_boundaries.json')
    data = json.load(jsonfile)

    a, b = get_state_contours('VI') #virgin island
    c, d = get_state_contours('south carolina') #now allow lower case

    for state in data.keys(): #for all states
        result = get_state_contours(state)
        if result:
            abs_dir = os.path.join('./state_shapes', data[state]['name']+'.csv')
            locs, _ = result
            locs.to_csv(abs_dir)
            print(f'finished processing state {state}')

