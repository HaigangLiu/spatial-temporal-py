import os, json, fiona, rasterio
import numpy as np
import pandas as pd
from datetime import date, timedelta
import geopandas as gpd
from shapely.geometry import Point
from shapely.prepared import prep

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

def get_in_between_dates(start_date, end_date):
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

def get_state_rectangle(state_name):
    '''
    find maximum and miminum of latitude and longitude for a given state,
    a rectangle covers the state, in other words

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

def get_state_contours(state_name):
    '''
    get the contour information for the give state name
    '''
    if len(state_name) == 2:
        state_name = get_state_rectangle(state_name)['name']
        print(state_name)
    shape_for_states = gpd.read_file('./state_shapes/cb_2017_us_state_500k.shp')
    # print(shape_for_states.NAME)
    state_contour = shape_for_states[shape_for_states.NAME == state_name]

    if len(state_contour):
        state_contour = state_contour['geometry'].values[0]
    else:
        print(f'There is no contour information for {state_name}')
        return None
    return state_contour


def get_state_grid_points(state_name='South Carolina', use_cache=True):
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
    state_information = get_state_rectangle(state_name) #use full name thereafter
    state_name = state_information['name']

    state_contour = get_state_contours(state_name)
    state_contour_ = prep(state_contour)

    if use_cache:
        try:
            file_name = os.path.join('./state_shapes/', '.'.join([state_name, 'csv']))
            state_locations = pd.read_csv(file_name)
            print(f'loaded cached file for {state_name}')
            return state_locations

        except FileNotFoundError:
            print('cached file not found. start downloading from national weather service site.')

    src_file = './all_locations/nws_precip_allpoint/nws_precip_allpoint.shp'
    #prescreen info
    min_lat = state_information["min_lat"]
    max_lat = state_information['max_lat']
    min_lng = state_information['min_lng']
    max_lng = state_information['max_lng']

    file_name = '.'.join([state_name, 'csv'])
    abs_file_dir = os.path.join('./data/', file_name)

    with fiona.open(src_file) as geo_file:
        with open(abs_file_dir, 'w') as local_save:
            local_save.write('LONGITUDE\tLATITUDE\n')
            for line in geo_file:
                lon, lat = line['geometry']['coordinates']
                if (lat>max_lat) or (lat<min_lat) or (lon<min_lng) or (lon>max_lng):
                    pass
                else:
                    if state_contour_.contains(Point([lon, lat])):
                        local_save.write('\t'.join([str(lon), str(lat)]))
                        local_save.write('\n')
    print(f'the grid information has been saved to {abs_file_dir}')
    return abs_file_dir

def get_elevation(input, key=None, lat='LATITUDE', lon='LONGITUDE'):
    '''
    find the elevation for given latitude and longitude
    the input can be of list like [lon, lat]
    or a dataframe that contains longitude column and latitude column

    Args:
        input (pandas dataframe, or list)
        key: if dataframe is given, key must be provided to indicate which
        column is station name
    Return:
        if input is a list:
            returns the height as a float number
        if input is dataframe:
            returns the original dataframe within an additional column for
            elevation
    '''
    source_1 = './data/shape_file/elevation_info/srtm_20_05/srtm_20_05.tif'
    source_2 = './data/shape_file/elevation_info/srtm_20_06/srtm_20_06.tif'
    source_3 = './data/shape_file/elevation_info/srtm_21_05/srtm_21_05.tif'
    source_4 = './data/shape_file/elevation_info/srtm_21_06/srtm_21_06.tif'
    source_collection = [source_1, source_2, source_3, source_4]

    def get_elevation_one_loc(input):
        elevation = None
        for source in source_collection:
            with rasterio.open(source) as f:
                try:
                    vals = f.sample([input])
                    for val in vals:
                        if val >= 0: #lowest point of sc is sea level
                            elevation = val[0]
                except IndexError:
                    continue
                else:
                    break
        return elevation

    if isinstance(input, list):
        print('assuming the format is [longitude, latitude]')
        print('the unit is measured in meter')
        return get_elevation_one_loc(input)

    elif isinstance(input, pd.DataFrame):
        if key is None:
            raise ValueError('Need to specify the name of a column of station identifier. e.g. STATIONID ')
        try:
            #construct a summary table to enhance performance
            summary = input.groupby([key]).first().reset_index()[[key, lon, lat]]

            input.set_index(key, inplace=True)
            for idx, row in summary.iterrows():
                key, lat, lon = row.values
                height = get_elevation_one_loc([lat, lon])
                input.loc[key, 'ELEVATION'] = height

        except KeyError:
            print(f'make sure the dataframe contains: LATITUDE, LONGITUDE and {key}')
            return None
    output = input.reset_index()
    return output

if __name__ == '__main__':

    a = get_state_contours('Wisconsin')
    b = get_state_grid_points('Wisconsin')
    c = get_state_rectangle('Wisconsin')
    d = get_elevation([33, -81])
    e = get_in_between_dates('1990-01-01','1990-01-04' )
    print(e)



