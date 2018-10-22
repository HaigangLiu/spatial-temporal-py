import os, json, fiona, rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import date, timedelta
from shapely.geometry import Point, shape
from shapely.prepared import prep

path_watershed ='./data/shape_file/hydrologic_HUC8_units/wbdhu8_a_sc.shp'

def coordinates_converter(lat_lon_df, lat='LATITUDE', lon='LONGITUDE', R=3959, lon_first=True):
    """
    Asssuming that earth is a perfect sphere.
    convert lon, lat coordinates of a point to a 3-D vector.
    The radius of earth is 3959
    """
    if isinstance(lat_lon_df, pd.DataFrame):
        try:
            lon_r = np.radians(lat_lon_df[lat])
            lat_r = np.radians(lat_lon_df[lon])
        except KeyError:
            print(f'cannot find {lat} and {lon} columns')
            return None
    else:
        if lon_first:
            lon_r = np.radians(lat_lon_df[:, 0])
            lat_r = np.radians(lat_lon_df[:, 1])
            print('assuming that longitude comes first')
            print('to change this, set lon_first to False')
        else:
            lon_r = np.radians(lat_lon_df[:, 1])
            lat_r = np.radians(lat_lon_df[:, 0])
            print('assuming that latitude comes first')
            print('to change this, set lon_first to True')

    x =  R * np.cos(lat_r) * np.cos(lon_r)
    y = R * np.cos(lat_r) * np.sin(lon_r)
    z = R * np.sin(lat_r)

    output = pd.DataFrame(np.array(list(zip(x, y, z))))
    output.columns = ['x', 'y', 'z']
    return output

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

    shape_for_states = fiona.open('./state_shapes/cb_2017_us_state_500k.shp')
    for row in shape_for_states:
        if row['properties']['NAME'] == state_name:
            polygon = shape(row['geometry'])
            break
    else:
        print(f'cannot find information for {state_name}. check the spelling')
        return None
    return polygon

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
    state_contour = prep(get_state_contours(state_name)) # precompiled version;
    # faster
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
                    if state_contour.contains(shape(line['geometry'])):
                        local_save.write('\t'.join([str(lon), str(lat)]))
                        local_save.write('\n')
    print(f'the grid information has been saved to {abs_file_dir}')
    return pd.read_csv(abs_file_dir, sep='\t')

def get_watersheds_list(reverse=False):
    '''
    get a list of watersheds and their code in a dictionary for SC.
    reverse: if true, the value and the key will be flipped in the output.
    '''
    watersheds_dict = {} #key is code, value is name
    with fiona.open(path_watershed) as f:
        for row in f:
            name = row['properties']['NAME']
            key = row['properties']['HUC8']
            watersheds_dict[key] = name
    if reverse:
        watersheds_dict = {v: k for k, v in watersheds_dict.items()} #key is name, value is code
    return watersheds_dict

def get_dict_basins_to_watershed(source_file=None, mode='name', reverse=False):
    '''
    generate a dictionary that maps basins to watershed. set reverse to true if
    users want to go the other way around.
    Only support South Carolina because of limited basin information nationwide.

    mode (string): code or name. code will return 8-digit code for watershed.
    reverse (boolean): if true, will return a dictionary with watershed as key. default is False.
    '''
    if source_file is None:
        source_file = './basin_list.txt'

    watersheds_dict = get_watersheds_list(reverse=False) #key is name, value is code
    temp_container = []
    with open(source_file) as file:
        each_loc = []
        for element in file:
            if element != '\n':
                each_loc.append(element.strip('\n'))
            else:
                temp_container.append(each_loc)
                each_loc = [] #reset
                continue

    basin_to_watershed = {} #each basin has which watershed
    for entry in temp_container:
        k = entry.pop(0)
        basin_to_watershed[k] = []
        for v in entry:
            if mode=='name':
                v = watersheds_dict[v]
            basin_to_watershed[k].append(v)
    output_dict = basin_to_watershed.copy()

    if reverse:
        watershed_to_basin = {}
        for k, vs in basin_to_watershed.items():
            for v in vs:
                watershed_to_basin[v] = k
        output_dict = watershed_to_basin.copy()
    return output_dict

def get_state_fullname(state, reverse=False):
    with open('./state_shapes/state_boundaries.json') as j:
        data = json.load(j)
        state_fullname_dict = {k:v['name'] for k, v in data.items()}
    if reverse:
        state_fullname_dict = {v:k for k, v in state_fullname_dict.items()}
    return state_fullname_dict[state]

if __name__ == '__main__':

    # a = get_state_contours('Wisconsin')
    # # b = get_state_grid_points('Wisconsin')
    # c = get_state_rectangle('Wisconsin')
    # d = get_elevation(get_watershed)
    # e = get_in_between_dates('1990-01-01','1990-01-04' )
    # # print(b.head())
    # print(d)
    from SampleDataLoader import load_flood_data
    ss = load_flood_data('daily')
    s = get_watershed([-81,33])
    sss = get_watershed(ss, key='SITENUMBER', lat='LATITUDE',lon='LONGITUDE')
    print(len(np.unique(sss.WATERSHED)))
    # sss.to_csv('/Users/haigangliu/Desktop/check_2.csv')

    print(get_dict_basins_to_watershed(mode='name', reverse=False))
    print(get_dict_basins_to_watershed(mode='code', reverse=False))
    print(get_dict_basins_to_watershed(mode='name', reverse=True))
    print(get_dict_basins_to_watershed(mode='code', reverse=True))



