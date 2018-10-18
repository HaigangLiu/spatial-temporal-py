import os, json, fiona, rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import date, timedelta
from shapely.geometry import Point, shape
from shapely.prepared import prep

path_watershed ='./data/shape_file/hydrologic_HUC8_units/wbdhu8_a_sc.shp'

def coordinates_converter(lat_lon_df, R=3959, lon_first=True):
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
                        if val[0] >= 0: #lowest point of sc is sea level
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
        input_df = input.copy()
        if key is None:
            raise ValueError('Need to specify the name of a column of station identifier. e.g. STATIONID ')
        try:
            #construct a summary table to enhance performance
            summary = input_df.groupby([key]).first().reset_index()[[key, lon, lat]]

            input_df.set_index(key, inplace=True)
            for idx, row in summary.iterrows():
                key, lat, lon = row.values
                height = get_elevation_one_loc([lat, lon])
                input_df.loc[key, 'ELEVATION'] = height

        except KeyError:
            print(f'make sure the dataframe contains: LATITUDE, LONGITUDE and {key}')
            return None
    return input_df.reset_index()


def get_watershed(input, shapfile=None, key=None, lat=None, lon= None, ):
    '''
    for each location, find which watershed it belongs to.
    A attribute 'data' will be generated which is a new
    dataframe with an additional column called WATERSHED
    '''
    if shapfile is None:
        huc8_units = gpd.read_file(path_watershed)
        water_shed_info = huc8_units[['NAME', 'geometry']]
        # water_shed_info = prep(water_shed_info)

    def get_watershed_for_one_loc(list_):
        for _, rows in water_shed_info.iterrows():
                    name, polygon = rows
                    if Point(list_).within(polygon):
                        return name
        else:
            raise ValueError('watershed not found')

    if isinstance(input, list):
        print('assuming the format is [longitude, latitude]')
        print('the unit is measured in meter')
        return get_watershed_for_one_loc(input)

    elif isinstance(input, pd.DataFrame):
        input_df = input.copy()
        if (key is None) or (lat is None) or (lon is None):
            raise ValueError('Need to specify the name of latitude column and longitude column and station name column')
        try:
            summary = input_df.groupby([key]).first().reset_index()[[key, lon, lat]] #better performance
        except KeyError:
            print(f'make sure the dataframe contains: LATITUDE, LONGITUDE and {key}')
            return None

        input_df.set_index(key, inplace=True)
        for idx, row in summary.iterrows():
            key_, lat, lon = row.values
            watershed_name = get_watershed_for_one_loc([lat, lon])
            input_df.loc[key_, 'WATERSHED'] = watershed_name

        input_df.reset_index(inplace=True)

        if True:
            number_of_obs = input_df.groupby(['WATERSHED']).count()[key]
            singular_huc_areas = number_of_obs[number_of_obs<=1].index
            input_df = input_df[~input_df.WATERSHED.isin(singular_huc_areas)]
        print('created a new column called WATERSHED to store the huc watershed information')
        return input_df.reset_index()


def get_dict_basins_to_watershed(mode='code', reverse=False):
    '''
    generate a dictionary that maps basins to watershed. set reverse to true if
    users want to go the other way around.
    Currently only support south carolina because of limited basin information nationwide.

    mode (string): code or name. code will return 8-digit code for watershed.
    reverse (boolean): if true, will return a dictionary with watershed as key. default is False.
    '''
    watersheds = {}
    with fiona.open(path_watershed) as f:
        for row in f:
            name = row['properties']['NAME']
            key = row['properties']['HUC8']
            watersheds[key] = name

    temp_container = []
    with open('./basin_list.txt') as file:
        each_loc = []
        for element in file:
            if element != '\n':
                each_loc.append(element.strip('\n'))
            else:
                temp_container.append(each_loc)
                each_loc = [] #reset
                continue

    basins = {}
    for entry in temp_container:
        k = entry.pop(0)
        basins[k] = []
        for v in entry:
            basins[k].append(v)

    if reverse:
        reversed_dict = {}
        for k, v in basins.items():
            for v_ in v:
                reversed_dict[v_] = k
        basins = reversed_dict

    if mode == 'name':
        nw_dict ={}
        for k, v in basins.items():
            nw_list = [watersheds[v_] for v_ in v]
            nw_dict[k] = nw_list
        return nw_dict

    elif mode == 'code':
        nw_dict = basins
        return nw_dict

    else:
        raise ValueError('allow either name (watershed name) or code (8-digit code)')
        return None

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



