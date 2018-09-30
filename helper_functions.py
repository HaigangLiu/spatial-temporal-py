import os, json
from datetime import date, timedelta
from shapely.geometry import Point
from shapely.prepared import prep
import geopandas as gpd

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
    '''
    jsonfile = open('./state_shapes/state_boundaries.json')
    data = json.load(jsonfile)

    if len(state_name) != 2:
        acrynoms = data.keys()
        full_names = [data[acrynom]['name'] for acrynom in acrynoms]
        lookup_table = {full_name: acrynom for acrynom, full_name in zip(acrynoms, full_names)}
        try:
            state_name = lookup_table[state_name]
        except KeyError:
            print('Check the spelling of state name')
            return None
    return data[state_name]

def get_state_information(state_name='South Carolina'):
    '''
    generate the dataframe file with all available locations for a certain state
    method:
    first use sc_lat_lon_dict to narrow down the lookup range. Then apply a more detailed lookup.
    Args:
        state_name: the name of the state to look up in the NWS database
        Can be full name or acrynom. e.g.: either SC or South Carolina works.
    '''

    nws_locations = gpd.read_file('./all_locations/nws_precip_allpoint/nws_precip_allpoint.shp')
    state_information = get_state_range(state_name)

    state_name = state_information['name']
    m1 = nws_locations.LAT > state_information["min_lat"]
    m2 = nws_locations.LAT < state_information['max_lat']
    m3 = nws_locations.LON > state_information['min_lng']
    m4 = nws_locations.LON < state_information['max_lng']

    pre_screened_points = nws_locations[m1 & m2 & m3 & m4]
    shape_for_states = gpd.read_file('./state_shapes/cb_2017_us_state_500k.shp')

    state_contour = shape_for_states[shape_for_states.NAME ==state_name]['geometry'].values[0]
    state_contour_ = prep(state_contour)

    in_domain = []
    for idx, row in pre_screened_points.iterrows():
        lon, lat = row[['LON', 'LAT']]
        if state_contour_.contains(Point([lon, lat])):
            in_domain.append(True)
        else:
            in_domain.append(False)

    in_domain_points = pre_screened_points[in_domain]
    state_locations = in_domain_points[['LAT', 'LON']].round(4)
    return state_locations, state_contour

if __name__ == '__main__':

    jsonfile = open('./state_shapes/state_boundaries.json')
    data = json.load(jsonfile)

    for state in data.keys():
        locs, _ = get_state_information(state)
        abs_dir = os.path.join(abs_dir, state+'.csv')
        locs.to_csv(abs_dir)
        print(f'finished processing state {state}')
