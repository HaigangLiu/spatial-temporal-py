import os
from shapely.geometry import Point
import geopandas as gpd
import pandas as pd

def generate_grid_file(file_name):
    '''
    generate the csv file with all available locations
    fill the unobserved locations with 0

    first use sc_lat_lon_dict to narrow down the lookup range. Then apply a more detailed lookup.

    Args:
        file_name: name of the csv file to save the lccations
    '''
    all_geopoints = gpd.read_file('./all_locations/nws_precip_allpoint')
    sc_lat_lon_dict = { "min_lat": 32.0453, "max_lat": 35.2075,
                        "min_lon": -83.3588,"max_lon": -78.4836}

    m1 = all_geopoints.LAT > sc_lat_lon_dict["min_lat"]
    m2 = all_geopoints.LAT < sc_lat_lon_dict['max_lat']
    m3 = all_geopoints.LON > sc_lat_lon_dict['min_lon']
    m4 = all_geopoints.LON < sc_lat_lon_dict['max_lon']
    points_sc_crude = all_geopoints[m1 & m2 & m3 & m4]

    sc_dir = os.path.join(os.getcwd(),'data/shape_file/south_carolina/tl_2010_45_state10.shp')
    sc_shape = gpd.read_file(sc_dir)['geometry'][0]

    inside_sc_or_not = []
    for idx, row in points_sc_crude.iterrows():
        lon, lat = row[['LON', 'LAT']]
        if Point([lon, lat]).within(sc_shape):
            inside_sc_or_not.append(True)
        else:
            inside_sc_or_not.append(False)

    points_sc_detailed = points_sc_crude[inside_sc_or_not]
    locs_ref_line = points_sc_detailed[['LAT', 'LON']].round(4)
    locs_ref_line.to_csv('all_locs.csv')

    return locs_ref_line

def fill_missing_locs(subset_file, original_file=None):
    if original_file is None:
        original_file = './all_locations/all_locs.csv'

    all_locs = pd.read_csv(original_file, index_col=0)
    subset_locs = pd.read_csv(subset_file, index_col=0).round(4)

    all_locs_list = all_locs.round(4).values.tolist()
    subset_locs_list = subset_locs[['LATITUDE','LONGITUDE']].values.tolist()

    zeros = []
    for loc in all_locs_list:
        if loc not in subset_locs_list:
            loc.append(0)
            zeros.append(loc)
    zeros_df = pd.DataFrame(zeros, columns=['LATITUDE','LONGITUDE', 'PRCP'] )
    df_filled = pd.concat([subset_locs, zeros_df], axis = 0)

    delta = df_filled.shape[0] - subset_locs.shape[0]
    print(f'{delta} more observations have been added to the dataset')
    print(f'dataset now has {df_filled.shape[0]} records')
    return df_filled

def batch_processor(folder):
    '''
    apply the fill_missing_locs function for every .csv file in given folder
    Args:
        folder: target folder that contains csv file.
    '''
    for file in os.listdir(folder):
        if file.endswith('csv'):

            old_file = os.path.join(folder, file)
            df_filled = fill_missing_locs(old_file)

            new_name = 'filled_' + file
            new_file = os.path.join(folder, new_name)
            df_filled.to_csv(new_file)

if __name__ == '__main__':
    test_run = fill_missing_locs('./rainfall_data_nc/20151003.csv')
    batch_processor('./rainfall_data_nc/')
