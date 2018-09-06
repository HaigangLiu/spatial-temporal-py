import os
from SampleDataLoader import load_flood_data
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

def water_shed_finder(data_frame, watershed_file=None, remove_singular=True):
    '''
    for each location, find which watershed it belongs to
    Args:
        data_frame (pandas dataframe) : Must have a LONGTITUDE column
            and a LATITUDE column
        watershed_file (string):  usually a file with extension shp, if not
            specified, the default watershed file: HUC8 in South Caorolina, will be
            used.
        remove_singular (boolean): if true, the function will get rid of
        locations without neighbors.
    return
        A new dataframe with an additional column called WATERSHED
    '''
    if watershed_file is None:
        watershed_file = 'data/shape_file/hydrologic_HUC8_units/wbdhu8_a_sc.shp'
        watershed_file = os.path.join(os.getcwd(), watershed_file)

    huc8_units = gpd.read_file(watershed_file)
    water_shed_info = huc8_units[['NAME', 'geometry']]

    names = []
    df = data_frame.copy()

    locs = df[['LONGITUDE', 'LATITUDE']]
    for _, loc in locs.iterrows():
        for _, rows in water_shed_info.iterrows():
            name, polygon = rows
            if Point(loc).within(polygon):
                names.append(name)
                break
        else:
            names.append('HUC Not Defined')
    df['WATERSHED'] = names

    if remove_singular:
        number_of_obs = df.groupby(['WATERSHED']).count()['SITENUMBER']
        singular_huc_areas = number_of_obs[number_of_obs==1].index
        df = df[~df.WATERSHED.isin(singular_huc_areas)]
    return df

def neighbor_finder(dataframe, column):
    watershed_list = dataframe[column]
    neighbor_matrix = [] # an n by n matrix, 1 for adjcent, 0 for not.
    for entry in watershed_list:
        w = [0 if entry!=comp else 1 for comp in watershed_list]
        neighbor_matrix.append(w)
    neighbor_matrix = np.array(neighbor_matrix)

    adjcent_matrix = [] # a list of neighbors for each loc
    weight_matrix = []
    for idx, row in enumerate(neighbor_matrix):
        mask = np.argwhere(row == 1).ravel().tolist()
        mask.remove(idx) #delete the location itself.
        adjcent_matrix.append(mask)
        weight_matrix.append([1]*len(mask))

    return neighbor_matrix, adjcent_matrix, weight_matrix


def matrix_padding(input_list, max_padding=None):

    '''
    construct the matrix of W by making sure that every
    row has the same length.
    args:
        input list (list): a list of spatial information for each location
        max padding (int): the length of padded row. Must be equal or longer than the longest row.
    '''
    max_vector_length = max([len(row) for row in input_list])
    if max_padding is None:
        max_padding = max_vector_length
    elif max_padding > max_vector_length:
        pass
    else:
        raise ValueError('max_padding must be equal or greater than the longest row')

    input_list_copy = input_list.copy()
    for row in input_list_copy:
        while len(row) < max_padding:
            row.append(0)

    return np.array(input_list_copy)


if __name__ == '__main__':
    daily_flood = load_flood_data(option = 'daily')#usage
    df_with_watershed_assigned = water_shed_finder(daily_flood)
    print(df_with_watershed_assigned)
    NM, AM, WM = neighbor_finder(df_with_watershed_assigned, 'WATERSHED')

    padded_weights = matrix_padding(AM)
    padded_adj = matrix_padding(WM)
