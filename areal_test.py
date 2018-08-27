import os
import pysal as ps
import geopandas as gpd
from shapely.geometry import Point
from SampleDataLoader import load_flood_data, load_rainfall_data

def convert_to_geopanda(df, crs=None):
    '''
    covert a dataframe to a geo pandas dataframe
    Args:
    df (dataframe): a pandas dataframe, must include a column called LATITUDE and another column called LONGITUDE.
    crs (dict, optional): specify the mapping
    Return:
    a geo-pandas dataframe, with an additional column 'geometry'.
    '''
    latitude = df['LATITUDE']
    longitude = df['LONGITUDE']

    geometry = [Point(xy) for xy in zip(longitude, latitude)]
    df = df.drop(['LATITUDE', 'LONGITUDE'], axis=1)

    if crs is None:
        crs = {'init': 'epsg:4326'}

    geopandas_df = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
    return geopandas_df

def fill_area_with_point_data(areal_data, point_data, column_name):

    '''
    Args:
    areal_data (geopandas dataframe): a geo-pandas dataframe with areal
     information (like water shed, or county or states)
    point (geopandas dataframe): a geo-pandas dataframe with point-reference infomation
    column name: (string): specifies which column to aggragate

    Return:
    the areal_data dataframe with average rainfall data in each areal.
    '''
    areal_data_copy = areal_data.copy()
    points = point_data['geometry']
    areas = areal_data['geometry']

    results = []
    for area in areas:
        filtered_points = point_data[points.within(area)]
        results.append(filtered_points[column_name].mean())

    areal_data_copy = areal_data_copy.assign(**{column_name: results})
    return areal_data_copy

if __name__ == '__main__':

    rainfall_daily = load_rainfall_data('daily')
    flood_daily = load_flood_data('daily')

    rain_geo_pandas = convert_to_geopanda(rainfall_daily)
    flood_geo_pandas = convert_to_geopanda(flood_daily)

    huc8_units_file ='/Users/haigangliu/SpatialTemporalBayes/data/shape_file/hydrologic_HUC8_units/wbdhu8_a_sc.shp'
    huc8_units = gpd.read_file(huc8_units_file)
    areal_data = fill_area_with_point_data(huc8_units, rain_geo_pandas, 'PRCP')
    areal_data = fill_area_with_point_data(areal_data, flood_geo_pandas, 'GAGE_MAX')

    test_dataset = areal_data[['NAME', 'geometry', 'PRCP', 'GAGE_MAX']]
