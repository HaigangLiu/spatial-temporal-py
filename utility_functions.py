import numpy as np
import pandas as pd

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
        raise ValueError('the only accepted input type is pandas dataframe')
        return None


