import os, re, tarfile, requests
from datetime import date, timedelta
import concurrent.futures
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

class nwsDataDownloader:
    '''
    Downloading Precipitation dataset from national weather service (NWS)
    The available dates ranges from 01/01/2005 to 06/27/2017
    Args:
        local_loc (string): the local dir to store data
        start (string): starting date: e.g. '1990-01-01'
        end (string): ending date: e.g. '1990-01-30'
        rm_na (bool): if true, all locations with missing data will be removed
        var_name (str): the name of the variable of interest.
        region (polygon file, optional): the polygon file specifes the area of interest
            default value is south carolina
        to_pd (boolean): if true, the data will be converted to pandas dataframe before output.
    '''
    def __init__(self, local_loc, start, end, rm_na=True, var_name='GLOBVALUE', region=None, to_pd=True):
        self.web_loc = 'https://water.weather.gov/precip/archive'
        self.local_loc = local_loc
        self.start = start
        self.end = end
        self.rm_na = rm_na #keep missing value or not
        self.var_name = var_name #response variable name
        self.region = region # the region user is interested in
        self.to_pd = to_pd #output a pandas dataframe

    @staticmethod
    def range_handler(start_date, end_date):
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

    @staticmethod
    def _make_link(web_repo_loc, local_loc, date_token):
        '''
        generate the url for data, and local dir to store the data.
        Args:
            web_repo_loc (string): the link for the web archive with rainfall data
            local_loc (string): the dir to store data
            date_token (string): the date of data user wish to retrieve
        Return
            the url and the local dir.
        '''
        year_, month_, date_ = date_token.split('-')
        date_nospace = ''.join([year_, month_, date_])

        web_file_name = f'{year_}/{month_}/{date_}/nws_precip_1day_observed_shape_{date_nospace}.tar.gz'
        local_file_name = date_nospace + '.tar.gz'

        link_in = os.path.join(web_repo_loc, web_file_name)
        dir_out = os.path.join(local_loc, local_file_name)

        return link_in, dir_out

    def post_process(self, shp_file):

        region_mask = []

        if self.region is None:
            print('No polygon file specified.')
            print('by default, all locations within south carolina will be retained')

            sc_dir = os.path.join(os.getcwd(),'data/shape_file/south_carolina/tl_2010_45_state10.shp')
            self.region = gpd.read_file(sc_dir)['geometry'][0]

        geopandas_file = gpd.read_file(shp_file)

        if self.rm_na: #remove na and negative values
            before = geopandas_file.shape[0]
            na_mask = np.isnan(geopandas_file[self.var_name])

            geopandas_file = geopandas_file[~na_mask]
            geopandas_file = geopandas_file[geopandas_file[self.var_name]>=0]
            after = geopandas_file.shape[0]

            print(f'removed missing data. {before-after} mssing values are removed')

        locs = geopandas_file[['LON', 'LAT']]
        for _, spatial_point in locs.iterrows():
            region_mask.append(Point(spatial_point).within(self.region))

        print(f'find {sum(region_mask)} values within in the specified region')
        output = geopandas_file[region_mask]

        if self.to_pd:
            numpy_array = output[['LAT', 'LON', self.var_name]].values
            output = pd.DataFrame(numpy_array, columns=['LATITUDE', 'LONGITUDE', 'PRCP'])
        return output

    def file_download_and_upack(self, in_and_out):

        input_link, name_out = in_and_out
        content = requests.get(input_link)
        with open(name_out, 'wb') as f:
            for chunk in content.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        file_size = os.path.getsize(name_out)

        if file_size:
            print(f'finished writing data to file {name_out}')

            dir_, fname = os.path.split(name_out)
            target_folder_name = re.findall('\d{8}', fname)[0]#8-number, should be date

            with tarfile.open(name_out) as tar:
                tar.extractall(os.path.join(dir_, target_folder_name))

            os.remove(name_out) #tear down

            abs_target_folder = os.path.join(dir_, target_folder_name)
            for file in os.listdir(abs_target_folder):
                if file.endswith('.shp'):
                    file_abs = os.path.join(abs_target_folder, file)
                    output_from_post_process = self.post_process(file_abs)

                    csv_name = abs_target_folder + '.csv'
                    output_from_post_process.to_csv(csv_name)
            return 0
        else:
            raise ValueError('the file is empty. Check the link')
            return 1

    def run(self):
        job_list = nwsDataDownloader.range_handler(self.start, self.end)
        in_and_outs = []

        for date_ in job_list:
            link_in, dir_out = nwsDataDownloader._make_link(self.web_loc,self.local_loc, date_)
            in_and_outs.append([link_in, dir_out])

        executor = concurrent.futures.ProcessPoolExecutor(max_workers = 8)
        start_downloading = executor.map(self.file_download_and_upack, in_and_outs)

if __name__ == '__main__':
    #example_link
    #https://water.weather.gov/precip/archive/2014/01/01/nws_precip_1day_observed_shape_20140101.tar.gz

    local_loc_ = '/Users/haigangliu/SpatialTemporalBayes/rainfall_data_nc'
    from_date = '2015-06-01'; to_date = '2016-06-01'
    download_handler = nwsDataDownloader(local_loc_, from_date, to_date).run()
