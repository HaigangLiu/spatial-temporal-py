import os
import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta

#example_link
# ='https://water.weather.gov/precip/archive/2015/10/02/nws_precip_conus_20151002.nc'
class NationWeatherServiceDataDownloader:
    '''
    Downloading Precipitation dataset from national weather service (NWS)
    The available dates ranges from 01/01/2005 to 06/27/2017
    '''
    def __init__(self, web_loc, local_loc, start, end):
        self.web_loc = 'https://water.weather.gov/precip/archive'
        self.local_loc = local_loc
        self.start = start
        self.end = end

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
        make the link, and the dir to store the file locally
        Args:
            web_repo_loc (string): the link for the web archive with rainfall data
            local_loc (string): the dir to store data
            date_token (string): the date of data user wish to retrieve
        '''
        year_, month_, date_ = date_token.split('-')
        date_nospace = ''.join([year_, month_, date_])

        web_file_name = f'{year_}/{month_}/{date_}/nws_precip_conus_{date_nospace}.nc'
        local_file_name = date_nospace + '.nc'

        link_in = os.path.join(web_repo_loc, web_file_name)
        dir_out = os.path.join(local_loc, local_file_name)

        return link_in, dir_out

    @staticmethod
    def file_download(input_link, name_out):
        content = requests.get(input_link)
        with open(name_out, 'wb') as f:
            for chunk in content.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        file_size = os.path.getsize(name_out)
        if file_size:
            return 0
        else:
            raise ValueError('the file is empty. Check the link')
            return 1

    def run(self):
        job_list = FileDownloader.range_handler(self.start, self.end)
        links_in = []; files_out = []

        for date_ in job_list:
            link_in, dir_out = FileDownloader._make_link(self.web_loc,
                self.local_loc, date_)
            FileDownloader.file_download(link_in, dir_out)
            print(f'successfully downloaded the rainfall data for {date_}')

if __name__ == '__main__':

    local_loc_ = '/Users/haigangliu/SpatialTemporalBayes/rainfall_data_nc'
    from_date = '2015-06-01'
    to_date = '2016-06-01'

    download_handler = FileDownloader(web_loc_, local_loc_, from_date, to_date).run()
