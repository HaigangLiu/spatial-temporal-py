import os
import re
import requests

from datetime import date, timedelta
import concurrent.futures
import tarfile
#example_link
#https://water.weather.gov/precip/archive/2014/01/01/nws_precip_1day_observed_shape_20140101.tar.gz

class nwsDataDownloader:
    '''
    Downloading Precipitation dataset from national weather service (NWS)
    The available dates ranges from 01/01/2005 to 06/27/2017
    Args:
        local_loc (string): the local dir to store data
        start (string): starting date: e.g. '1990-01-01'
        end (string): ending date: e.g. '1990-01-30'
    '''
    def __init__(self, local_loc, start, end):
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

        web_file_name = f'{year_}/{month_}/{date_}/nws_precip_1day_observed_shape_{date_nospace}.tar.gz'
        local_file_name = date_nospace + '.tar.gz'

        link_in = os.path.join(web_repo_loc, web_file_name)
        dir_out = os.path.join(local_loc, local_file_name)

        return link_in, dir_out

    @staticmethod
    def file_download_and_upack(in_and_out):

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
        start_downloading = executor.map(nwsDataDownloader.file_download_and_upack, in_and_outs)




if __name__ == '__main__':

    local_loc_ = '/Users/haigangliu/SpatialTemporalBayes/rainfall_data_nc'
    from_date = '2015-06-01'; to_date = '2016-06-01'

    download_handler = nwsDataDownloader(local_loc_, from_date, to_date).run()

