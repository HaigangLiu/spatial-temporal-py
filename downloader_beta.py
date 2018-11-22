import re, os, requests
from bs4 import BeautifulSoup
from utility_functions import get_state_fullname
import pandas as pd

ATTRIBUTE_NUMBER = {'GAGE': '00065'}  #gage level
STAT_IDENTIFIER = {'MAX': '00001', 'MIN':'00002', 'MEAN': '00003'} #max, min, mean

# name = get_stations('South Carolina', short_version=True)
# https://waterdata.usgs.gov/nwis/dv?cb_00065=on&format=rdb&site_no=02153051&referred_module=sw&period=&begin_date=2017-11-20&end_date=2018-11-20

class DailyFloodDataDownloader:

    def __init__(self, start, end, state, short_version=True, verbose=False):

        self.state = state if len(state) == 2 else get_state_fullname(state,
            reverse=True)
        self.verbose =verbose
        self.job_list = []
        self.save_file_name = '_'.join([start, 'to',end])

        var_code = ATTRIBUTE_NUMBER['GAGE']
        _url_part_1 = f'https://waterdata.usgs.gov/nwis/dv?cb_{var_code}=on&format=rdb&site_no='
        _url_part_2 = f'&referred_module=sw&period=&begin_date={start}&end_date={end}'

        print('-'*40)
        print('start looking for all available stations in South Carolina')
        if short_version:
            print('only 8-digit station will be kept since they are more likely to be relevant')
            print('to turn off this option, set short_version=False')
        else:
            print('all locations, including wells and glaciers, will be examined.')
            print('this might make the process longer.')
        self.stations = self._get_stations(self.state, short_version=True)
        print('finished generating locations, now start downloading files.')
        print('-'*40)

        for station in self.stations:
            job = _url_part_1 + str(station) + _url_part_2
            self.job_list.append(job)

    def _get_stations(self, state, short_version):
        if len(state) > 2:
            state = get_state_fullname(state, reverse=True)#get acrynom
        url = f'https://waterdata.usgs.gov/nwis/inventory?state_cd={state.lower()}'
        soup_object = BeautifulSoup(requests.get(url).content, 'lxml')
        search_result = soup_object.find(id='stationTable')

        list_of_station_numbers = []
        for option in search_result.find_all('option'):
            station_num = option['value']
            list_of_station_numbers.append(station_num)
        if short_version: #only keep 8-digit locations
            list_of_station_numbers = [entry for entry in list_of_station_numbers
            if len(entry) == 8]
        return list_of_station_numbers

    def page_parser(self, job_url, jobname):
        '''
        Args: the station id (str)
        return: a dataframe of gage informaion
        '''
        jobname = str(jobname)
        content = requests.get(job_url).text
        obs = []; header = None

        if content is None:
            print(f'Got an empty page')
            return None

        textfile_name = '.'.join([jobname, 'txt'])
        textfile_name = os.path.join('./flood2', textfile_name)

        with open(textfile_name, 'w') as f:
            for line in content.split('\n'):
                is_obs = line.startswith('USGS')
                is_var_name = line.startswith('agency_cd')

                if is_var_name:
                    header = line
                    f.write(header+'\n')

                if is_obs:
                    f.write(line+'\n')

        if os.path.getsize(textfile_name):
            if self.verbose:
                print('file downloading finished')
            return textfile_name
        else:
            if self.verbose:
                print('no valid data found')
            os.remove(textfile_name)
            return None

    def file_formatter(self, filename):
        '''
        renaming several variables
        '''
        look_up_table = {} # for renaming scheme
        for att_num in  ATTRIBUTE_NUMBER.items():
            for stat in STAT_IDENTIFIER.items():
                k1, v1 = att_num
                k2, v2 = stat
                new_key = ('_'.join([k1, k2]))
                new_v = ('_'.join([v1, v2]))
                look_up_table[new_key] = new_v

        additional_info = {'DATE':'datetime', 'SITENUMBER': 'site_no'}
        look_up_table.update(additional_info)

        dataframe = pd.read_csv(filename, sep='\t', dtype={'site_no':str})
        for colname, colcode in look_up_table.items():
            for col in dataframe.columns:
                if col.endswith(colcode) and colname not in dataframe.columns:
                    dataframe.rename(columns={col: colname}, inplace=True)
                elif col.endswith(colcode) and colname in dataframe.columns:
                    print(f'source file has duplicate columns of {colname}')
                elif col.endswith('_cd'):
                    dataframe.drop([col], axis=1, inplace=True)
                else:
                    pass #delete all others
        dataframe.to_csv(filename)

    def run(self):

        for idx, job in enumerate(self.job_list):
            filename = self.page_parser(job, jobname=idx)
            if filename:
                self.file_formatter(filename)
            if idx%20 == 0: #progress tracker
                print(f'finished processing {idx}/{len(self.job_list)}')

        with open(save_file_name, 'w') as file:
            header = ','.join(['SITENUMBER','DATE', 'GAGE_MAX', 'GAGE_MIN',
           'GAGE_MEAN\n'])
            file.write(header)
            for small_file in os.listdir('./flood2'):
                path_to_small_file = os.path.join('./flood2', small_file)
                if path_to_small_file.endswith('.txt') and os.path.getsize(path_to_small_file):
                    with open(path_to_small_file) as sf:
                        next(sf)
                        file.write(sf.read())
        print(f'the data have have been saved into the file {self.save_file_name}.')

if __name__ == '__main__':
    test = DailyFloodDataDownloader(start='2010-01-01', end='2010-01-02', state='SC')
    test.run()
    # print(test.job_list[1])

    # test.page_parser(test.job_list[1])
    # # print(s)


