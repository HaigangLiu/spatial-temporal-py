import re, os, requests
from bs4 import BeautifulSoup
from utility_functions import get_state_fullname
import pandas as pd

ATTRIBUTE_NUMBER = {'GAGE': '00065'}  #gage level
STAT_IDENTIFIER = {'MAX': '00001', 'MIN':'00002', 'MEAN': '00003'} #max, min, mean

class DailyFloodDataDownloader:
    def __init__(self, start, end, state, short_version=True, verbose=False):

        '''
        download flood data for given state within a given time frame
        1. first go to nws website to grab a list of available stations in that area
        2. generate a url
        3. parse that url and clean the data, which are written to a file
        4. merge all files for all location in that state

        Args:
            start (string): the starting time, format: '1990-01-01'
            end (string): the end time, format: '1990-01-01'
            state (string): the name or acrynom for that state
                all of these will work: 'SC', 'South Carolina', 'sc'
            short_version (boolean): if true, only use stream station and discard other stations which might be irrelelavent
            verbose (boolean): if true, more information will be given when downloading. Otherwise only progress will be reported.

        Example:
            >>test = DailyFloodDataDownloader(start='2010-01-01', end='2010-01-02', state='SC')
            >>test.run()
        '''
        self.state = state if len(state) == 2 else get_state_fullname(state,
            reverse=True)
        self.verbose =verbose
        self.job_list = []
        self.save_file_name = '_'.join([start, 'to',end]) + '.txt'

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

        self.station_info_dict = self._get_stations(self.state, short_version=True)
        self.stations = list(self.station_info_dict.keys())
        print('finished generating locations, now start downloading files.')
        print('-'*40)

        for station in self.stations:
            job = _url_part_1 + str(station) + _url_part_2
            self.job_list.append(job)


    def _get_stations(self, state, short_version):
        if len(state) > 2:
                state = get_state_fullname(state, reverse=True)#get acrynom
        state = state.lower()

        dict_of_stations = {}
        url = f'https://waterdata.usgs.gov/nwis/nwismap?state_cd={state}&format=sitefile_output&sitefile_output_format=rdb&column_name=agency_cd&column_name=site_no&column_name=station_nm&column_name=dec_lat_va&column_name=dec_long_va'

        for line in requests.get(url).text.split('\n'):
            if line.startswith('USGS'):
                siteid, name, latitude, longitude = line.split('\t')[1:5]
                dict_of_stations[siteid] = (name, latitude, longitude)

        if short_version:
            shorter_dict = {}
            for k, v in dict_of_stations.items():
                if len(k) == 8:
                    shorter_dict[k] = v
                else:
                    pass
            return shorter_dict
        return dict_of_stations

    def page_parser(self, station_url, stationid):
        '''
        Args: the station id (str)
        return: a dataframe of gage informaion
        '''

        name, latitude, longitude = self.station_info_dict[stationid]
        stationid = str(stationid)
        content = requests.get(station_url).text
        obs = []; header = None

        if content is None:
            print(f'Got an empty page')
            return None

        textfile_name = '.'.join([stationid, 'txt'])
        textfile_name = os.path.join('./flood2', textfile_name)

        with open(textfile_name, 'w') as f:
            for line in content.split('\n'):
                is_obs = line.startswith('USGS')
                is_var_name = line.startswith('agency_cd')

                if is_var_name:
                    header = line
                    header = '\t'.join(['name', 'latitude', 'longitude', header])
                    print(header)
                    f.write(header+'\n')

                if is_obs:
                    line = '\t'.join([name, latitude, longitude, line])
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
        renaming several variables and remove duplicate columns
        '''
        look_up_table = {} # for renaming scheme
        for att_num in  ATTRIBUTE_NUMBER.items():
            for stat in STAT_IDENTIFIER.items():
                k1, v1 = att_num
                k2, v2 = stat
                new_key = ('_'.join([k1, k2]))
                new_v = ('_'.join([v1, v2]))
                look_up_table[new_key] = new_v

        additional_info = {'DATE':'datetime',
                           'SITENUMBER': 'site_no',
                           'LATITUDE':'latitude',
                           'LONGITUDE': 'longitude',
                           'STATIONNAME': 'name'}

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
                    pass
        dataframe.to_csv(filename, index=False)
        #index will be detemined after combining all small files

    def run(self):
        for idx, (stationurl, stationid) in enumerate(zip(self.job_list, self.stations)):
            filename = self.page_parser(station_url=stationurl, stationid=stationid)
            if filename:
                self.file_formatter(filename)
            if idx%20 == 0: #progress tracker
                print(f'finished processing {idx}/{len(self.job_list)}')

        with open(self.save_file_name, 'w') as file:
            header = ','.join(['SITENNAME','LATITUDE','LONGITUDE', 'SITENUMBER','DATE', 'GAGE_MAX', 'GAGE_MIN',
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

    # test.page_parser(test.job_list[1])
    # # print(s)



    # name = get_stations('South Carolina', short_version=True)
    # https://waterdata.usgs.gov/nwis/dv?cb_00065=on&format=rdb&site_no=02153051&referred_module=sw&period=&begin_date=2017-11-20&end_date=2018-11-20
