import os
import requests
import pandas as pd
from shutil import rmtree
from utility_functions import get_state_fullname

ATTRIBUTE_NUMBER = {'GAGE': '00065'}  #gage level
STAT_IDENTIFIER = {'MAX': '00001', 'MIN':'00002','MEAN':'00003'} #max, min, mean

class DailyFloodDataDownloader:
    def __init__(self, start, end, state, eight_digit_station_only=True, cap=None, verbose=False):
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
            eight_digit_station_only (boolean): if true, only use stream station and discard other stations which might be irrelelavent
            verbose (boolean): if true, more information will be given when downloading. Otherwise only progress will be reported.
            cap (int) for demo purpose, we can limit the number of locations
        Example:
            >>test = DailyFloodDataDownloader(start='2010-01-01', end='2010-01-02', state='SC')
            >>test.run()
        '''
        self.state = state if len(state) == 2 else get_state_fullname(state,
            reverse=True)
        self.verbose =verbose
        self.cap = cap
        self.job_list = []
        self.filename = self._make_filename(start=start, end=end, state=self.state)

        var_code = ATTRIBUTE_NUMBER['GAGE']
        _url_part_1 = f'https://waterdata.usgs.gov/nwis/dv?cb_{var_code}=on&format=rdb&site_no='
        _url_part_2 = f'&referred_module=sw&period=&begin_date={start}&end_date={end}'

        print('-'*40)
        print(f'start looking for all available stations in {state}')
        if eight_digit_station_only:
            print('only 8-digit station will be kept since they are more likely to be relevant')
            print('to turn off this option, set eight_digit_station_only=False')
        else:
            print('all locations, including wells and glaciers, will be examined.')
            print('this might make the process longer.')
            print('please turn this off by eight_digit_station_only=True unless you have reason to do otherwise')

        self.station_info_dict = self._get_stations(self.state, eight_digit_station_only=eight_digit_station_only)
        self.stations = list(self.station_info_dict.keys())

        print('finished generating locations, now start downloading files.')
        print(f'found {len(self.stations)} in {state}')
        if self.cap:
            print(f'the program is going to terminate after finding {self.cap} valid observations. To find all, set cap=None')
        print('-'*40)

        for station in self.stations:
            job = _url_part_1 + str(station) + _url_part_2
            self.job_list.append(job)

        self.temp_dir = os.path.join(os.getcwd(), 'temp_dir_flood')
        if not os.path.exists(self.temp_dir):
            os.mkdir(self.temp_dir)
        self.colnames_table = self.make_header()

    def _make_filename(self, start, end, state):
        start_ = ''.join(start.split('-'))
        end_ = ''.join(end.split('-'))
        name = '-'.join([state, start_, end_])
        return name + '.txt'

    def _get_stations(self, state, eight_digit_station_only):
        if len(state) > 2:
                state = get_state_fullname(state, reverse=True)#get acrynom
        state = state.lower()

        dict_of_stations = {}
        url = f'https://waterdata.usgs.gov/nwis/nwismap?state_cd={state}&format=sitefile_output&sitefile_output_format=rdb&column_name=agency_cd&column_name=site_no&column_name=station_nm&column_name=dec_lat_va&column_name=dec_long_va'

        for line in requests.get(url).text.split('\n'):
            if line.startswith('USGS'):
                siteid, name, latitude, longitude = line.split('\t')[1:5]
                dict_of_stations[siteid] = (name, latitude, longitude)

        if eight_digit_station_only:
            shorter_dict = {}
            for k, v in dict_of_stations.items():
                if len(k) == 8:
                    shorter_dict[k] = v
                else:
                    pass
            return shorter_dict
        return dict_of_stations

    def page_parser(self, station_url, additional_info, outfilename):
        '''
        parse the url and save the result in a txt file
        allow users to add additional information like latitude and longitude
        the additional information has to be a dict

        Note that if the file has 2 lines, the additional info will be
        appended to each of them.
        '''
        keywords = '\t'.join(list(additional_info.keys()))
        values = '\t'.join(list(additional_info.values()))

        try:
            content = requests.get(station_url).text
        except requests.exceptions.SSLError:
            if self.verbose:
                print('encountered a bad connection, will pass this one')
            return None

        obs = []; header = None

        if content is None:
            print(f'Got an empty page')
            return None

        textfile_name = '.'.join([outfilename, 'txt'])
        textfile_name = os.path.join(self.temp_dir, textfile_name)

        with open(textfile_name, 'w') as f:
            for line in content.split('\n'):
                is_obs = line.startswith('USGS')
                is_var_name = line.startswith('agency_cd')

                if is_var_name:
                    line = '\t'.join([keywords, line])
                    f.write(line+'\n')

                if is_obs:
                    line = '\t'.join([values, line])
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

    def make_header(self):
        look_up_table = {} # for renaming scheme
        for att_num in  ATTRIBUTE_NUMBER.items():
            for stat in STAT_IDENTIFIER.items():
                k1, v1 = att_num
                k2, v2 = stat
                new_key = ('_'.join([k1, k2]))
                new_v = ('_'.join([v1, v2]))
                look_up_table[new_key] = new_v

        header = {'SITENUMBER': 'site_no',
                  'STATIONNAME': 'name',
                  'LATITUDE':'latitude',
                  'LONGITUDE': 'longitude',
                  'DATE':'datetime',
                  }
                #'gage_max': '--006450--001'
        header.update(look_up_table)
        return header

    def file_formatter(self, filename, format_dict):
        '''
        format_dict gives the instruction of how to rename file
        renaming several variables and remove duplicate columns
        '''
        look_up_table = format_dict
        reversed_table = {v: k for k, v in look_up_table.items()}
        dataframe = pd.read_csv(filename, sep='\t', dtype={'site_no':str})

        for col in dataframe.columns:
            if not col.endswith(tuple(list(look_up_table.values()))):
                dataframe.drop([col], axis=1, inplace=True)
            else:
                try:
                    new_name = reversed_table[col]
                except KeyError:
                    list_of_keys = list(reversed_table.keys())
                    right_key = list(filter(col.endswith, list_of_keys))[0]
                    new_name = reversed_table[right_key]

                dataframe.rename(columns={col: new_name}, inplace=True)
        dataframe.to_csv(filename, index=False)

    def run(self):

        valid_station = 0

        for idx, (stationurl, stationid) in enumerate(zip(self.job_list, self.stations)):
            name, latitude, longitude = self.station_info_dict[stationid]
            additional_info_dict = {'name': name, 'latitude': latitude, 'longitude': longitude}
            filename = self.page_parser(station_url=stationurl,
                                        additional_info=additional_info_dict,
                                        outfilename=stationid)

            if filename:
                self.file_formatter(filename, self.colnames_table)
                valid_station = valid_station + 1
            if idx%10 == 0: #progress tracker reported every 10 station
                if self.cap:
                    print(f'number of stations processed: {idx}; number of stations valid: {valid_station}')
                    print(f'progress: {valid_station}/{self.cap}')
                else:
                    print(f'number of stations processed: {idx}; number of stations valid: {valid_station}')
                    print(f'progress {idx}/{len(self.job_list)}')

            if self.cap and valid_station >= self.cap:
                print(f'found {self.cap} stations already. teminating the program')
                break

        with open(self.filename, 'w') as file:
            header = ','.join(list(self.colnames_table.keys())) + '\n'
            file.write(header)
            for small_file in os.listdir(self.temp_dir):
                path_to_small_file = os.path.join(self.temp_dir, small_file)
                if path_to_small_file.endswith('.txt') and os.path.getsize(path_to_small_file):
                    with open(path_to_small_file) as sf:
                        next(sf)
                        file.write(sf.read())

        rmtree(self.temp_dir) #tear down
        print(f'the data have have been saved into the file {self.filename}.')
        print(f'the location of file is {os.getcwd()}')

if __name__ == '__main__':
    test = DailyFloodDataDownloader(start='2010-01-01', end='2016-12-31', state='SC', cap=None, eight_digit_station_only=True)
    test.run()

    # https://waterdata.usgs.gov/nwis/dv?cb_00065=on&format=rdb&site_no=02153051&referred_module=sw&period=&begin_date=2017-11-20&end_date=2018-11-20
