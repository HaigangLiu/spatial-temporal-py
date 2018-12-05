import requests
from utility_functions import get_state_fullname
from collections import defaultdict

class DailyFloodDataDownloader:

    def __init__(self, start, end, state, eight_digit_station_only=True, cap=None, verbose=False, filename=None, attr_code='00065', header=None):
        '''
        download flood data for given state within a given time frame
        recipes:
            1. first go to nws website to grab a list of available stations in that area
            2. generate a url for each location in that area
            3. parse that url and clean the data, which are written to a dictionary
            4. write all dictionaries to a file with given name

        Args:
            start (string): the starting time, format: '1990-01-01'
            end (string): the end time, format: '1990-01-01'
            state (string): the name or acrynom for that state
                all of these will work: 'SC', 'South Carolina', 'sc'
            eight_digit_station_only (boolean): if true, only use stream station and discard other stations which might be irrelelavent
            verbose (boolean): if true, more information will be given when downloading. Otherwise only progress will be reported.
            cap (int) for demo purpose, we can limit the number of locations.
            Program will stop after cap is reached.

            verbose (boolean): more downloading information will be printed if
            set to True
            filename (string): the path of target file.
            attr_code (string): the code

        Example:
            >>test = DailyFloodDataDownloader(start='2010-01-01', end='2010-01-02', state='SC')
            >>test.run()
        '''
        self.state = state if len(state) == 2 else get_state_fullname(state, reverse=True)
        self.verbose =verbose
        self.cap = cap
        self.job_list = []

        if filename is None:
            self.filename = self._make_filename(start=start, end=end, state=self.state)
        else:
            self.filename = filename

        attr_code = str(attr_code)
        _url_part_1 = f'https://waterdata.usgs.gov/nwis/dv?cb_{attr_code}=on&format=rdb&site_no='
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

        self.station_info_dict = self.get_stations(self.state, eight_digit_station_only=eight_digit_station_only)
        self.stations = list(self.station_info_dict.keys())

        print('finished generating locations, now start downloading files.')
        print(f'found {len(self.stations)} in {state}')
        if self.cap:
            print(f'the program is going to terminate after finding {self.cap} valid observations. To find all, set cap=None')
        print('-'*40)

        for station in self.stations:
            job = _url_part_1 + str(station) + _url_part_2
            self.job_list.append(job)

        self.header = None #user can customize it
        if attr_code != '00065' and self.header is None:
            raise ValueError(f'a different attribute code detected: {attr_code}. You should  assign a corresponding header as well.')

    def _make_filename(self, start, end, state):
        start_ = ''.join(start.split('-'))
        end_ = ''.join(end.split('-'))
        name = '-'.join([state, start_, end_])
        return name + '.txt'

    def make_header(self):
        if self.header is None: #default ones
            header = {'STATIONNAME': 'name',
                      'LATITUDE':'latitude',
                      'LONGITUDE': 'longitude',
                      'SITENUMBER': 'site_no',
                      'DATE':'datetime',
                      'GAGE_MAX': '00065_00001',
                      'GAGE_MIN': '00065_00002',
                      'GAGE_MEAN': '00065_00003'}
            self.header = {v: k for k, v in header.items()}
        return self.header

    def get_stations(self, state, eight_digit_station_only):
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

    def page_parser(self, station_url, additional_info=None):
        '''
        parse the url and save the result into a default dict
        allow users to add additional information like latitude and longitude
        the additional information has to be a dict

        Note that if the file has 2 lines, the additional info will be
        appended to each of them.
        '''
        try:
            content = requests.get(station_url).text
        except requests.exceptions.SSLError:
            if self.verbose:
                print('encountered a bad connection, will pass this one')
            return None

        if content:
            content = content.split('\n')
        else:
            print(f'Got an empty page')
            return None

        output_dict = defaultdict(list)
        counter_num_days = 0
        header_not_set = True
        keys = None

        while content:
            line = content.pop(0)
            if header_not_set and line.startswith('agency_cd'):
                keys = line.split('\t')
                for key in keys:
                    output_dict[key]  #initialize
                header_not_set = False #only check once

            elif line.startswith('USGS'):
                if keys:
                    values = line.split('\t')
                    for key, value in zip(keys, values):
                        output_dict[key].append(value)
                    counter_num_days += 1
                else:
                    raise ValueError('content comes before header. Should not happen')
            else:
                pass

        if counter_num_days:
            print(f'found {counter_num_days} days')
            if additional_info:
                for key, value in additional_info.items():
                    value_ = [value]*counter_num_days
                    output_dict[key].extend(value_)
            return output_dict
        else:
            if self.verbose:
                print('no valid data found')
            return None

    def file_formatter(self, result_dict, formatter):
        '''
        match differet web page dynamically
        only need target data column name to match tails (endswith)
        return a one to many dict
        '''
        translated_header = defaultdict(list) #different for each file
        for key in result_dict.keys():
            for k, v in formatter.items():
                if key.endswith(k):
                    translated_header[v].append(key)

        formatted_dict = defaultdict(list)
        for key in translated_header.keys():
            values = translated_header[key]

            if len(values) >= 2:
                pooled_list = [] #only record the first viable value
                multiple_cols = [result_dict[value] for value in values]
                for entry in zip(*multiple_cols): #[1,2,3]
                    for val in entry:
                        if val:
                            pooled_list.append(val)
                            break
                    else:
                        pooled_list.append('-9999')

                formatted_dict[key].extend(pooled_list)
            elif len(values) == 1:
                formatted_dict[key].extend(result_dict[values[0]])
            else:
                assert False, 'should not happen'
        return formatted_dict

    def run(self):
        valid_station = 0
        if self.header is None:
            self.make_header()

        no_header_written = True
        with open(self.filename, 'w') as file:
            for idx, (stationurl, stationid) in enumerate(zip(self.job_list, self.stations)):

                name, latitude, longitude = self.station_info_dict[stationid]
                additional_info_dict = {'name': name, 'latitude': latitude, 'longitude': longitude}
                file_dict = self.page_parser(station_url=stationurl,
                                             additional_info=additional_info_dict)
                if file_dict:
                    file_dict = self.file_formatter(file_dict, self.header)
                    valid_station = valid_station + 1

                if idx%10 == 0: #progress tracker reported every 10 station
                    if self.cap:
                        print(f'number of stations processed: {idx}; number of stations valid: {valid_station}')
                        print(f'progress: {valid_station}/{self.cap}')
                    else:
                        print(f'number of stations processed: {idx}; number of stations valid: {valid_station}')
                        print(f'progress {idx}/{len(self.job_list)}')

                if file_dict:
                    if len(file_dict) == len(self.header):
                        if no_header_written:
                            header = '\t'.join(list(file_dict.keys()))
                            file.write(header+'\n')
                            no_header_written = False

                        for value in zip(*file_dict.values()):
                            file.write('\t'.join(value) + '\n')

                if self.cap and valid_station >= self.cap + 1:
                    print(f'found {self.cap} stations already. teminating the program')
                    break

if __name__ == '__main__':
    test = DailyFloodDataDownloader(start='2010-01-01', end='2016-12-31',
        state='SC', cap=None, eight_digit_station_only=False,
        filename='test.txt')
    s = test.run()

    # https://waterdata.usgs.gov/nwis/dv?cb_00065=on&format=rdb&site_no=02153051&referred_module=sw&period=&begin_date=2017-11-20&end_date=2018-11-20
