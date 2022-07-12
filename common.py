import configparser
import pickle
import os
from shutil import copyfile
from multiprocessing import Pool
from contextlib import closing


def save_obj(path, obj):
    '''
    pickle 파일로 출력
    :param path:
    :param obj:
    :return:
    '''
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    '''
    pickle 파일 로드
    :param path:
    :return:
    '''
    with open(path, 'rb') as f:
        return pickle.load(f)


class ConfigManager:
    DEFAULT_PATH = './config/config.ini'
    SECTION_DEFAULT = 'DEFAULT'
    config_path = None

    def __init__(self, file_name=DEFAULT_PATH):
        self.config_path = file_name

    def copy_file(self, output_path):
        copyfile(self.DEFAULT_PATH, output_path)

    def load(self):
        config_file = configparser.ConfigParser()
        config_file.read(self.config_path)

        return config_file[self.SECTION_DEFAULT]

    def save(self, config_dict, output_path=None):
        if output_path is None:
            output_path = self.config_path

        if output_path is None:
            return

        config_file = configparser.ConfigParser()

        for key in config_dict:
            val = None
            if type(config_dict[key]) == int:
                val = "%d" % config_dict[key]
            elif type(config_dict[key]) == float:
                val = "%f" % config_dict[key]
            elif type(config_dict[key]) == bool:
                if config_dict[key]:
                    val = 'True'
                else:
                    val = 'False'
            elif type(config_dict[key]) == str:
                val = config_dict[key]

            if '%s' in val:
                val = val.replace('%s', '%%s')

            if val:
                config_file[self.SECTION_DEFAULT][key] = val

        with open(output_path, 'w') as writeFile:
            config_file.write(writeFile)

    # 현재 설정 내용을 출력함
    def dump(self, config_dict):
        for key in config_dict:
            if type(config_dict[key]) == int:
                print("%s = %d[int]" % (key, config_dict[key]))
            elif type(config_dict[key]) == float:
                print("%s = %f[float]" % (key, config_dict[key]))
            elif type(config_dict[key]) == bool:
                if config_dict[key]:
                    print("%s = True[bool]" % key)
                else:
                    print("%s = False[bool]" % key)
            elif type(config_dict[key]) == str:
                print("%s = %s[str]" % (key, config_dict[key]))


class BatchProcessor:
    def __init__(self, input_dir, output_dir, input_fmt, output_fmt, process_num=8):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.input_fmt = input_fmt
        self.output_fmt = output_fmt
        self.process_num = process_num

    def batch_run(self, fnc, use_parallel=True):
        '''
        입력 디렉토리 내 파일에 대해서 fnc 을 적용한 뒤
        출력 디렉토리에 결과 출력.
        :param fnc:
        :param use_parallel:
        :return:
        '''
        input_dir, output_dir = self.input_dir, self.output_dir
        file_lst = os.listdir(input_dir)
        args_lst = []
        for name in file_lst:
            if not name.endswith(self.input_fmt):
                continue

            # 환자명 인식 후 3자리 정수값으로 출력하도록 설정
            input_path = '%s/%s' % (input_dir, name)
            base_name = os.path.basename(input_path).split('.')[0]
            if '_' in base_name:
                base_name = base_name.split('_')[1]
            base_name = '%03d' % int(base_name)
            output_path = '%s/%s%s' % (output_dir, base_name, self.output_fmt)

            args_lst.append((input_path, output_path))

        if use_parallel:
            with closing(Pool(self.process_num)) as p:
                result_lst = p.starmap(fnc, args_lst)
        else:
            result_lst = []
            for args in args_lst:
                result = fnc(*args)
                if result is not None:
                    result_lst.append(result)