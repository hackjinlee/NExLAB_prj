import pandas as pd
import enum
import os
from random import shuffle, randint, uniform

import numpy as np
from torch.utils.data import Dataset
import torch.utils.data

from common import *


class DataPurpose(enum.Enum):
    Train = 'Train'
    Validation = 'Validation'
    Test = 'Test'
    Total = 'Total'


class TargetTypes(enum.Enum):
    Class2 = 'c2'
    Class3 = 'c3'
    Class4 = 'c4'
    Class6 = 'c6'


class SubtractionMode(enum.Enum):
    REF = 'Reference'
    MEAN = 'Mean'
    MIN = 'Minimum'
    NA = 'NotApplied'


class RepresentationMode(enum.Enum):
    ALL = 'AllSpectrum'
    MEAN = 'Mean'
    MED = 'Median'
    GROUP = 'Group'


class NormalizationMode(enum.Enum):
    LC = 'Local'
    GB = 'Global'
    NA = 'NotApplied'


class SpectrumDataset(Dataset):
    organ2class = {
        "GALLBLADDER": 0,
        "PANCREAS": 1,
        "STOMACH": 2,
        "COLORECTUM": 3,
        "BREAST": 4,
        "THYROID": 5,
    }
    organ2class4 = {
        "GALLBLADDER": 0,
        "PANCREAS": 1,
        "STOMACH": 2,
        "COLORECTUM": 2,
        "BREAST": 3,
        "THYROID": 3,
    }

    def __init__(self, data_purpose, fold_index=0, config=None):
        self.data_purpose = data_purpose
        self.fold_index = fold_index
        if config is None:
            config = ConfigManager().load()
        self.config = config

        # affine transformation 세팅
        transform_lst = []
        if config['use_scaling'].lower() == 'true':
            transform_lst.append(add_linear)
        if config['use_shifting'].lower() == 'true':
            transform_lst.append(shift_spectrum)
        if config['use_rotation'].lower() == 'true':
            transform_lst.append(rotate)
        self.transform_lst = transform_lst
        self.transform_prob = int(config["transform_prob"])
        self.target_type = TargetTypes(config["target_type"])

        # Data purpose 에 따라서 case 선택
        pp_dir = config["pp_dir"]
        case_info_path = config["case_info_path"]
        case_info = pd.read_csv(case_info_path)
        fold_info = case_info['fold'].values
        if data_purpose == DataPurpose.Train:
            indexes = [i for i in range(len(fold_info)) if fold_info[i] != fold_index]
            part_info = case_info.iloc[indexes, :]
        elif data_purpose == DataPurpose.Validation:
            indexes = [i for i in range(len(fold_info)) if fold_info[i] == fold_index]
            part_info = case_info.iloc[indexes, :]
        elif data_purpose == DataPurpose.Total:
            part_info = case_info

        # input data, label list 불러오기
        spectrum_lst, label_lst, info_lst = [], [], []
        id2info_dict, sid2size_dict = dict(), dict()
        wave_numbers = None
        for i, row in part_info.iterrows():
            # sid, name, sex, age, is_pdac, fold = row[:]
            id, sex, age, c_type, o_type = row[:5]
            c2_label = 0 if c_type.lower() == 'normal' else 1
            c3_label = c2_label if c_type.lower() != 'etc' else 2
            c4_label = self.organ2class4[o_type]
            c6_label = self.organ2class[o_type]

            csv_path = "%s/%03d.csv" % (pp_dir, id)
            if not os.path.exists(csv_path):
                print('error! there are no file at %s' % csv_path)
                continue

            wn, spectra = self.load_spectrum(csv_path)
            wave_numbers = wn if wave_numbers is None else wave_numbers
            for idx in range(spectra.shape[0]):
                spectrum_lst.append(spectra[idx, :])
                if self.target_type == TargetTypes.Class2:
                    label_lst.append(c2_label)
                elif self.target_type == TargetTypes.Class3:
                    label_lst.append(c3_label)
                elif self.target_type == TargetTypes.Class4:
                    label_lst.append(c4_label)
                elif self.target_type == TargetTypes.Class6:
                    label_lst.append(c6_label)

                info_lst.append([sex, age, o_type])
            sid2size_dict[id] = spectra.shape[0]
            id2info_dict[id] = [sex, age, o_type]

        self.spectrum_lst = spectrum_lst
        self.label_lst = label_lst
        self.id2info_dict = id2info_dict
        self.sid2size_dict = sid2size_dict
        self.wave_numbers = wave_numbers
        self.info_lst = info_lst
        if len(spectrum_lst[0].shape) == 1:
            self.input_shape = (1, len(spectrum_lst[0]))
        else:
            self.input_shape = spectrum_lst[0].shape

    def load_spectrum(self, csv_path):
        '''
        스펙트럼을 불러오고 전처리 적용.
        https://doaiacropolis.atlassian.net/wiki/spaces/AIRaman/pages/2131361800/22-01-18
        :param csv_path:
        :return:
        '''
        config = self.config
        subtraction_mode = SubtractionMode(config["SubtractionMode"])
        representation_mode = RepresentationMode(config["RepresentationMode"])
        normalization_mode = NormalizationMode(config["NormalizationMode"])
        min_val = float(config["lower_bound"])
        max_val = float(config["upper_bound"])
        group_num = int(config['group_num'])

        data = pd.read_csv(csv_path).values
        wave_numbers, spectra = data[:, 0], data[:, 1:]

        # phase 1 Affine Transformation
        if self.data_purpose == DataPurpose.Train:
            t_spectra = spectra.copy()
            for fnc in self.transform_lst:
                if randint(1, 100) <= self.transform_prob:
                    t_spectra = fnc(t_spectra)

            spectra = t_spectra

        # phase 2 reference spectrum subtraction
        ref_spectrum = np.zeros(data.shape[0])
        if subtraction_mode == SubtractionMode.REF:
            reference_path = config["reference_path"]
            ref_spectrum = pd.read_csv(reference_path).values[:, 1:]
            ref_spectrum = np.mean(ref_spectrum, axis=1)
        elif subtraction_mode == SubtractionMode.MEAN:
            ref_spectrum = np.mean(spectra, axis=1)
        elif subtraction_mode == SubtractionMode.MIN:
            ref_spectrum = np.min(spectra, axis=0)

        if subtraction_mode != SubtractionMode.NA:
            min_val = min_val - (max_val - min_val) // 2
            max_val = max_val - (max_val - min_val) // 2

        for col_idx in range(spectra.shape[1]):
            spectrum = spectra[:, col_idx] - ref_spectrum[col_idx]
            spectra[:, col_idx] = spectrum

        # phase 3 spectrum representation
        spectrum_lst = []
        if representation_mode == RepresentationMode.MEAN:
            spectrum = np.mean(spectra, axis=1)
            spectrum_lst.append(spectrum)
        elif representation_mode == RepresentationMode.MED:
            spectrum = np.median(spectra, axis=1)
            spectrum_lst.append(spectrum)
        elif representation_mode == RepresentationMode.ALL:
            for col_idx in range(spectra.shape[1]):
                spectrum_lst.append(spectra[:, col_idx])
        elif representation_mode == RepresentationMode.GROUP:
            e_idx = spectra.shape[1] - group_num + 1
            index_lst = list(range(spectra.shape[1]))
            if self.data_purpose == DataPurpose.Train:
                shuffle(index_lst)
            for si in np.arange(0, e_idx, group_num):
                indexes = index_lst[si:si+group_num]
                spectrum = spectra[:,  indexes].transpose(1, 0)
                spectrum_lst.append(spectrum)
        spectra = np.array(spectrum_lst)

        # phase 4 normalization
        if normalization_mode == NormalizationMode.LC:
            min_val, max_val = np.min(spectra), np.max(spectra)
            spectra = (spectra - min_val) / (max_val - min_val)
        elif normalization_mode == NormalizationMode.GB:
            spectra = (spectra - min_val) / (max_val - min_val)

        return wave_numbers, spectra

    def __getitem__(self, index):
        spectrum = self.spectrum_lst[index]
        label = self.label_lst[index]
        reshaped = np.reshape(spectrum, self.input_shape).astype(np.float32)
        label = np.array(label).astype(np.int32)
        info = self.info_lst[index]
        return reshaped, label, info

    def __len__(self):
        return len(self.label_lst)


def shift_spectrum(spectra, jump_int=2):
    '''
    https://doaiacropolis.atlassian.net/wiki/spaces/AIRaman/pages/1978695683/Data+Augmentation
    :param spectra:
    :param jump_int:
    :return:
    '''
    t_spectra = spectra.copy()
    for idx in range(spectra.shape[0]):
        t_spectrum = spectra[idx, :]
        jump = randint(-jump_int, jump_int)
        size = len(t_spectrum)
        s_spectrum = t_spectrum.copy()
        for trg_idx in range(size):
            src_idx = trg_idx + jump
            src_idx = 0 if src_idx < 0 else src_idx
            src_idx = size - 1 if src_idx >= size else src_idx

            t_spectrum[trg_idx] = s_spectrum[src_idx]

        t_spectra[idx, :] = t_spectrum
    return t_spectra


def add_linear(spectra, theta=0.5):
    '''
    https://doaiacropolis.atlassian.net/wiki/spaces/AIRaman/pages/1978695683/Data+Augmentation
    :param spectra:
    :param theta:
    :return:
    '''
    t_spectra = spectra.copy()
    for idx in range(spectra.shape[0]):
        spectrum = t_spectra[idx, :]

        size = len(spectrum)
        gap = (max(spectrum) - min(spectrum)) * theta * 0.5
        head = uniform(-gap, gap)
        tail = uniform(-gap, gap)
        a = (tail - head) / size

        for i in range(size):
            spectrum[i] = spectrum[i] + a * i + head

        t_spectra[idx, :] = spectrum

    return t_spectra


def rotate(spectra, rad=20):
    '''
    https://doaiacropolis.atlassian.net/wiki/spaces/AIRaman/pages/1978695683/Data+Augmentation
    :param spectra:
    :param rad:
    :return:
    '''
    t_spectra = spectra.copy()
    zero_idx = randint(0, spectra.shape[1] - 1)
    x_lst = [x - zero_idx for x in range(spectra.shape[1])]
    theta = np.radians(randint(-rad, rad))
    cos, sin = np.cos(theta), np.sin(theta)
    r_mat = np.array(((cos, -sin), (sin, cos)))
    for idx in range(spectra.shape[0]):
        spectrum = t_spectra[idx, :]
        s_mat = np.array((x_lst, spectrum))
        t_spectrum = np.dot(r_mat, s_mat)[1, :]
        t_spectra[idx, :] = t_spectrum
    return t_spectra


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        label_lst = []
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
            label_lst.append(label)

        # weight for each sample
        weights = [1.0 / label_to_count[label_lst[idx]]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.label_lst[idx]

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class SpectrumDatasetDiff(SpectrumDataset):
    def __init__(self, data_purpose, fold_index=0, config=None):
        self.data_purpose = data_purpose
        self.fold_index = fold_index
        if config is None:
            config = ConfigManager().load()
        self.config = config

        transform_lst = []
        if config['use_scaling'].lower() == 'true':
            transform_lst.append(add_linear)
        if config['use_shifting'].lower() == 'true':
            transform_lst.append(shift_spectrum)
        if config['use_rotation'].lower() == 'true':
            transform_lst.append(rotate)
        self.transform_lst = transform_lst

        pp_dir = config["pp_dir"]
        case_info_path = config["case_info_path"]
        train_indexes = config["train_indexes"].split(',')
        train_indexes = list(map(int, train_indexes))
        # print(train_indexes)

        case_info = pd.read_csv(case_info_path)
        fold_info = case_info['fold'].values
        if data_purpose == DataPurpose.Train:
            indexes = [i for i in range(len(fold_info)) if fold_info[i] in train_indexes]
            part_info = case_info.iloc[indexes, :]
        elif data_purpose == DataPurpose.Validation:
            indexes = [i for i in range(len(fold_info)) if fold_info[i] == fold_index]
            part_info = case_info.iloc[indexes, :]
        elif data_purpose == DataPurpose.Total:
            part_info = case_info

        spectrum_lst, label_lst, info_lst = [], [], []
        id2info_dict, sid2size_dict = dict(), dict()
        wave_numbers = None
        for i, row in part_info.iterrows():
            # sid, name, sex, age, is_pdac, fold = row[:]
            id, sex, age, p_type = row[:4]
            is_cancer = 1 if p_type.lower() == 'malignant' else 0

            csv_path = "%s/%03d.csv" % (pp_dir, id)
            if not os.path.exists(csv_path):
                print('error! there are no file at %s' % csv_path)
                continue

            wn, spectra = self.load_spectrum(csv_path)
            wave_numbers = wn if wave_numbers is None else wave_numbers
            for idx in range(spectra.shape[0]):
                spectrum_lst.append(spectra[idx, :])
                label_lst.append(is_cancer)
                info_lst.append([sex, age])
            sid2size_dict[id] = spectra.shape[0]
            id2info_dict[id] = [sex, age]

        self.spectrum_lst = spectrum_lst
        self.label_lst = label_lst
        self.id2info_dict = id2info_dict
        self.sid2size_dict = sid2size_dict
        self.wave_numbers = wave_numbers
        self.info_lst = info_lst
        self.input_shape = (1, len(spectrum_lst[0]))


def split_fold_with_stage(case_info_path, fold_num=5):
    '''
    환자정보 파일을 읽어 병기별로 fold 를 구분해줌.
    :param case_info_path:
    :param fold_num:
    :return:
    '''
    df = pd.read_csv(case_info_path)

    organ2info = dict()
    for idx, row in df.iterrows():
        pid, organ = row['pid'], row['organ']
        stage = row['stage']
        if stage != stage:
            stage = 'normal'

        stage = stage.lower()
        if 'iii' in stage or 'iib' in stage:
            stage = 3
        elif 'iia' in stage or 'ii' in stage:
            stage = 2
        elif 'i' in stage:
            stage = 1
        else:
            stage = 0

        if organ not in organ2info:
            organ2info[organ] = []
        organ2info[organ].append([pid, stage])

    pid2fold = dict()
    tag2cnt = dict()
    for organ, info_lst in organ2info.items():
        stage2pid = dict()
        for pid, stage in info_lst:
            if stage not in stage2pid:
                stage2pid[stage] = []
            stage2pid[stage].append(pid)

        stage_lst = list(set(stage2pid.keys()))
        fold = 0
        for s in stage_lst:
            pid_lst = stage2pid[s]
            shuffle(pid_lst)
            for pid in pid_lst:
                pid2fold[pid] = fold
                tag = '%s_%s' % (organ, s)
                if tag not in tag2cnt:
                    tag2cnt[tag] = np.zeros(fold_num)
                tag2cnt[tag][fold] += 1
                fold = 0 if fold + 1 == fold_num else fold + 1
    total_cnt = np.zeros(fold_num)
    for tag, cnt in tag2cnt.items():
        print('%s: %s' % (tag, cnt))
        total_cnt = total_cnt + cnt
    print('%s: %s' % ('\ntotal', total_cnt))

    fold_lst = np.zeros(df.shape[0])
    for idx, row in df.iterrows():
        pid = row['pid']
        fold = pid2fold[pid]
        fold_lst[idx] = fold

    result_dict = df.to_dict('list')
    result_dict['fold'] = fold_lst
    # output_path = './config/case_info_160.csv'
    pd.DataFrame(result_dict).to_csv(case_info_path, index=False)

