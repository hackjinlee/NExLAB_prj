import os
import pandas as pd
import numpy as np


def convert_txt2csv(input_path, output_path):
    '''
    '.txt' format (raw data)을 '.csv' format 으로 변환
    (optional) removing the outlier
    :param input_path:
    :param output_path:
    :return:
    '''
    df = pd.read_csv(input_path, delimiter='\t', header=0)

    wn_lst = []
    time2inten_dict = dict()
    for idx, row in df.iterrows():
        time, wave_num, intensity, _, _ = row

        if time == 0:
            wn_lst.append(wave_num)

        if time not in time2inten_dict:
            time2inten_dict[time] = []
        time2inten_dict[time].append(intensity)

    # 데이터 역순으로 입력
    result_dict = {'WaveNumber':wn_lst[::-1]}
    key_lst = sorted(list(time2inten_dict.keys()))
    for key in key_lst:
        tag = 'time%s' % int(key)
        intensity_arr = np.array(time2inten_dict[key][::-1])
        intensity_arr = remove_outlier(intensity_arr)

        result_dict[tag] = intensity_arr
    pd.DataFrame(result_dict).to_csv(output_path, index=False)
    base_name = os.path.basename(output_path)
    print('..%s has been exported' % base_name)


def remove_outlier(intensity_arr, win_size=100, win_period=50, alpha=5):
    '''
    outlier 제거
    :param intensity_arr: spectrum
    :param win_size: window size
    :param win_period: window period (update)
    :param alpha: for calculation of the upper/lower boundary
    :return: spectrum (outlier removed)
    '''
    size = len(intensity_arr)
    remove_dict = dict()
    result_lst = intensity_arr.copy()
    si_lst = list(range(0, size - win_size, win_period))
    if si_lst[-1] + win_size < size:
        si_lst.append(size-win_size)

    for si in si_lst:
        ei = si + win_size if si + win_size < size else size
        indeces = list(range(si, ei))
        ri_lst, rm_cnt = [], 100
        while rm_cnt > 0:
            rm_cnt = 0
            mn, sd = np.mean(intensity_arr[indeces]), np.std(intensity_arr[indeces])
            upper, lower = mn + alpha * sd, mn - alpha * sd
            for i in range(si, ei):
                if i in ri_lst:
                    continue

                if intensity_arr[i] > upper or intensity_arr[i] < lower:
                    indeces.remove(i)
                    ri_lst.append(i)
                    rm_cnt += 1

        if len(ri_lst) > 0:
            for ri in ri_lst:
                remove_dict[ri] = mn

    for idx, mn in remove_dict.items():
        result_lst[idx] = mn
    return result_lst


def batch_merge_csv(dir1, dir2, out_dir):
    wn_tag = 'WaveNumber'
    n_lst = os.listdir(dir1)
    for name in n_lst:
        n_only = name.split('.')[0]
        p1 = '%s/%s' % (dir1, name)
        p2 = '%s/%s' % (dir2, name)

        if os.path.exists(p1) and os.path.exists(p2):
            dict1 = pd.read_csv(p1).to_dict('list')
            dict2 = pd.read_csv(p2).to_dict('list')
            merged = dict()
            merged[wn_tag] = dict1[wn_tag]
            c_cnt = 0
            for key, val1 in dict1.items():
                if key == wn_tag:
                    continue

                if key not in dict2:
                    print('error on %s! %s has failed to find!' % (n_only, key))
                    continue

                val2 = dict2[key]
                k1, k2 = '%s_1' % key, '%s_2' % key
                merged[k1] = val1
                merged[k2] = val2
                c_cnt += 1
            out_p = '%s/%s' % (out_dir, name)
            pd.DataFrame(merged).to_csv(out_p, index=False)

            print('... %s has been merged. (%s)' % (n_only, c_cnt))
    print('done!')
