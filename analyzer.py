import os

import matplotlib.pyplot as plt

from spectrum_loader import *
from statistics import *


def export_group_plot(spectrum_lst, label_lst, wave_numbers, output_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    pos_lst, neg_lst = [], []
    for idx in range(len(spectrum_lst)):
        spectrum, label = spectrum_lst[idx], label_lst[idx]
        if label == 0:
            c = 'blue'
            neg_lst.append(spectrum)
        else:
            c = 'red'
            pos_lst.append(spectrum)
        ax1.plot(wave_numbers, spectrum, c=c, alpha=0.25)

    spectra = np.array(pos_lst)
    name, c, fc = 'PDAC', 'red', 'lightcoral'
    mean_lst, sd_lst = np.mean(spectra, axis=0), np.std(spectra, axis=0)
    lower_bound, upper_bound = mean_lst - sd_lst, mean_lst + sd_lst
    ax2.fill_between(wave_numbers, lower_bound, upper_bound, facecolor=fc, edgecolors='None', alpha=0.5,
                     label=name)
    ax2.plot(wave_numbers, mean_lst, c=c)

    spectra = np.array(neg_lst)
    name, c, fc = 'Normal', 'blue', 'lightskyblue'
    mean_lst, sd_lst = np.mean(spectra, axis=0), np.std(spectra, axis=0)
    lower_bound, upper_bound = mean_lst - sd_lst, mean_lst + sd_lst
    ax2.fill_between(wave_numbers, lower_bound, upper_bound, facecolor=fc, edgecolors='None', alpha=0.5,
                     label=name)
    ax2.plot(wave_numbers, mean_lst, c=c)

    # output_path = '%s/SM %s-RM %s-NM %s.jpg' % (output_dir, sm.value, rm.value, nm.value)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def plot_spectrum():
    '''
    https://doaiacropolis.atlassian.net/wiki/spaces/AIRaman/pages/2131361800
    하단 시각화 자료 생성
    :return:
    '''
    result_dir = './result'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    output_dir = './result/spectrum_figure'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config = ConfigManager().load()
    result_dict = {'SubtractionMode':[], 'RepresentationMode':[], 'NormalizationMode':[],
                   'Significant_ratio':[], 'mean_p_value':[], 'mean_p_value_sig':[], 'mean_auc':[]}
    sm_lst = {SubtractionMode.MIN, SubtractionMode.NA}
    rm_lst = {RepresentationMode.MED, RepresentationMode.ALL}
    nm_lst = {NormalizationMode.LC, NormalizationMode.GB}
    for sm in sm_lst:
        config["SubtractionMode"] = sm.value
        for rm in rm_lst:
            config["RepresentationMode"] = rm.value
            for nm in nm_lst:
                config["NormalizationMode"] = nm.value
                print(sm.value, rm.value, nm.value)

                dataset = SpectrumDataset(DataPurpose.Total, config=config)
                wave_numbers = dataset.wave_numbers
                spectrum_lst = dataset.spectrum_lst
                label_lst = dataset.label_lst

                spectra = np.array(spectrum_lst)
                sig_cnt = 0
                all_auc_lst, sig_pval_lst, all_pval_lst = [], [], []

                output_path = '%s/SM %s-RM %s-NM %s.jpg' % (output_dir, sm.value, rm.value, nm.value)
                export_group_plot(spectrum_lst, label_lst, wave_numbers, output_path)

                for idx in range(spectra.shape[1]):
                    wn = wave_numbers[idx]
                    val_lst = spectra[:, idx]

                    stat_result = run_statistic(label_lst, val_lst)
                    auc = stat_result["AUROC"]
                    p_value = stat_result["p-value"]

                    if p_value < 0.05:
                        sig_cnt += 1
                        sig_pval_lst.append(p_value)
                    all_auc_lst.append(auc)
                    all_pval_lst.append(p_value)
                result_dict['SubtractionMode'].append(sm.value)
                result_dict['RepresentationMode'].append(rm.value)
                result_dict['NormalizationMode'].append(nm.value)

                result_dict['Significant_ratio'].append(sig_cnt/spectra.shape[1])
                result_dict['mean_p_value'].append(np.nanmean(sig_pval_lst))
                result_dict['mean_p_value_sig'].append(np.nanmean(all_pval_lst))
                result_dict['mean_auc'].append(np.nanmean(all_auc_lst))

    output_path = '%s/stat_result_binary.csv' % result_dir
    pd.DataFrame(result_dict).to_csv(output_path, index=False)

def show_scatter_plot():
    df = pd.read_csv('./config/case_info_160.csv')
    label_lst = df.values[:, 3]
    label2color = {'normal':'blue', 'PDAC':'red', 'etc':'orange'}
    x_lst, y_lst = df.values[:, 6], df.values[:, 9]

    class_lst = sorted(list(set(label_lst)))
    for now_c in class_lst:
        index_lst = [i for i in range(df.shape[0]) if label_lst[i] == now_c]
        x_part, y_part = x_lst[index_lst], y_lst[index_lst]
        c = label2color[now_c]
        plt.scatter(x_part, y_part, c=c)
    plt.show()


def export_statistics():
    '''
    환자 정보 파일 내 바이오마커에 대한 통계결과 출력
    :return:
    '''
    config = ConfigManager().load()
    case_info_path = config['case_info_path']
    df = pd.read_csv(case_info_path)
    label_lst = [0 if x.lower() == 'normal' else 1 for x in df.values[:, 3]]

    result_dict = {'Tag': [], 'Positive_median': [], 'Positive_IQR': [],
                   'Negative_median': [], 'Negative_IQR': [], 'p-value': [], 'AUROC': [], 'CI95': [],
                   'Positive_mean': [], 'Negative_mean': [], 'Threshold':[],
                   'ACC':[], 'SEN':[], 'SPE':[], 'PPV':[], 'NPV':[]}

    print('data is ready!')
    # 바이오마커 정보의 컬럼 위치를 지정해줘야 함.
    for col in range(6, 10):
        value_lst = df.values[:, col]
        tag = df.columns[col]

        stat_result = run_statistic(label_lst, value_lst)
        result_dict['Tag'].append(tag)
        result_dict['Positive_median'].append('%.3f' % stat_result['pos_Q2'])
        result_dict['Positive_IQR'].append('%.3f-%.3f' % (stat_result['pos_Q1'], stat_result['pos_Q3']))
        result_dict['Negative_median'].append('%.3f' %stat_result['neg_Q2'])
        result_dict['Negative_IQR'].append('%.3f-%.3f' % (stat_result['neg_Q1'], stat_result['neg_Q3']))
        result_dict['p-value'].append(stat_result['p-value'])
        result_dict['AUROC'].append('%.3f' % stat_result['AUROC'])
        result_dict['CI95'].append('%.3f-%.3f' % (stat_result['CI_lower'], stat_result['CI_upper']))
        result_dict['Positive_mean'].append('%.3f' % stat_result['pos_mean'])
        result_dict['Negative_mean'].append('%.3f' % stat_result['neg_mean'])
        threshold = stat_result['threshold']

        if 'CA' in tag:
            threshold = 37
        elif 'CEA' in tag:
            threshold = 5

        result_dict['Threshold'].append('%f' % threshold)
        binary_lst = [1 if x >= threshold else 0 for x in value_lst]
        accuracy, sen, spe, ppv, npv = get_binary_metric(label_lst, binary_lst)
        result_dict['ACC'].append('%.3f' % accuracy)
        result_dict['SEN'].append('%.3f' % sen)
        result_dict['SPE'].append('%.3f' % spe)
        result_dict['PPV'].append('%.3f' % ppv)
        result_dict['NPV'].append('%.3f' % npv)

        print('..%s has been analysed' % tag)

    result_dir = './result'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    output_path = './result/biomarker_stat_220510.csv'
    pd.DataFrame(result_dict).to_csv(output_path, index=False)
    print('done!')


def concat_file():
    dir1 = 'D:/Dropbox/data/AI-Raman/220111/pp_220107'
    dir2 = 'D:/Dropbox/data/AI-Raman/220111/pp_220110'
    out_dir = 'D:/Dropbox/data/AI-Raman/220111/pp_merge'
    for f_name in os.listdir(dir1):
        path2 = '%s/%s' % (dir2, f_name)
        if not os.path.exists(path2):
            continue
        path1 = '%s/%s' % (dir2, f_name)
        dict1 = pd.read_csv(path1).to_dict('list')
        dict2 = pd.read_csv(path2).to_dict('list')

        for key, lst in dict2.items():
            if key == 'WaveNumber':
                continue
            new_key = 'v2_%s' % key
            dict1[new_key] = lst
        out_path = '%s/%s' % (out_dir, f_name)
        pd.DataFrame(dict1).to_csv(out_path, index=False)
        print('...%s has been concatenated.' % f_name)

def plot_spectrum_group(organ2group, group2color, tag):
    dataset = SpectrumDataset(DataPurpose.Total)
    spectrum_lst = dataset.spectrum_lst
    info_lst = dataset.info_lst
    wave_numbers = dataset.wave_numbers

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    group2spectrum = dict()
    for idx in range(len(spectrum_lst)):
        spectrum, info = spectrum_lst[idx], info_lst[idx]
        sex, age, o_type = info
        group = organ2group[o_type]
        if group not in group2spectrum:
            group2spectrum[group] = []
        group2spectrum[group].append(spectrum)
        c, fc = group2color[group]
        ax1.plot(wave_numbers, spectrum, c=c, alpha=0.25)
    #ax1.legend()

    for group, spectrum_lst in group2spectrum.items():
        spectra = np.array(spectrum_lst)
        c, fc = group2color[group]

        mean_lst, sd_lst = np.median(spectra, axis=0), np.std(spectra, axis=0)
        lower_bound, upper_bound = mean_lst - sd_lst, mean_lst + sd_lst
        # ax2.fill_between(wave_numbers, lower_bound, upper_bound, facecolor=fc, edgecolors='None', alpha=0.5,
        #                  label=organ)
        ax2.plot(wave_numbers, mean_lst, c=c, label=group)
    ax2.legend()

    output_path = './result/plot_group_%s.png' % tag
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def export_plot_3group():
    '''
    https://doaiacropolis.atlassian.net/wiki/spaces/AIRaman/pages/2217017345
    상기 페이지 하단 '스펙트럼 시각화' 부분 출력
    :return:
    '''
    result_dir = './result'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    organ2group = {
        "GALLBLADDER": "GALLBLADDER",
        "PANCREAS":  "PANCREAS",
        "STOMACH": "STOMACH",
        "COLORECTUM": "COLORECTUM",
        "BREAST":  "BREAST",
        "THYROID": "THYROID",
    }
    group2color = {
        "GALLBLADDER": ['blue', 'lightcoral'],
        "PANCREAS": ['red', 'lightcoral'],
        "STOMACH": ['orange', 'salmon'],
        "COLORECTUM": ['olive', 'lightgreen'],
        "BREAST": ['purple', 'orchid'],
        "THYROID": ['chocolate', 'brown'],
    }
    plot_spectrum_group(organ2group, group2color, '6')

    organ2group = {
        "GALLBLADDER": "Normal",
        "PANCREAS":  "PDAC",
        "STOMACH": "etc",
        "COLORECTUM": "etc",
        "BREAST":  "etc",
        "THYROID": "etc",
    }
    group2color = {
        "Normal": ['blue', 'lightcoral'],
        "PDAC": ['red', 'lightcoral'],
        "etc": ['orange', 'salmon'],
    }
    plot_spectrum_group(organ2group, group2color, '3')

    organ2group = {
        "GALLBLADDER": "Normal",
        "PANCREAS":  "cancer",
        "STOMACH": "cancer",
        "COLORECTUM": "cancer",
        "BREAST":  "cancer",
        "THYROID": "cancer",
    }
    group2color = {
        "Normal": ['blue', 'lightcoral'],
        "cancer": ['red', 'lightcoral'],
    }
    plot_spectrum_group(organ2group, group2color, '2')


def get_mae(original, reconstructed):
    return np.mean(np.abs(original - reconstructed))


def get_rmse(original, reconstructed):
    return np.sqrt(((original - reconstructed) ** 2).mean())


def get_prd(original, reconstructed):
    return np.sqrt(((original - reconstructed) ** 2).sum() / (original ** 2).sum()) * 100.0


def get_prdn(original, reconstructed):
    avg = original.mean()
    return np.sqrt(((original - reconstructed) ** 2).sum() / ((original - avg) ** 2).sum()) * 100.0


def get_snr(original, reconstructed):
    avg = original.mean()
    return np.log(((original - avg) ** 2).sum() / ((original - reconstructed) ** 2).sum()) * 10.0


def calculate_score(list1, list2, self_check=False, fnc=get_snr, max_iter=100):
    indexes1 = list(range(len(list1)))
    indexes2 = list(range(len(list2)))
    shuffle(indexes1)
    shuffle(indexes2)
    iterator = 0
    score_lst = []
    for idx1 in indexes1:
        for idx2 in indexes2:
            if idx1 == idx2 and self_check:
                continue
            iterator += 1
            s1, s2 = list1[idx1], list2[idx2]
            score = fnc(s1, s2)
            if np.isnan(score) == False and np.isinf(score) == False:
                score_lst.append(score)
            if iterator >= max_iter:
                break
    return score_lst


def compare_old_and_new(remove_average=False):
    old_dir = 'D:/Dropbox/data/AI-Raman/220629/04_old'
    new_dir = 'D:/Dropbox/data/AI-Raman/220629/04_pp'
    fig_dir = 'D:/Dropbox/data/AI-Raman/220629/04_fig'

    f_lst = os.listdir(old_dir)
    self_score_lst = []
    for f_name in f_lst:
        path = '%s/%s' % (new_dir, f_name)
        old_mat = pd.read_csv(path).values
        s_lst = []
        for ci in range(1, old_mat.shape[1]):
            spectrum = old_mat[:, ci]
            if remove_average:
                spectrum -= np.mean(spectrum)
            s_lst.append(spectrum)

        score_lst = calculate_score(s_lst, s_lst, True)
        if len(score_lst) == 0:
            print('error on %s file comparison' % f_name)
        mean_score = np.mean(score_lst)
        self_score_lst.append(mean_score)
    min_s, max_s = np.min(self_score_lst), np.max(self_score_lst)
    mean_s, med_s = np.mean(self_score_lst), np.median(self_score_lst)
    sd_s = np.std(self_score_lst)
    m1 = '===== self score =====\n'
    m2 = 'Mean: %.1f, S.D: %.1f\n' % (mean_s, sd_s)
    m3 = 'Min: %.1f, Median %.1f, Max: %.1f\n' % (min_s, med_s, max_s)
    print(m1+m2+m3)

    score_path = '%s/score_list.csv' % os.path.dirname(old_dir)
    if os.path.exists(score_path):
        score_dict = pd.read_csv(score_path).to_dict('list')
    else:
        score_dict = dict()
    pid_lst, tot_score_lst = [], []
    for f_name in f_lst:
        pid = f_name.split('.')[0]
        path = '%s/%s' % (old_dir, f_name)
        old_mat = pd.read_csv(path).values
        old_lst = []
        for ci in range(1, old_mat.shape[1]):
            spectrum = old_mat[:, ci]
            if remove_average:
                spectrum -= np.mean(spectrum)
            old_lst.append(spectrum)
        path = '%s/%s' % (new_dir, f_name)
        new_mat = pd.read_csv(path).values
        new_lst = []
        for ci in range(1, new_mat.shape[1]):
            spectrum = new_mat[:, ci]
            if remove_average:
                spectrum -= np.mean(spectrum)
            new_lst.append(spectrum)
        wave_numbers = new_mat[:, 0]

        score_lst = calculate_score(old_lst, new_lst, False)
        if len(score_lst) == 0:
            print('error on %s file comparison' % pid)
        mean_score = np.mean(score_lst)
        pid_lst.append(pid)
        tot_score_lst.append(mean_score)

        # med_lst, sd_lst = np.median(old_mat[:, 1:], axis=1), np.std(old_mat[:, 1:], axis=1)
        # lower_bound, upper_bound = med_lst - sd_lst, med_lst + sd_lst
        # c, fc = 'blue', 'lightblue'
        # plt.fill_between(wave_numbers, lower_bound, upper_bound, facecolor=fc, edgecolors='None', alpha=0.5)
        # plt.plot(wave_numbers, med_lst, c=c)
        #
        # med_lst, sd_lst = np.median(new_mat[:, 1:], axis=1), np.std(new_mat[:, 1:], axis=1)
        # lower_bound, upper_bound = med_lst - sd_lst, med_lst + sd_lst
        # c, fc = 'red', 'lightcoral'
        # plt.fill_between(wave_numbers, lower_bound, upper_bound, facecolor=fc, edgecolors='None', alpha=0.5)
        # plt.plot(wave_numbers, med_lst, c=c)
        # plt.title('%s: SNR=%.1fdB' % (pid, mean_score))
        # #plt.show()
        #
        # output_path = '%s/%s.png' % (fig_dir, pid)
        # plt.savefig(output_path, bbox_inches='tight')
        # plt.close()

    score_dict['pid'] = pid_lst
    tag = os.path.basename(new_dir)
    score_dict[tag] = tot_score_lst
    pd.DataFrame(score_dict).to_csv(score_path, index=False)

    min_s, max_s = np.min(tot_score_lst), np.max(tot_score_lst)
    mean_s, med_s = np.mean(tot_score_lst), np.median(tot_score_lst)
    sd_s = np.std(tot_score_lst)
    m1 = '===== old vs new score =====\n'
    m2 = 'Mean: %.1f, S.D: %.1f\n' % (mean_s, sd_s)
    m3 = 'Min: %.1f, Median %.1f, Max: %.1f\n' % (min_s, med_s, max_s)
    print(m1+m2+m3)


def copy_old_to_old():
    import shutil
    config = ConfigManager().load()
    case_info_path = config['case_info_path']

    case_info = pd.read_csv(case_info_path)
    fold_info = case_info['fold'].values
    indexes = [i for i in range(len(fold_info)) if fold_info[i] == 1]
    pid_lst = case_info.iloc[indexes, 0].values

    src_dir = 'D:/Dropbox/data/AI-Raman/220502/01_pp'
    des_dir = 'D:/Dropbox/data/AI-Raman/220629/04_old'

    for pid in pid_lst:
        src_path = '%s/%03d.csv' % (src_dir, int(pid))
        des_path = '%s/%03d.csv' % (des_dir, int(pid))
        shutil.copyfile(src_path, des_path)
