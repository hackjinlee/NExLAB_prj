import os
from common import *
from preprocessor import *
from analyzer import *
from dl_manager import *


def preprocess(split_fold=False):
    config = ConfigManager().load()
    raw_dir, pp_dir = config['raw_dir'], config['pp_dir']

    processor = BatchProcessor(raw_dir, pp_dir, '.txt', '.csv', process_num=8)
    processor.batch_run(convert_txt2csv, use_parallel=True)

    # fold 정보를 새로 만들 경우
    if split_fold:
        case_info_path = config['case_info_path']
        fold_num = int(config['fold_num'])
        split_fold_with_stage(case_info_path, fold_num)


def test_dataset():
    for fold_idx in range(5):
        dataset = SpectrumDataset(DataPurpose.Train, fold_idx)
        for spectra, label, info in dataset:
            x = dataset.wave_numbers
            for s_idx in range(spectra.shape[0]):
                y = spectra[s_idx, :]
                plt.plot(x, y)
            plt.show()
            break

        print('..%s fold %s spectrum %s cases' % (fold_idx, len(dataset), len(dataset.sid2size_dict)))


def batch_train():
    net_lst = {NetTypes1D.CNN1D, NetTypes1D.Resnet, NetTypes1D.ConvLSTM}
    ver_lst = {32, 33, 34, 36}
    for net_type in net_lst:
        for ver in ver_lst:
            run_batch_train(ver, net_type)


if __name__ == "__main__":
    print('NExLAB project 22-06-20')
    # preprocess()
    # batch_train


