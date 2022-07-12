import os
from common import *
from preprocessor import *
from analyzer import *
from dl_manager import *


def preprocess(split_fold=False):
    config = ConfigManager().load()
    # 원래는 ./config/config.ini 에서 정보를 가져오나 직접 경로를 설정해도 괜찮습니다.
    # raw_dir, pp_dir = config['raw_dir'], config['pp_dir']
    raw_dir = 'D:/Dropbox/Data/NExLAB_220502/01_EtOH pretretment'   # txt format 입력경로
    pp_dir = 'D:/Dropbox/Data/NExLAB_220502/01_pp'                  # 전처리된 csv 파일 출력경로

    if not os.path.exists(pp_dir):
        os.makedirs(pp_dir)

    processor = BatchProcessor(raw_dir, pp_dir, '.txt', '.csv', process_num=8)
    processor.batch_run(convert_txt2csv, use_parallel=True)

    # fold 정보를 새로 만들 경우
    if split_fold:
        case_info_path = config['case_info_path']
        fold_num = int(config['fold_num'])
        split_fold_with_stage(case_info_path, fold_num)


def test_dataset():
    for fold_idx in range(1,2):
        dataset = SpectrumDataset(DataPurpose.Train, fold_idx)
        # for spectra, label, info in dataset:
        #     x = dataset.wave_numbers
        #     for s_idx in range(spectra.shape[0]):
        #         y = spectra[s_idx, :]
        #         plt.plot(x, y)
        #     plt.show()
        #     break

        print('..%s fold %s spectrum %s cases' % (fold_idx, len(dataset), len(dataset.sid2size_dict)))

        dataset = SpectrumDataset(DataPurpose.Validation, fold_idx)
        print('..%s fold %s spectrum %s cases' % (fold_idx, len(dataset), len(dataset.sid2size_dict)))


def batch_train():
    net_lst = [NetTypes1D.CNN1D, NetTypes1D.Resnet, NetTypes1D.ConvLSTM]
    ver_lst = [12, 13, 22, 23, 32, 33, 42, 43]
    for net_type in net_lst:
        for ver in ver_lst:
            run_batch_train(ver, net_type)


def train_model(gpu_num=0):
    cm = ConfigManager()
    config = cm.load()
    # 원래는 ./config/config.ini 에 쓰는 내용이나 이해를 돕기위해 코드에 반영했습니다.
    config['pp_dir'] = 'D:/Dropbox/Data/NExLAB_220502/01_pp'    # 전처리된 csv 파일 경로
    config['case_info_path'] = './config/case_info_160.csv'     # 환자 정보

    # 스펙트럼을 모델에 입력하기 전 처리 방법
    config["SubtractionMode"] = SubtractionMode.NA.value    # 스펙트럼에서 평균, 최소값을 뺄 건지 여부
    config["RepresentationMode"] = RepresentationMode.ALL.value     # 대표값 (평균, 최빈값)을 사용할지 그대로 모든 스펙트럼을 쓸 지
    config["NormalizationMode"] = NormalizationMode.GB.value    # normalization 종류

    # 학습 관련
    transform_prob, fold_num, tolerance = 80, 5, 50
    config["use_scaling"] = str(True)   # data augmentation 기능 중 scaling
    config["use_shifting"] = str(True)  # data augmentation 기능 중 shifting
    config["use_rotation"] = str(True)  # data augmentation 기능 중 rotation
    config["transform_prob"] = "80"       # 각 data augmentation 함수 적용 확률 (%)
    config["target_type"] = "c2"        # 모델이 예측할 클래스 개수 (c2의 경우 암환자 vs 정상)
    config["fold_num"] = str(fold_num)  # fold 개수 (=k)
    config["tolerance"] = str(tolerance)  # early stopping 을 하기 까지 인내할 epoch 개수

    # optimizer 관련
    lr, b1, b2, wd = 1e-4, 0.9, 0.999, 1e-5
    config["lr"] = str(lr)
    config["b1"] = str(b1)
    config["b2"] = str(b2)
    config["wd"] = str(wd)

    # 딥러닝 모델 관련
    batch_size, depth, filters = 16, 10, 1024
    config["batch_size"] = str(batch_size)  # 배치 사이즈
    config["depth"] = str(depth)  # network 의 block 개수
    config["filters"] = str(filters)  # convolutional filter 개수
    network_type = NetTypes1D.CNN1D  # network 종류

    output_root = 'D:/Dropbox/Data/NExLAB_220502/01_result'     # 출력 경로 1
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    output_sub = '%s/%s' % (output_root, network_type.value)    # 출력 경로 2
    if not os.path.exists(output_sub):
        os.makedirs(output_sub)

    # config 정보 출력
    cm.save(config, '%s/config.ini' % output_sub)

    msg = 'ready to train %s model (%s-%s) with gpu %s' % (network_type.value, depth, filters, gpu_num)
    print(msg)

    # k-fold cross validation 진행
    for fold_index in range(fold_num):
        config["fold_index"] = str(fold_index)
        output_dir = '%s/fold_%s' % (output_sub, fold_index)   # 출력 경로 3

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        dl_manager = DeepLearningManager(network_type, output_dir, config)
        dl_manager.train(gpu_num)                               # 모델 학습
        print('..finished: fold %s' % fold_index)
    msg = '%s model has finished training with gpu %s' % (network_type.value, gpu_num)
    print(msg)


def evaluate_model(gpu_num=0):
    network_type = NetTypes1D.CNN1D

    output_root = 'D:/Dropbox/Data/NExLAB_220502/01_result'  # 모델 경로 1
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    t_perform_path = '%s/total_perform.csv' % output_root   # 각 스펙트럼 별 성능 출력경로
    s_perform_path = '%s/sample_perform.csv' % output_root  # 각 환자 별 성능 출력경로
    c_perform_path = '%s/class_perform.csv' % output_root   # 각 클래스 별 성능 출력경로

    output_sub = '%s/%s' % (output_root, network_type.value)  # 모델 경로 2
    if not os.path.exists(output_sub):
        os.makedirs(output_sub)

    output_path = '%s/val_perform.csv' % output_sub     # 각 샘플 별 예측 결과 출력경로
    if os.path.exists(output_path):
        os.remove(output_path)

    # config 정보 불러오기
    config = ConfigManager('%s/config.ini' % output_sub).load()
    fold_num = int(config['fold_num'])

    msg = 'ready to evaluate %s model with gpu %s' % (network_type.value, gpu_num)
    print(msg)

    # k-fold cross-validation 진행
    y_real_total, y_pred_total = [], []
    s_real_total, s_pred_total = [], []
    for fold_index in range(fold_num):
        config["fold_index"] = str(fold_index)
        output_dir = '%s/fold_%s' % (output_sub, fold_index)  # 출력 경로 3

        dl_manager = DeepLearningManager(network_type, output_dir, config)
        y_real, y_pred, samp_real, samp_pred = dl_manager.test_best(output_path, gpu_num)
        y_real_total.extend(y_real)
        y_pred_total.extend(y_pred)
        s_real_total.extend(samp_real)
        s_pred_total.extend(samp_pred)
        print('..finished: fold %s' % fold_index)

    # 최종 성능 출력
    tag = network_type.value
    # 각 스펙트럼 별로 성능 평가
    if dl_manager.n_classes == 1:
        metrics = binary_evaluate(y_real_total, y_pred_total)
    else:
        metrics, metric_lst = multi_evaluate(y_real_total, y_pred_total)
    export_metric(t_perform_path, metrics, tag)

    # 각 샘플 별로 성능 평가
    if dl_manager.n_classes == 1:
        metrics = binary_evaluate(s_real_total, s_pred_total)
    else:
        metrics, metric_lst = multi_evaluate(s_real_total, s_pred_total)
        for m_idx in range(len(metric_lst)):
            n_tag = '%s_%s' % (tag, m_idx)
            export_metric(c_perform_path, metric_lst[m_idx], n_tag)
    export_metric(s_perform_path, metrics, tag)

    msg = '%s model has evaluated with gpu %s' % (network_type.value, gpu_num)
    print(msg)


if __name__ == "__main__":
    print('NExLAB project 22-07-12')

    # 첫 번째 단계: raw data (.txt)를 전처리 하여 .csv 포맷으로 출력
    preprocess(False)

    # 두 번째 단계: deep learning model 학습
    train_model(gpu_num=0)

    # 세 번째 단계: model evaluation
    evaluate_model(gpu_num=0)
