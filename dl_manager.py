import os

from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score, accuracy_score, average_precision_score
import pandas as pd
import numpy as np

from dl_core import *
from common import *
from statistics import *
from spectrum_loader import *


class Metrics(enum.Enum):
    '''
    성능평가 지표
    '''
    TP = 'True Positive'
    TN = 'True Negative'
    FP = 'False Positive'
    FN = 'False Negative'

    ACC = 'Accuracy'
    SEN = 'Sensitivity'
    SPE = 'Specificity'
    PPV = 'Positive predictive values'
    NPV = 'Negative predictive values'
    F1 = 'F1score'
    F2 = 'F2score'

    AUROC = 'AUROC'
    AUPRC = 'AUPRC'

class Logger:
    '''
    학습 시 loss 정보를 출력하기 위한 class
    '''
    def __init__(self, log_path):
        self.log_path = log_path
        self.log_dict = {'tag': []}
        if os.path.exists(log_path):
            df = pd.read_csv(log_path)
            self.log_dict = df.to_dict(orient='list')

    def get_last_epoch(self):
        tag_lst = self.log_dict['tag']
        epoch = -1
        if len(tag_lst) > 0:
            epoch = int(tag_lst[-1])
        return epoch

    def log(self, tag, loss_dict=None):
        log_dict = self.log_dict
        log_dict['tag'].append(tag)
        if loss_dict is not None:
            for key, value in loss_dict.items():
                if key not in log_dict:
                    log_dict[key] = []
                log_dict[key].append(value)

        df = pd.DataFrame(log_dict)
        df.to_csv(self.log_path, index=False)


class DeepLearningManager:
    def __init__(self, network_type, output_dir, config):
        self.network_type = network_type
        self.output_dir = output_dir
        self.config = config
        target_type = TargetTypes(config['target_type'])
        if target_type == TargetTypes.Class2:
            n_classes = 1
        elif target_type == TargetTypes.Class3:
            n_classes = 3
        elif target_type == TargetTypes.Class6:
            n_classes = 6

        self.n_classes = n_classes
        self.depth = int(config["depth"])
        self.filters = int(config["filters"])
        self.fold_index = int(config["fold_index"])
        self.batch_size = int(config["batch_size"])
        # self.spectra_number = int(config["spectra_number"])
        # self.spectrum_length = int(config["spectrum_length"])
        self.tolerance = int(config["tolerance"])
        self.metric = Metrics.F1

    def get_score(self, y_real, y_pred):
        if self.metric == Metrics.ACC:
            score = accuracy_score(y_real, y_pred)
        elif self.n_classes == 1 and self.metric == Metrics.F1:
            score = f1_score(y_real, y_pred)
        elif self.metric == Metrics.F1:
            score = f1_score(y_real, y_pred, average='macro')
        return score

    def test_best(self, output_path, gpu_num=0):
        '''
        가장 성능이 좋은 경우를 로드하여 성능평가
        :param output_path:
        :param gpu_num:
        :return:
        '''
        dir_name = os.path.basename(self.output_dir)
        print('..start test_model from %s (gpu:%s)' % (dir_name, gpu_num))
        weight_path = '%s/best.pt' % self.output_dir
        device = torch.device('cuda:%s' % gpu_num)

        val_set = SpectrumDataset(DataPurpose.Validation, self.fold_index, self.config)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)

        input_shape = val_set.input_shape
        model = DeepLearningModel1D(self.network_type, input_shape, self.n_classes, self.depth, self.filters).to(device)

        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.eval()

        y_real, y_pred, test_loss = self._run(DataPurpose.Test, model, device, val_loader)
        sid2size_dict = val_set.sid2size_dict
        si = 0
        samp_real, samp_pred = [], []
        for sid, size in sid2size_dict.items():
            y_real_part = y_real[si:si + size]
            y_pred_part = y_pred[si:si + size]
            if len(y_pred_part) == 0:
                si += size
                continue
            samp_real.append(y_real_part[0])
            if self.n_classes == 1:
                #pred = 1 if np.mean(y_pred_part) >= 0.5 else 0
                pred = np.mean(y_pred_part)
                samp_pred.append(pred)
                metrics = binary_evaluate(y_real_part, y_pred_part)
            else:
                pred = max(set(y_pred_part), key=y_pred_part.count)
                samp_pred.append(pred)
                metrics, _ = multi_evaluate(y_real_part, y_pred_part)
            export_metric(output_path, metrics, sid)
            tag = Metrics.ACC.value
            score = metrics[tag]
            print('..finished: test_model from %s - %s: %.3f' % (dir_name, tag, score))
            si += size
        return y_real, y_pred, samp_real, samp_pred

    def train(self, gpu_num, num_epochs=500):
        '''
        모델 학습
        :param gpu_num:
        :param num_epochs:
        :return:
        '''
        dir_name = os.path.basename(self.output_dir)
        print('..start train_model to %s (gpu:%s)' % (dir_name, gpu_num))
        device = torch.device('cuda:%s' % gpu_num)

        # 데이터 로더, 모델, 로그파일 등 준비
        val_set = SpectrumDataset(DataPurpose.Validation, self.fold_index, self.config)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)

        input_shape = val_set.input_shape
        model = DeepLearningModel1D(self.network_type, input_shape, self.n_classes, self.depth, self.filters).to(device)

        log_path = '%s/log.csv' % self.output_dir
        logger = Logger(log_path)
        start_epoch = 0
        max_score, min_score, patience = 0, 10 ** 5, 0
        w_path = '%s/best.pt' % self.output_dir

        # 이전 weight 가 있을 경우 불러와 학습
        if os.path.exists(w_path):
            model.load_state_dict(torch.load(w_path, map_location=device))
            y_real, y_pred, val_loss = self._run(DataPurpose.Validation, model, device, val_loader)
            max_score = self.get_score(y_real, y_pred)
            start_epoch = logger.get_last_epoch()
            print('..old weight is detected %s' % start_epoch)
            start_epoch += 1

        optim_c = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.999), weight_decay=1e-5)
        for epoch in range(start_epoch, num_epochs):
            # 각 epoch 별로 모델 학습 > 검증 반복
            train_set = SpectrumDataset(DataPurpose.Train, self.fold_index, self.config)

            if self.n_classes == 1:
                sampler = ImbalancedDatasetSampler(train_set)
                train_loader = DataLoader(train_set, batch_size=self.batch_size, sampler=sampler)
            else:
                train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)

            y_real, y_pred, train_loss = self._run(DataPurpose.Train, model, device, train_loader, optim_c=optim_c)
            train_score = self.get_score(y_real, y_pred)
            y_real, y_pred, val_loss = self._run(DataPurpose.Validation, model, device, val_loader)
            val_score = self.get_score(y_real, y_pred)
            logger.log(str(epoch), {"train_loss": train_loss, 'val_loss': val_loss,
                                    'train_score': train_score, 'vali_score': val_score})

            # 학습 성능과 검증 성능을 비교하여 early stropping 적용
            best_path = '%s/best.pt' % self.output_dir
            if epoch == 0:
                torch.save(model.state_dict(), best_path)
            elif val_score < 0.1 or train_score + 0.05 < val_score:
                if 0.9 < train_score:
                    patience += 10
                else:
                    patience += 1
            elif max_score < val_score:
                max_score = val_score
                torch.save(model.state_dict(), best_path)
                patience = 0
            elif 0.9 < train_score:
                patience += 10
            else:
                patience += 1
            msg = '...epoch %s, train:%.3f, val:%.3f (best:%.3f, %s)' % \
                  (epoch, train_score, val_score, max_score, patience)
            print(msg)
            if patience >= self.tolerance:
                break

    def _run(self, data_purpose, net_C, device, loader, epoch='E', optim_c=None):
        '''
        data_purpose 에 따라서 모델 학습/검증/테스트
        :param data_purpose:
        :param net_C:
        :param device:
        :param loader:
        :param epoch:
        :param optim_c:
        :return:
        '''
        n_classes = self.n_classes
        crit_bce = nn.BCEWithLogitsLoss()
        crit_cel = nn.CrossEntropyLoss()

        idx = 0
        c_loss_sum = 0
        if data_purpose == DataPurpose.Train:
            net_C.train()
        else:
            net_C.eval()

        y_real, y_pred = [], []
        for spectra, label, info in loader:
            vol_cuda = Variable(spectra).to(device)
            target_cuda = Variable(label).to(device)

            if data_purpose == DataPurpose.Train:
                optim_c.zero_grad()
                output = net_C(vol_cuda)
            else:
                with torch.no_grad():
                    output = net_C(vol_cuda)

            if n_classes == 1:
                if output.shape[0] > 1:
                    output = output.squeeze()
                else:
                    output = output[0]

                c_loss = crit_bce(output, target_cuda.float())
            else:
                c_loss = crit_cel(output, target_cuda.long())

            if data_purpose == DataPurpose.Train:
                c_loss.backward()
                optim_c.step()

            loss_itm = c_loss.item()
            if not np.isnan(loss_itm):
                c_loss_sum += loss_itm
            idx += 1

            label_arr = label.numpy()
            if n_classes == 1:
                pred_arr = torch.sigmoid(output).cpu().detach().numpy()
            else:
                pred_arr = torch.softmax(output, dim=1).cpu().detach().numpy()

            for i in range(len(label)):
                y_real.append(label_arr[i])
                if n_classes == 1:
                    pred = 1 if pred_arr[i] >= 0.5 else 0
                else:
                    pred = np.argmax(pred_arr[i])
                y_pred.append(pred)

            if idx % (len(loader) // 3) == 0:
                msg = '...%s [%s-%s], loss_C:%.3f' % (data_purpose.value, epoch, idx, c_loss_sum / idx)
                print(msg)

        loss = c_loss_sum / idx
        return y_real, y_pred, loss


def run_batch_train(ver, network_type=NetTypes1D.CNN1D):
    '''
    미리 정해둔 version 정보와 네트워크 종류에 따라서 모델 학습 및 성능 평가
    :param ver:
    :param network_type:
    :return:
    '''
    gpu_num = 0
    cm = ConfigManager()
    config = cm.load()
    fold_num = int(config['fold_num'])

    config['case_info_path'] = './config/case_info_160.csv'
    # 입력 디렉토리 설정
    d_ver = ver // 10
    config['pp_dir'] = 'D:/Dropbox/data/AI-Raman/220502/0%s_pp' % d_ver

    c_ver = ver % 10
    # class 개수 선택
    config['target_type'] = 'c%s' % c_ver
    dir_name = 'p%s_c%s' % (d_ver, c_ver)

    output_first = '%s/%s' % (config['result_dir'], dir_name)
    if not os.path.exists(output_first):
        os.makedirs(output_first)

    msg = 'Batch training %s [gpu:%s] (output:%s)' % (network_type.value, gpu_num, dir_name)
    print(msg)

    output_root = '%s/%s' % (output_first, network_type.value)
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # 네크워크 구조 최적화를 위한 depth 와 filters, batch size 범위
    comb_dict = {
        "depth": [10, 9, 8],
        "filters": [1024, 512, 256],
    }
    f2b_dict = {32: 64, 64: 32, 128: 16, 196: 8, 256: 64, 384: 48, 512: 32, 1024: 16}
    t_perform_path = '%s/total_perform.csv' % output_first
    s_perform_path = '%s/sample_perform.csv' % output_first
    c_perform_path = '%s/class_perform.csv' % output_first

    for now_c in list(ParameterGrid(comb_dict)):
        # 모든 경우의 수에 따라서 모델을 학습 시킴.
        depth = now_c["depth"]
        filters = now_c["filters"]

        batch_size = f2b_dict[filters]
        config["batch_size"] = str(batch_size)
        config["depth"] = str(depth)
        config["filters"] = str(filters)

        name1 = '%s_%s' % (depth, filters)
        output_dir = '%s/%s' % (output_root, name1)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_path = '%s/%s/val_perform.csv' % (output_root, name1)
        if os.path.exists(output_path):
            os.remove(output_path)

        cm.save(config, '%s/%s/config.ini' % (output_root, name1))

        y_real_total, y_pred_total = [], []
        s_real_total, s_pred_total = [], []
        for fold_index in range(fold_num):
            # 5 fold cross validation 실행
            config["fold_index"] = str(fold_index)
            name2 = 'fold_%s' % fold_index
            output_dir = '%s/%s/%s' % (output_root, name1, name2)

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            dl_manager = DeepLearningManager(network_type, output_dir, config)
            dl_manager.train(gpu_num)

            y_real, y_pred, samp_real, samp_pred = dl_manager.test_best(output_path, gpu_num)
            y_real_total.extend(y_real)
            y_pred_total.extend(y_pred)
            s_real_total.extend(samp_real)
            s_pred_total.extend(samp_pred)

        # 최종 성능 출력
        tag = '%s_%s' % (network_type.value, name1)
        if dl_manager.n_classes == 1:
            metrics = binary_evaluate(y_real_total, y_pred_total)
        else:
            metrics, metric_lst = multi_evaluate(y_real_total, y_pred_total)
        export_metric(t_perform_path, metrics, tag)

        if dl_manager.n_classes == 1:
            metrics = binary_evaluate(s_real_total, s_pred_total)
        else:
            metrics, metric_lst = multi_evaluate(s_real_total, s_pred_total)
            for m_idx in range(len(metric_lst)):
                n_tag = '%s_%s' % (tag, m_idx)
                export_metric(c_perform_path, metric_lst[m_idx], n_tag)
        export_metric(s_perform_path, metrics, tag)
        score = metrics[Metrics.ACC.value]
        print('..finished: test_model from %s - accuracy: %.3f' % (name1, score))


def analyze_train_size(gpu_num=0, network_type=NetTypes1D.CNN1D):
    cm = ConfigManager()
    config = cm.load()
    fold_num = int(config['fold_num'])

    config['pp_dir'] = '/home/Dataset/AI-Raman/Solution/pp_220107'
    config['case_info_path'] = './config/case_info_101.csv'

    depth, filters, batch_size = 9, 1024, 32
    config["batch_size"] = str(batch_size)
    config["depth"] = str(depth)
    config["filters"] = str(filters)

    output_root = './for_train_size'
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    cm.save(config, '%s/config.ini' % output_root)
    t_perform_path = '%s/total_perform.csv' % output_root
    s_perform_path = '%s/sample_perform.csv' % output_root

    # if os.path.exists(t_perform_path):
    #     os.remove(t_perform_path)

    # if os.path.exists(s_perform_path):
    #     os.remove(s_perform_path)

    for test_index in range(fold_num):
        config["fold_index"] = str(test_index)
        name1 = 'test_%s' % test_index

        output_dir1 = '%s/%s' % (output_root, name1)
        if not os.path.exists(output_dir1):
            os.makedirs(output_dir1)

        for train_index in range(fold_num):
            indexes = list(range(train_index + 1))
            if test_index in indexes:
                indexes.remove(test_index)

            if len(indexes) == 0:
                continue

            name2 = 'train_%s' % train_index
            output_dir = '%s/%s/%s' % (output_root, name1, name2)
            config["train_indexes"] = str(indexes)[1:-1]
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            dl_manager = DeepLearningManager(network_type, output_dir, config)
            # dl_manager.train(gpu_num, use_differ_size=True)

            output_path = '%s/val_perform.csv' % output_dir
            y_real, y_pred, samp_real, samp_pred = dl_manager.test_best(output_path, gpu_num)

            tag = '%s_%s' % (test_index, train_index)
            metrics = binary_evaluate(y_real, y_pred)
            export_metric(t_perform_path, metrics, tag)

            metrics = binary_evaluate(samp_real, samp_pred)
            export_metric(s_perform_path, metrics, tag)
            metric = str(dl_manager.metric.value)
            score = metrics[metric]
            print('..finished: test_model from %s - %s: %.3f' % (name1, tag, score))


def multi_evaluate(y_real, y_pred):
    label_lst = sorted(list(set(y_real)))
    total_dict, result_lst = dict(), []
    for label in label_lst:
        result_dict = binary_evaluate(y_real, y_pred, label=label)
        for key, val in result_dict.items():
            if key not in total_dict:
                total_dict[key] = []
            total_dict[key].append(val)
        result_lst.append(result_dict)

    for key, val in total_dict.items():
        if key == Metrics.TP.value or key == Metrics.TN.value:
            new_val = np.round(np.nanmean(val))
        elif key == Metrics.FP.value or key == Metrics.FN.value:
            new_val = np.round(np.nanmean(val))
        else:
            new_val = np.nanmean(val)
        #new_val = np.nanmean(val)
        total_dict[key] = new_val
    return total_dict, result_lst


def binary_evaluate(y_real, y_pred, label=1):
    '''
    이진 분류에 대한 성능평가 실행
    :param y_real: 실제값
    :param y_pred: 예측값
    :param label:
    :return: 성능결과
    '''
    if len(set(y_pred)) == 2:
        y_pred_new = [1 if pred >=0.5 else 0 for pred in y_pred]
    else:
        y_pred_new = y_pred

    tp = tn = fp = fn = 0
    for i in range(len(y_real)):
        actual = y_real[i]
        predict = y_pred_new[i]
        if predict == label and actual == label:
            tp += 1
        elif predict != label and actual != label:
            tn += 1
        elif predict == label and actual != label:
            fp += 1
        elif predict != label and actual == label:
            fn += 1

    accuracy = (tp + tn) / (tp + tn + fn + fp) if (tp + tn + fn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall = sensitivity
    f1_value = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    f2_value = (5 * precision * recall) / ((4 * precision) + recall) if (precision + recall) > 0 else 0

    result_dict = dict()
    result_dict[Metrics.TP.value] = tp
    result_dict[Metrics.TN.value] = tn
    result_dict[Metrics.FP.value] = fp
    result_dict[Metrics.FN.value] = fn

    result_dict[Metrics.ACC.value] = accuracy
    result_dict[Metrics.SEN.value] = sensitivity
    result_dict[Metrics.SPE.value] = specificity
    result_dict[Metrics.PPV.value] = precision
    result_dict[Metrics.NPV.value] = npv
    result_dict[Metrics.F1.value] = f1_value
    result_dict[Metrics.F2.value] = f2_value

    try:
        auroc = roc_auc_score(y_real, y_pred)
        auprc = average_precision_score(y_real, y_pred)
        result_dict[Metrics.AUROC.value] = auroc
        result_dict[Metrics.AUPRC.value] = auprc
    except:
        return result_dict
    return result_dict


def export_metric(output_path, metric_dict, tag=None):
    '''
    성능결과를 csv로 출력
    :param output_path: 출력 경로
    :param metric_dict: 성능 결과
    :param tag: 첫 번째 컬럼에 출력할 tag
    :return:
    '''
    head = 'tag'
    line = tag if tag is not None else ' '
    for key, value in metric_dict.items():
        head = '%s,%s' % (head, key)
        if type(value) == int:
            line = '%s,%d' % (line, value)
        else:
            line = '%s,%.3f' % (line, value)
    head = '%s\n' % head
    line = '%s\n' % line

    if not os.path.exists(output_path):
        with open(output_path, 'w') as f:
            f.write(head)

    with open(output_path, 'a') as f:
        f.write(line)
