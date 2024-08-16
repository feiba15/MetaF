import torch
import os
import sys
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from loss.class_balanced_loss import CE_weight
from collections import OrderedDict

from utils.function import init_logging, init_environment, get_lr, \
    print_loss_sometime
from utils.metric import mean_class_recall
import config
import dataset
import model
from loss import class_balanced_loss
import torch.distributed as dist
import json
from al import al_selection
import warnings
warnings.filterwarnings('ignore')

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def sen(Y_test, Y_pred, n):  # n为分类数

    sen = []
    con_mat = confusion_matrix(Y_test, Y_pred, labels=[0,1,2,3,4,5,6])
    for i in range(n):
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        sen1 = tp / (tp + fn + 0.000000001)
        sen.append(sen1)

    return sen

def pre(Y_test, Y_pred, n):
    pre = []
    con_mat = confusion_matrix(Y_test, Y_pred, labels=[0,1,2,3,4,5,6])
    for i in range(n):
        tp = con_mat[i][i]
        fp = np.sum(con_mat[:, i]) - tp
        pre1 = tp / (tp + fp + 0.000000001)
        pre.append(pre1)

    return pre

def spe(Y_test, Y_pred, n):
    spe = []
    con_mat = confusion_matrix(Y_test, Y_pred, labels=[0,1,2,3,4,5,6])
    for i in range(n):
        number = np.sum(con_mat[:, :])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        fp = np.sum(con_mat[:, i]) - tp
        tn = number - tp - fn - fp
        spe1 = tn / (tn + fp + 0.000000001)
        spe.append(spe1)

    return spe

configs = config.Config()
configs_dict = configs.get_config()
# Load hyper parameter from config file
exp = configs_dict["experiment_index"]
cuda_ids = configs_dict["cudas"]
num_workers = configs_dict["num_workers"]
seed = configs_dict["seed"]
n_epochs = configs_dict["n_epochs"]
log_dir = configs_dict["log_dir"]
model_dir = configs_dict["model_dir"]
batch_size = configs_dict["batch_size"]
learning_rate = configs_dict["learning_rate"]
backbone = configs_dict["backbone"]
eval_frequency = configs_dict["eval_frequency"]
resume = configs_dict["resume"]
optimizer = configs_dict["optimizer"]
initialization = configs_dict["initialization"]
num_classes = configs_dict["num_classes"]
iter_fold = configs_dict["iter_fold"]
loss_fn = configs_dict["loss_fn"]
eval = configs_dict['eval']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
init_environment(seed=seed, cuda_id=cuda_ids)
_print = init_logging(log_dir, exp).info
configs.print_config(_print)
tf_log = os.path.join(log_dir, exp)
writer = SummaryWriter(log_dir=tf_log)

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

cudnn.benchmark = False  # True
# cudnn.deterministic = True

# Pre-peocessed input image
if backbone in ["resnet50", "resnet18"]:
    re_size = 300
    input_size = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
elif backbone in ["NASNetALarge", "PNASNet5Large"]:
    re_size = 441
    input_size = 331
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
else:
    _print("Need backbone")
    sys.exit(-1)

_print("=> Image resize to {} and crop to {}".format(re_size, input_size))

train_transform = transforms.Compose([
    transforms.Resize(re_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.02, 0.02, 0.02, 0.01),
    transforms.RandomRotation([-180, 180]),
    transforms.RandomAffine([-180, 180], translate=[0.1, 0.1],
                            scale=[0.7, 1.3]),
    transforms.RandomCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
    ])
val_transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
    ])

input_channel = 3
trainset = dataset.Skin7(root="./data/Keratitis/", iter_fold=iter_fold, train=True,
                         transform=train_transform)
valset = dataset.Skin7(root="./data/Keratitis/", iter_fold=iter_fold, train=False,
                       transform=val_transform)


net = model.Network(backbone=backbone, num_classes=num_classes,
                    input_channel=input_channel, pretrained=initialization)


start_epoch = 0
if resume:
    _print("=> Resume from model at epoch {}".format(resume))
    ckpt = torch.load(resume_path)
    new_state_dict_1 = OrderedDict()
    for k, v in ckpt.items():
        name = k.replace('module.', '')
        new_state_dict_1[name] = v.cpu()
    # net.load_state_dict(new_state_dict_1)
    net_state_dict = net.state_dict()
    new_state_dict = OrderedDict()
    for k, v in net_state_dict.items():
        name = k.replace('module.','')
        new_state_dict[name] = 0.2 * net_state_dict[name] + 0.8 * new_state_dict_1[name]
    net.load_state_dict(new_state_dict)

    # net.load_state_dict({k.replace('module.', ''): v for k,v in ckpt.items()})
    start_epoch = 0#int(resume) + 1
else:
    _print("Train from scrach!!")

_print("=> Using device ids: {}".format(cuda_ids))
device_ids = list(range(len(cuda_ids.split(","))))
train_sampler = val_sampler = None
if len(device_ids) == 1:
    _print("Model single cuda")
    net = net.to(device)
else: ###################################改到这里了
    _print("Model parallel !!")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    gpu = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
    torch.distributed.init_process_group(backend="nccl", init_method='env://',
                                         world_size=world_size, rank=rank)
    # torch.distributed.barrier()

    # print("111111111111111111111111111111111111")
    # print(torch.cuda.current_device())
    # exit(0), find_unused_parameters=True

    net = net.to(device)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    # val_sampler = torch.utils.data.distributed.DistributedSampler(valset)
    if eval:
        train_sampler = torch.utils.data.SequentialSampler(trainset)
    val_sampler = torch.utils.data.SequentialSampler(valset)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[gpu], find_unused_parameters=True)
    # net = nn.DataParallel(net, device_ids=device_ids).to(device)

_print("=> iter_fold is {}".format(iter_fold))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        pin_memory=True,
                                          num_workers=num_workers,
                                          sampler=train_sampler,drop_last=False)
if eval:
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              sampler=train_sampler, drop_last=False)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                        pin_memory=True,
                                        num_workers=num_workers,
                                        sampler=val_sampler,
                                        drop_last=False)


# Loss.to(device)
if loss_fn == "WCE":
    _print("Loss function is WCE")
    # weights = [0.036, 0.002, 0.084, 0.134, 0.037, 0.391, 0.316]
    # class_weights = torch.FloatTensor(weights).cuda()
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    cls_num_list = [275, 341, 523, 725]
    criterion = CE_weight(cls_num_list)

    sen_class_list = [1116, 748]
    sen_class_list = torch.cuda.FloatTensor(sen_class_list)
    # weight of each class for imbalance dataset
    sen_class_weight = torch.cuda.FloatTensor(1.0 / sen_class_list)
    sen_class_weight = (sen_class_weight / sen_class_weight.sum()) * len(sen_class_list)
    criterion_sen = nn.CrossEntropyLoss(weight=sen_class_weight)
elif loss_fn == "CE":
    _print("Loss function is CE")
    criterion = nn.CrossEntropyLoss()
    criterion_sen = nn.CrossEntropyLoss()
else:
    _print("Need loss function.")



# Optmizer
scheduler = None
if optimizer == "SGD":
    _print("=> Using optimizer SGD with lr:{:.4f}".format(learning_rate))
    opt = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode='min', factor=0.1, patience=50, verbose=True,
                threshold=1e-4)
elif optimizer == "Adam":
    _print("=> Using optimizer Adam with lr:{:.4f}".format(learning_rate))
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate,
                           betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
else:
    _print("Need optimizer")
    sys.exit(-1)

desc = "Exp-{}-Train".format(exp)
sota = {}
sota["epoch"] = start_epoch
sota["mcr"] = -1.0


def EO_calculate(prob_list, i, j, number_total_list):
    abs_sum = 0
    count = 0.
    for index in range(4):
        for prob_index in range(2):
            if number_total_list[index][i] < 3 or number_total_list[index][j] < 3:
                continue
            count+=1
            abs_sum+=abs(prob_list[prob_index][index][i]-prob_list[prob_index][index][j])
        # abs_sum+=abs(prob_list_right[index][i]-prob_list_right[index][j])
        # abs_sum+=abs(prob_list_wrong[index][i]-prob_list_wrong[index][j])
    if count == 0:
        return 0
    result = abs_sum/count
    # result = (abs(prob_list_right[0][i]-prob_list_right[0][j]) \
    #         + abs(prob_list_right[1][i]-prob_list_right[1][j]) \
    #          + abs(prob_list_wrong[0][i] - prob_list_wrong[0][j]) \
    #          + abs(prob_list_wrong[1][i] - prob_list_wrong[1][j]))/4.0
    return result


def computeEO(sensitive_targets, targets, outputs, num_sensitive_classes):
    # prob_list_right = []
    # prob_list_wrong = []
    prob_list = []
    for pred_type_index in range(2):
        prob_list.append([])
    number_total_list = []
    for type_index in range(4):
        for pred_type_index in range(2):
            prob_list[pred_type_index].append([])
        number_total_list.append([])
        for index in range(num_sensitive_classes):
            for pred_type_index in range(2):
                prob_list[pred_type_index][type_index].append(0)
            # prob_list_right[type_index].append(0)
            # prob_list_wrong[type_index].append(0)
            number_total_list[type_index].append(0)
    sensitive_index = 1    ### 0为年龄，1为性别
    for index in range(len(targets)):
        target = int(targets[index])
        sensitive_target = int(sensitive_targets[index])
        output_max_index = outputs[index]
        number_total_list[target][sensitive_target] += 1.
        if output_max_index == target:
            prob_list[0][target][sensitive_target]+=1
        else:
            prob_list[1][target][sensitive_target] += 1
        # if output_max_index == target:
        #     prob_list_right[target][sensitive_target] += 1.
        # else:
        #     prob_list_wrong[target][sensitive_target] += 1.

    for i in range(len(number_total_list)):
        for j in range(len(number_total_list[0])):
            for k in range(2):
                prob_list[k][i][j]/=number_total_list[i][j]+0.000000001
            # prob_list_right[i][j] /= number_total_list[i][j]+0.000000001
            # prob_list_wrong[i][j] /= number_total_list[i][j]+0.000000001
    EO_list = []
    for i in range(len(number_total_list[0])):
        for j in range(len(number_total_list[0])):
            if i>=j:
                continue
            EO = EO_calculate(prob_list, i, j, number_total_list)
            EO_list.append(EO)

    # print("111111111111111111111111111111111111111")
    # print(sensitive_targets)
    # print(prob_list)
    # print(EO_list)
    # exit(0)
    # print(prob_list)
    # print(EO_list)
    return max(EO_list)


if eval:
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            return json.JSONEncoder.default(self,obj)
    epoch=0
    net.eval()
    y_true = []
    y_pred = []
    y_pred_prob = []
    y_sex = []
    y_age = []
    count=0
    for _, (data, target, tone, age) in enumerate(valloader):
        count+=1
        data = data.to(device)
        pred_prob, predict_sensitive_prob, _ = net(data)
        y_pred_prob.extend(F.softmax(pred_prob, dim=-1).cpu().data.numpy())
        predict = torch.argmax(pred_prob, dim=1).cpu().data.numpy()
        y_pred.extend(predict)
        target = target.cpu().data.numpy()
        y_true.extend(target)
        # sex = sex.cpu().data.numpy()
        # y_sex.extend(sex)
        age = age.cpu().data.numpy()
        y_age.extend(age)



    data_indices_1, data_indices_2 = al_selection(net, valloader)
    data_list_1 = np.array(data_indices_1).tolist()
    data_list_2 = np.array(data_indices_2).tolist()
    # split_model_wrong_data_list = json.load(open("src/total_test_sen_wrong_id_train_part1.json"))
    # split_model_wrong_data_list = json.load(open("src/total_test_sen_wrong_id_train.json"))
    # split_model_wrong_data_list = json.load(open("src/total_test_sen_wrong_id.json"))
    if is_main_process():
        # data_list = list(set(data_list_1).union(set(data_list_2)))
        # data_list = list(set(split_model_wrong_data_list).union(set(data_list)))
        data_list = []#split_model_wrong_data_list

        result_saved_path = "src/tem_record_baseline2.csv"
        result_file = open(result_saved_path, 'w')
        result_dict = {}
        result_dict['target'] = y_true
        result_dict['output'] = y_pred
        total_count = 0
        line_count = -1
        correctness_count = 0
        # y_true_new = [y_true[index] for index in range(len(y_true)) if index not in data_list]
        # y_pred_new = [y_pred[index] for index in range(len(y_pred)) if index not in data_list]
        # y_pred_prob_new = [y_pred_prob[index] for index in range(len(y_pred_prob)) if index not in data_list]
        # y_sex_new = [y_sex[index] for index in range(len(y_sex)) if index not in data_list]
        # y_age_new = [y_age[index] for index in range(len(y_age)) if index not in data_list]

        for line in open(src_path, 'r'):
            line_count+=1
            if line_count == 0:
                continue
            line_name = line.split(',')[0]
            if y_true[line_count-1] != y_pred[line_count-1]:
                correctness = 0
            # elif (line_count-1) in data_list:
            #     total_count+=1
            #     correctness = 0
            #     # result_file.write(line_name + ',' + str(correctness) + ',1' + ',1' + '\n')
            #     # result_file.write(line_name + ',' + str(correctness) + ',1' + ',1' + '\n')
            #     # result_file.write(line_name + ',' + str(correctness) + ',1' + ',1' + '\n')
            #     # result_file.write(line_name + ',' + str(correctness) + ',1' + ',1' + '\n')
            #     # result_file.write(line_name + ',' + str(correctness) + ',1' + ',1' + '\n')
            #     # result_file.write(line_name + ',' + str(correctness) + ',1' + ',1' + '\n')
            #     # continue
            else:
                correctness = 1
            if (line_count-1) not in data_list:
                correctness_count+=correctness
                # result_file.write(line.strip()+","+str(0.95)+"\n") #去掉bus后,0.65最佳
            else:
                # result_file.write(line.strip() + "," + str(1.0) + "\n")
                # result_file.write(line.strip() + "," + str(1.0) + "\n")
                1+1
                # result_file.write(line)
                # result_file.write(line)
                # result_file.write(line_name + ',' + str(correctness) + ',1' + ',1' + '\n')
            # line_list = line.strip().split(',')
            result_file.write(line.strip() + "," + str(y_pred[line_count-1]) + "\n")


        result_file.close()
        print("111111111111111111111111")
        print(len(data_list_1)*1./line_count)
        print(len(data_list_2)*1./line_count)
        # print(len(split_model_wrong_data_list)*1./line_count)
        print(len(data_list)*1./line_count)
        print(correctness_count*1.0/(line_count-len(data_list)))
        # print(data_list)
        # print(len(data_list))
        # print(total_count)
        # exit(0)
        # json.dump(result_dict, result_file, cls=NpEncoder)

        eo = computeEO(y_age, y_true, y_pred, num_sensitive_classes=2)
        acc = accuracy_score(y_true, y_pred)
        mcr = mean_class_recall(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro")
        recall = recall_score(y_true, y_pred, average="macro")
        f1 = f1_score(y_true, y_pred, average="macro")
        sen_score = np.mean(sen(y_true, y_pred, 4))
        print(sen(y_true, y_pred, 4))
        spe_score = np.mean(spe(y_true, y_pred, 4))
        print(spe(y_true, y_pred, 4))
        auc = roc_auc_score(y_true, y_pred_prob, labels=[0, 1, 2, 3], average="macro", multi_class='ovo')
        print("Total Eval: ")
        print("=> Epoch:{} - train eo: {:.4f}".format(epoch, eo))
        print("=> Epoch:{} - train acc: {:.4f}".format(epoch, acc))
        print("=> Epoch:{} - train mcr: {:.4f}".format(epoch, mcr))
        print("=> Epoch:{} - train pre: {:.4f}".format(epoch, prec))
        print("=> Epoch:{} - train rec: {:.4f}".format(epoch, recall))
        print("=> Epoch:{} - train f1: {:.4f}".format(epoch, f1))
        print("=> Epoch:{} - train sen: {:.4f}".format(epoch, sen_score))
        print("=> Epoch:{} - train spe: {:.4f}".format(epoch, spe_score))
        print("=> Epoch:{} - train auc: {:.4f}".format(epoch, auc))


        y_true_new = [y_true[index] for index in range(len(y_true)) if index not in data_list]
        y_pred_new = [y_pred[index] for index in range(len(y_pred)) if index not in data_list]
        y_pred_prob_new = [y_pred_prob[index] for index in range(len(y_pred_prob)) if index not in data_list]
        y_sex_new = [y_sex[index] for index in range(len(y_sex)) if index not in data_list]
        y_age_new = [y_age[index] for index in range(len(y_age)) if index not in data_list]

        eo = computeEO(y_age_new, y_true_new, y_pred_new, num_sensitive_classes=2)
        acc = accuracy_score(y_true_new, y_pred_new)
        mcr = mean_class_recall(y_true_new, y_pred_new)
        prec = precision_score(y_true_new, y_pred_new, average="macro")
        recall = recall_score(y_true_new, y_pred_new, average="macro")
        f1 = f1_score(y_true_new, y_pred_new, average="macro")
        sen_score = np.mean(sen(y_true_new, y_pred_new, 4))
        print(sen(y_true_new, y_pred_new, 4))
        spe_score = np.mean(spe(y_true_new, y_pred_new, 4))
        print(spe(y_true_new, y_pred_new, 4))
        auc = roc_auc_score(y_true_new, y_pred_prob_new, labels=[0, 1, 2, 3], average="macro",
                            multi_class='ovo')
        print("Data Preserved: ")
        print("=> Epoch:{} - train eo: {:.4f}".format(epoch, eo))
        print("=> Epoch:{} - train acc: {:.4f}".format(epoch, acc))
        print("=> Epoch:{} - train mcr: {:.4f}".format(epoch, mcr))
        print("=> Epoch:{} - train pre: {:.4f}".format(epoch, prec))
        print("=> Epoch:{} - train rec: {:.4f}".format(epoch, recall))
        print("=> Epoch:{} - train f1: {:.4f}".format(epoch, f1))
        print("=> Epoch:{} - train sen: {:.4f}".format(epoch, sen_score))
        print("=> Epoch:{} - train spe: {:.4f}".format(epoch, spe_score))
        print("=> Epoch:{} - train auc: {:.4f}".format(epoch, auc))


        y_true_new = [y_true[index] for index in range(len(y_true)) if index in data_list]
        y_pred_new = [y_pred[index] for index in range(len(y_pred)) if index in data_list]
        y_pred_prob_new = [y_pred_prob[index] for index in range(len(y_pred_prob)) if index in data_list]
        y_sex_new = [y_sex[index] for index in range(len(y_sex)) if index in data_list]
        y_age_new = [y_age[index] for index in range(len(y_age)) if index in data_list]

        eo = computeEO(y_age_new, y_true_new, y_pred_new, num_sensitive_classes=2)
        acc = accuracy_score(y_true_new, y_pred_new)
        mcr = mean_class_recall(y_true_new, y_pred_new)
        prec = precision_score(y_true_new, y_pred_new, average="macro")
        recall = recall_score(y_true_new, y_pred_new, average="macro")
        f1 = f1_score(y_true_new, y_pred_new, average="macro")
        sen_score = np.mean(sen(y_true_new, y_pred_new, 4))
        spe_score = np.mean(spe(y_true_new, y_pred_new, 4))
        auc = roc_auc_score(y_true_new, y_pred_prob_new, labels=[0, 1, 2, 3], average="macro",
                            multi_class='ovo')
        print("Data Preserved: ")
        print("=> Epoch:{} - train eo: {:.4f}".format(epoch, eo))
        print("=> Epoch:{} - train acc: {:.4f}".format(epoch, acc))
        print("=> Epoch:{} - train mcr: {:.4f}".format(epoch, mcr))
        print("=> Epoch:{} - train pre: {:.4f}".format(epoch, prec))
        print("=> Epoch:{} - train rec: {:.4f}".format(epoch, recall))
        print("=> Epoch:{} - train f1: {:.4f}".format(epoch, f1))
        print("=> Epoch:{} - train sen: {:.4f}".format(epoch, sen_score))
        print("=> Epoch:{} - train spe: {:.4f}".format(epoch, spe_score))
        print("=> Epoch:{} - train auc: {:.4f}".format(epoch, auc))


    # print(data_indices)
    exit(0)

for epoch in range(start_epoch+1, n_epochs+1):
    if epoch == 500:
        break
    net.train()
    losses = []
    train_sampler.set_epoch(epoch)
    for batch_idx, (data, target, tone, age) in enumerate(trainloader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        age = age.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            predict, predict_sensitive = net(data, is_filter=False)
            opt.zero_grad()
            loss = criterion(predict, target, epoch)
            loss_sen = criterion_sen(predict_sensitive, age)
            if True:#epoch>200:#
                sen_w = (epoch-100)*1.0/100
                (0.00000000000 * loss_sen+0*loss).backward(create_graph=True)
                for name, param in net.named_parameters():
                    name_list = name.split('.')
                    if len(name_list) >= 3 and \
                        name_list[2] != 'sensitive_classifier' \
                        and name_list[2] != 'classifier':
                        if param.grad is not None:
                            # print(param.grad)-0.00005  -0.0000005  055:1_0.001 057:10_-0.01
                            # param.grad *= -0.78
                            # param.grad *= -1.0#-0.65#-0.1#-0.42#
                            param.grad *= -0.1
            src_w = 1.5+(epoch - 10) * 1.0 / 10
            (0*loss_sen+1.0*loss).backward()
            opt.step()
        losses.append(loss.item())



    # print to log
    dicts = {
        "epoch": epoch, "n_epochs": n_epochs, "loss": loss.item()
    }
    if is_main_process():
        print_loss_sometime(dicts, _print=_print)

    train_avg_loss = np.mean(losses)


    # opt.param_groups[0]['lr'] *= 2

    if scheduler is not None:
        scheduler.step(train_avg_loss)

    if is_main_process():
        writer.add_scalar("Lr", get_lr(opt), epoch)
        writer.add_scalar("Loss/train/", train_avg_loss, epoch)

    if epoch % eval_frequency == 0:
        with torch.no_grad():
            net.eval()
            y_true = []
            y_pred = []
            y_pred_prob = []
            y_sex = []
            y_age = []
            for _, (data, target, tone, age) in enumerate(trainloader):
                data = data.to(device)
                pred_prob, predict_sensitive_prob = net(data, is_filter=False)
                y_pred_prob.extend(F.softmax(pred_prob, dim=-1).cpu().data.numpy())
                predict = torch.argmax(pred_prob, dim=1).cpu().data.numpy()
                y_pred.extend(predict)
                target = target.cpu().data.numpy()
                y_true.extend(target)
                # sex = sex.cpu().data.numpy()
                # y_sex.extend(sex)
                age = age.cpu().data.numpy()
                y_age.extend(age)


            eo = computeEO(y_age, y_true, y_pred, num_sensitive_classes=2)
            acc = accuracy_score(y_true, y_pred)
            mcr = mean_class_recall(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average="macro")
            recall = recall_score(y_true, y_pred, average="macro")
            f1 = f1_score(y_true, y_pred, average="macro")

            sen_score = np.mean(sen(y_true, y_pred, 4))
            spe_score = np.mean(spe(y_true, y_pred, 4))

            # print(F.softmax(pred_prob, dim=-1))
            # print(y_true.size())
            # print(y_pred_prob.size())
            # exit(0)

            auc = roc_auc_score(y_true, y_pred_prob, labels=[0,1,2,3], average="macro", multi_class='ovo')
            if is_main_process():
                _print("=> Epoch:{} - train eo: {:.4f}".format(epoch, eo))
                _print("=> Epoch:{} - train acc: {:.4f}".format(epoch, acc))
                _print("=> Epoch:{} - train mcr: {:.4f}".format(epoch, mcr))
                _print("=> Epoch:{} - train pre: {:.4f}".format(epoch, prec))
                _print("=> Epoch:{} - train rec: {:.4f}".format(epoch, recall))
                _print("=> Epoch:{} - train f1: {:.4f}".format(epoch, f1))
                _print("=> Epoch:{} - train sen: {:.4f}".format(epoch, sen_score))
                _print("=> Epoch:{} - train spe: {:.4f}".format(epoch, spe_score))
                _print("=> Epoch:{} - train auc: {:.4f}".format(epoch, auc))
                writer.add_scalar("Eo/train/", eo, epoch)
                writer.add_scalar("Acc/train/", acc, epoch)
                writer.add_scalar("Mcr/train/", mcr, epoch)
                writer.add_scalar("Pre/train/", prec, epoch)
                writer.add_scalar("Rec/train/", recall, epoch)
                writer.add_scalar("F1/train/", f1, epoch)
                writer.add_scalar("Sen/train/", sen_score, epoch)
                writer.add_scalar("Spe/train/", spe_score, epoch)
                writer.add_scalar("AUC/train/", auc, epoch)
            y_true = []
            y_pred = []
            y_pred_prob = []
            y_sex = []
            y_age = []
            for _, (data, target, tone, age) in enumerate(valloader):
                data = data.to(device)
                pred_prob, predict_sensitive_prob = net(data, is_filter=False)
                y_pred_prob.extend(F.softmax(pred_prob, dim=-1).cpu().data.numpy())
                predict = torch.argmax(pred_prob, dim=1).cpu().data.numpy()
                y_pred.extend(predict)
                target = target.cpu().data.numpy()
                y_true.extend(target)
                # sex = sex.cpu().data.numpy()
                # y_sex.extend(sex)
                age = age.cpu().data.numpy()
                y_age.extend(age)

            # acc = accuracy_score(y_true, y_pred)valloader
            # mcr = mean_class_recall(y_true, y_pred)
            # if is_main_process():
            #     _print("=> Epoch:{} - val acc: {:.4f}".format(epoch, acc))
            #     _print("=> Epoch:{} - val mcr: {:.4f}".format(epoch, mcr))
            #     writer.add_scalar("Acc/val/", acc, epoch)
            #     writer.add_scalar("Mcr/val/", mcr, epoch)average="micro", multi_class='ovo'

            eo = computeEO(y_age, y_true, y_pred, num_sensitive_classes=2)
            acc = accuracy_score(y_true, y_pred)
            mcr = mean_class_recall(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average="macro")
            recall = recall_score(y_true, y_pred, average="macro")
            f1 = f1_score(y_true, y_pred, average="macro")

            sen_score = np.mean(sen(y_true, y_pred, 4))
            spe_score = np.mean(spe(y_true, y_pred, 4))
            # print(y_true[0])
            # print(y_pred_prob[0])
            # exit(0)
            auc = roc_auc_score(y_true, y_pred_prob, labels=[0,1,2,3], average="macro", multi_class='ovo')

            if is_main_process():
                _print("=> Epoch:{} - val eo: {:.4f}".format(epoch, eo))
                _print("=> Epoch:{} - val acc: {:.4f}".format(epoch, acc))
                _print("=> Epoch:{} - val mcr: {:.4f}".format(epoch, mcr))
                _print("=> Epoch:{} - val pre: {:.4f}".format(epoch, prec))
                _print("=> Epoch:{} - val rec: {:.4f}".format(epoch, recall))
                _print("=> Epoch:{} - val f1: {:.4f}".format(epoch, f1))
                _print("=> Epoch:{} - val sen: {:.4f}".format(epoch, sen_score))
                _print("=> Epoch:{} - val spe: {:.4f}".format(epoch, spe_score))
                _print("=> Epoch:{} - val auc: {:.4f}".format(epoch, auc))
                writer.add_scalar("Eo/val/", eo, epoch)
                writer.add_scalar("Acc/val/", acc, epoch)
                writer.add_scalar("Mcr/val/", mcr, epoch)
                writer.add_scalar("Pre/val/", prec, epoch)
                writer.add_scalar("Rec/val/", recall, epoch)
                writer.add_scalar("F1/val/", f1, epoch)
                writer.add_scalar("Sen/val/", sen_score, epoch)
                writer.add_scalar("Spe/val/", spe_score, epoch)
                writer.add_scalar("AUC/val/", auc, epoch)


                # eo = computeEO(y_age_new, y_true_new, y_pred_new, num_sensitive_classes=2)
                # acc = accuracy_score(y_true_new, y_pred_new)
                # mcr = mean_class_recall(y_true_new, y_pred_new)
                # prec = precision_score(y_true_new, y_pred_new, average="macro")
                # recall = recall_score(y_true_new, y_pred_new, average="macro")
                # f1 = f1_score(y_true_new, y_pred_new, average="macro")
                # sen_score = np.mean(sen(y_true_new, y_pred_new, 4))
                # spe_score = np.mean(spe(y_true_new, y_pred_new, 4))
                # auc = roc_auc_score(y_true_new, y_pred_prob_new, labels=[0, 1, 2, 3], average="macro",
                #                     multi_class='ovo')
                # _print("Data Preserved: ")
                # _print("=> Epoch:{} - train eo: {:.4f}".format(epoch, eo))
                # _print("=> Epoch:{} - train acc: {:.4f}".format(epoch, acc))
                # _print("=> Epoch:{} - train mcr: {:.4f}".format(epoch, mcr))
                # _print("=> Epoch:{} - train pre: {:.4f}".format(epoch, prec))
                # _print("=> Epoch:{} - train rec: {:.4f}".format(epoch, recall))
                # _print("=> Epoch:{} - train f1: {:.4f}".format(epoch, f1))
                # _print("=> Epoch:{} - train sen: {:.4f}".format(epoch, sen_score))
                # _print("=> Epoch:{} - train spe: {:.4f}".format(epoch, spe_score))
                # _print("=> Epoch:{} - train auc: {:.4f}".format(epoch, auc))
                # writer.add_scalar("Eo/val/", eo, epoch)
                # writer.add_scalar("Acc/val/", acc, epoch)
                # writer.add_scalar("Mcr/val/", mcr, epoch)
                # writer.add_scalar("Pre/val/", prec, epoch)
                # writer.add_scalar("Rec/val/", recall, epoch)
                # writer.add_scalar("F1/val/", f1, epoch)
                # writer.add_scalar("Sen/val/", sen_score, epoch)
                # writer.add_scalar("Spe/val/", spe_score, epoch)
                # writer.add_scalar("AUC/val/", auc, epoch)


                # y_true_new = [y_true[index] for index in range(len(y_true)) if index in data_list]
                # y_pred_new = [y_pred[index] for index in range(len(y_pred)) if index in data_list]
                # y_pred_prob_new = [y_pred_prob[index] for index in range(len(y_pred_prob)) if index in data_list]
                # y_sex_new = [y_sex[index] for index in range(len(y_sex)) if index in data_list]
                # y_age_new = [y_age[index] for index in range(len(y_age)) if index in data_list]

                # eo = computeEO(y_age_new, y_true_new, y_pred_new, num_sensitive_classes=2)
                # acc = accuracy_score(y_true_new, y_pred_new)
                # mcr = mean_class_recall(y_true_new, y_pred_new)
                # prec = precision_score(y_true_new, y_pred_new, average="macro")
                # recall = recall_score(y_true_new, y_pred_new, average="macro")
                # f1 = f1_score(y_true_new, y_pred_new, average="macro")
                # sen_score = np.mean(sen(y_true_new, y_pred_new, 4))
                # spe_score = np.mean(spe(y_true_new, y_pred_new, 4))
                # auc = roc_auc_score(y_true_new, y_pred_prob_new, labels=[0, 1, 2, 3], average="macro",
                #                     multi_class='ovo')

                # _print("Data Deleted: ")
                # _print("=> Epoch:{} - train eo: {:.4f}".format(epoch, eo))
                # _print("=> Epoch:{} - train acc: {:.4f}".format(epoch, acc))
                # _print("=> Epoch:{} - train mcr: {:.4f}".format(epoch, mcr))
                # _print("=> Epoch:{} - train pre: {:.4f}".format(epoch, prec))
                # _print("=> Epoch:{} - train rec: {:.4f}".format(epoch, recall))
                # _print("=> Epoch:{} - train f1: {:.4f}".format(epoch, f1))
                # _print("=> Epoch:{} - train sen: {:.4f}".format(epoch, sen_score))
                # _print("=> Epoch:{} - train spe: {:.4f}".format(epoch, spe_score))
                # _print("=> Epoch:{} - train auc: {:.4f}".format(epoch, auc))
                # writer.add_scalar("Eo/val/", eo, epoch)
                # writer.add_scalar("Acc/val/", acc, epoch)
                # writer.add_scalar("Mcr/val/", mcr, epoch)
                # writer.add_scalar("Pre/val/", prec, epoch)
                # writer.add_scalar("Rec/val/", recall, epoch)
                # writer.add_scalar("F1/val/", f1, epoch)
                # writer.add_scalar("Sen/val/", sen_score, epoch)
                # writer.add_scalar("Spe/val/", spe_score, epoch)
                # writer.add_scalar("AUC/val/", auc, epoch)

            # Val acc mcr > sota["mcr"] and
            if is_main_process():
                sota["mcr"] = mcr
                sota["epoch"] = epoch
                model_path = os.path.join(model_dir, str(exp), str(epoch))
                _print("=> Save model in {}".format(model_path))
                net_state_dict = net.state_dict()
                torch.save(net_state_dict, model_path)

if is_main_process():
    _print("=> Finish Training")
    _print("=> Best epoch {} with {} on Val: {:.4f}".format(sota["epoch"],
                                                            "sota",
                                                            sota["mcr"]))

