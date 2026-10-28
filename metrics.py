import torch
from torch import Tensor

'''
按元素逐像素对比 来判断是否预测正确或错误,通过对每个像素值的比较
# TP 预测和真实标签都为该类别的像素数。
# TN 预测和真实标签都不属于该类别的像素数。
# FN 预测不为该类别，但真实标签是该类别的像素数。
# FP 预测为该类别，但真实标签不属于该类别的像素数。 
get_table中pred和target传入的值是0或者1，也就是说，里面的公式都是合乎逻辑的
例如TP只有都返回的是1，才证明预测的是正确的，那也不对啊 ，都为0的话那也是预测正确啊？？？这怎么归纳？
对于这种情况下，预测的"错误"不会被算作TP，而是通过其它类别的度量（比如TN，FP，FN）来处理
'''


def get_table(pred: Tensor, target: Tensor):
    TP = (pred * target).sum()
    TN = ((1 - pred) * (1 - target)).sum()
    FN = ((1 - pred) * target).sum()
    FP = (pred * (1 - target)).sum()
    return TP, TN, FN, FP


##########
## 原型 ##
##########

def Iou_score(pred: Tensor, target: Tensor):
    TP, TN, FN, FP = get_table(pred, target)
    return TP / (TP + FP + FN + 1e-5)


def Dice_score(pred: Tensor, target: Tensor):
    # predictions=pred
    # gt_mask =target
    # gt_mask = gt_mask.float()
    # smooth = 1e-5
    # intersect = torch.sum(predictions * gt_mask, dim=(-1,-2))
    # y_sum = torch.sum(gt_mask * gt_mask, dim=(-1,-2))
    # z_sum = torch.sum(predictions * predictions, dim=(-1,-2))
    # batch_dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    # return batch_dice
    TP, TN, FN, FP = get_table(pred, target)
    return (2. * TP) / (2. * TP + FP + FN + 1e-5)


def Spe_score(pred: Tensor, target: Tensor):
    TP, TN, FN, FP = get_table(pred, target)
    return TN / (TN + FP + 1e-5)


def Sen_score(pred: Tensor, target: Tensor):
    TP, TN, FN, FP = get_table(pred, target)
    return TP / (TP + FN + 1e-5)


def Acc_score(pred: Tensor, target: Tensor):
    TP, TN, FN, FP = get_table(pred, target)
    return (TP + TN) / (TP + TN + FP + FN + 1e-5)


def Pre_score(pred: Tensor, target: Tensor):
    TP, TN, FN, FP = get_table(pred, target)
    return TP / (TP + FP + 1e-5)


def F1_score(pred: Tensor, target: Tensor):
    """
    计算F1分数 (Dice系数的另一种表达)
    F1 = 2 * (precision * recall) / (precision + recall)
    """
    precision = Pre_score(pred, target)
    recall = Sen_score(pred, target)
    return 2 * (precision * recall) / (precision + recall + 1e-5)


#####################################
#    获取可用的值，针对青光眼数据集    #
# [损失（1-均分），视盘分数，视杯分数] #
#####################################

# 获取dice。
def get_Dice(input: Tensor, target: Tensor):
    assert input.size() == target.size()
    od = Dice_score(input[:, 1, ...], target[:, 1, ...])
    oc = Dice_score(input[:, 2, ...], target[:, 2, ...])
    return [(1. - (oc + od) / 2), od, oc]


# 获取IOU，Jaccard
def get_Iou(input: Tensor, target: Tensor):
    assert input.size() == target.size()
    od = Iou_score(input[:, 1, ...], target[:, 1, ...])
    oc = Iou_score(input[:, 2, ...], target[:, 2, ...])
    return [(1. - (oc + od) / 2), od, oc]


# 获取Specificity
def get_Spe(input: Tensor, target: Tensor):
    assert input.size() == target.size()
    od = Spe_score(input[:, 1, ...], target[:, 1, ...])
    oc = Spe_score(input[:, 2, ...], target[:, 2, ...])
    return [(1. - (oc + od) / 2), od, oc]


# 获取sensitivity，Recall
def get_Sen(input: Tensor, target: Tensor):
    assert input.size() == target.size()
    od = Sen_score(input[:, 1, ...], target[:, 1, ...])
    oc = Sen_score(input[:, 2, ...], target[:, 2, ...])
    return [(1. - (oc + od) / 2), od, oc]


# 获取Accuracy
def get_Acc(input: Tensor, target: Tensor):
    assert input.size() == target.size()
    od = Acc_score(input[:, 1, ...], target[:, 1, ...])
    oc = Acc_score(input[:, 2, ...], target[:, 2, ...])
    return [(1. - (oc + od) / 2), od, oc]


# 获取Precision
def get_Pre(input: Tensor, target: Tensor):
    assert input.size() == target.size()
    od = Pre_score(input[:, 1, ...], target[:, 1, ...])
    oc = Pre_score(input[:, 2, ...], target[:, 2, ...])
    return [(1. - (oc + od) / 2), od, oc]


# 获取F1分数
def get_F1(input: Tensor, target: Tensor):
    """
    获取视盘和视杯的F1分数
    """
    assert input.size() == target.size()
    od = F1_score(input[:, 1, ...], target[:, 1, ...])
    oc = F1_score(input[:, 2, ...], target[:, 2, ...])
    return [(1. - (oc + od) / 2), od, oc]


# 修改：获取视盘(OD)的Dice、Precision、Recall和IoU
def get_OD_metrics(input: Tensor, target: Tensor):
    """
    获取视盘(OD)的Dice、Precision、Recall和IoU值

    参数:
        input: 预测结果 [B,C,H,W]
        target: 真实标签 [B,C,H,W]

    返回:
        包含视盘Dice、Precision、Recall和IoU值的字典
    """
    assert input.size() == target.size()
    od_dice = Dice_score(input[:, 1, ...], target[:, 1, ...])
    od_precision = Pre_score(input[:, 1, ...], target[:, 1, ...])
    od_recall = Sen_score(input[:, 1, ...], target[:, 1, ...])
    od_iou = Iou_score(input[:, 1, ...], target[:, 1, ...])  # 修改：miou -> iou
    od_f1 = F1_score(input[:, 1, ...], target[:, 1, ...])

    return {
        'dice': od_dice.item(),
        'precision': od_precision.item(),
        'recall': od_recall.item(),
        'iou': od_iou.item(),  # 修改：miou -> iou
        'f1': od_f1.item()
    }


# 修改：获取视杯(OC)的Dice、Precision、Recall和IoU
def get_OC_metrics(input: Tensor, target: Tensor):
    """
    获取视杯(OC)的Dice、Precision、Recall和IoU值

    参数:
        input: 预测结果 [B,C,H,W]
        target: 真实标签 [B,C,H,W]

    返回:
        包含视杯Dice、Precision、Recall和IoU值的字典
    """
    assert input.size() == target.size()
    oc_dice = Dice_score(input[:, 2, ...], target[:, 2, ...])
    oc_precision = Pre_score(input[:, 2, ...], target[:, 2, ...])
    oc_recall = Sen_score(input[:, 2, ...], target[:, 2, ...])
    oc_iou = Iou_score(input[:, 2, ...], target[:, 2, ...])  # 修改：miou -> iou
    oc_f1 = F1_score(input[:, 2, ...], target[:, 2, ...])

    return {
        'dice': oc_dice.item(),
        'precision': oc_precision.item(),
        'recall': oc_recall.item(),
        'iou': oc_iou.item(),  # 修改：miou -> iou
        'f1': oc_f1.item()
    }


# 新增：计算真正的MIoU（所有类别IoU的平均值）
def calculate_miou(input: Tensor, target: Tensor, include_background=False):
    """
    计算真正的MIoU（所有类别IoU的平均值）

    参数:
        input: 预测结果 [B,C,H,W]
        target: 真实标签 [B,C,H,W]
        include_background: 是否包含背景类在计算中

    返回:
        MIoU值
    """
    assert input.size() == target.size()

    # 获取各类别的IoU
    iou_values = get_Iou(input, target)  # [背景, 视盘, 视杯]

    if include_background:
        # 包含背景类的MIoU
        return sum(iou_values) / len(iou_values)
    else:
        # 不包含背景类的MIoU（只计算视盘和视杯）
        return (iou_values[1] + iou_values[2]) / 2


# test
# 计算类别分布
# 这是get
def get_class_distribution(pred: Tensor, target: Tensor):
    background = torch.sum(target == 0).item()
    optic_disc = torch.sum(target == 1).item()
    optic_cup = torch.sum(target == 2).item()
    return {'background': background, 'optic_disc': optic_disc, 'optic_cup': optic_cup}


class Scoring():
    def __init__(self, classes) -> None:
        self.number = 0
        self.dice = torch.zeros(classes)
        self.iou = torch.zeros(classes)
        self.spe = torch.zeros(classes)
        self.sen = torch.zeros(classes)
        self.acc = torch.zeros(classes)
        self.pre = torch.zeros(classes)
        self.f1 = torch.zeros(classes)  # 新增F1分数跟踪

        # 新增：真正的MIoU计算
        self.miou_with_bg = 0.0  # 包含背景的MIoU
        self.miou_without_bg = 0.0  # 不包含背景的MIoU

        # 修改：单独跟踪视盘和视杯的详细指标
        self.od_dice = 0.0
        self.od_precision = 0.0
        self.od_recall = 0.0
        self.od_iou = 0.0  # 修改：od_miou -> od_iou
        self.od_f1 = 0.0

        self.oc_dice = 0.0
        self.oc_precision = 0.0
        self.oc_recall = 0.0
        self.oc_iou = 0.0  # 修改：oc_miou -> oc_iou
        self.oc_f1 = 0.0

    def add(self, pred, target):
        self.number += 1
        dice = torch.tensor(get_Dice(pred, target))
        self.dice += dice
        self.iou += torch.tensor(get_Iou(pred, target))
        self.spe += torch.tensor(get_Spe(pred, target))
        self.sen += torch.tensor(get_Sen(pred, target))
        self.acc += torch.tensor(get_Acc(pred, target))
        self.pre += torch.tensor(get_Pre(pred, target))
        self.f1 += torch.tensor(get_F1(pred, target))  # 新增F1分数计算

        # 新增：计算真正的MIoU
        miou_with_bg = calculate_miou(pred, target, include_background=True)
        miou_without_bg = calculate_miou(pred, target, include_background=False)
        # 确保是标量值
        if isinstance(miou_with_bg, torch.Tensor):
            self.miou_with_bg += miou_with_bg.item()
        else:
            self.miou_with_bg += miou_with_bg

        if isinstance(miou_without_bg, torch.Tensor):
            self.miou_without_bg += miou_without_bg.item()
        else:
            self.miou_without_bg += miou_without_bg

        # 修改：更新视盘和视杯的详细指标
        od_metrics = get_OD_metrics(pred, target)
        oc_metrics = get_OC_metrics(pred, target)

        self.od_dice += od_metrics['dice']
        self.od_precision += od_metrics['precision']
        self.od_recall += od_metrics['recall']
        self.od_iou += od_metrics['iou']  # 修改：miou -> iou
        self.od_f1 += od_metrics['f1']

        self.oc_dice += oc_metrics['dice']
        self.oc_precision += oc_metrics['precision']
        self.oc_recall += oc_metrics['recall']
        self.oc_iou += oc_metrics['iou']  # 修改：miou -> iou
        self.oc_f1 += oc_metrics['f1']

        return dice

    def result(self):
        # 这三个数值分别表示 背景、视盘（OD）和 视杯（OC）这三个类别的评估指标结果。
        result_dict = {
            'dice': [round(i, 5) for i in (self.dice / self.number).numpy().tolist()],
            'iou': [round(i, 5) for i in (self.iou / self.number).numpy().tolist()],
            # 'spe': [round(i, 5) for i in (self.spe / self.number).numpy().tolist()],
            'sen': [round(i, 5) for i in (self.sen / self.number).numpy().tolist()],
            # 'acc': [round(i, 5) for i in (self.acc / self.number).numpy().tolist()],
            'pre': [round(i, 5) for i in (self.pre / self.number).numpy().tolist()],
            'f1': [round(i, 5) for i in (self.f1 / self.number).numpy().tolist()],  # 新增F1分数结果

            # 新增：真正的MIoU结果
            'miou': {
                'with_background': round(self.miou_with_bg / self.number, 5),
                'without_background': round(self.miou_without_bg / self.number, 5)
            }
        }

        # 修改：添加视盘和视杯的详细指标到结果字典
        result_dict.update({
            'od_metrics': {
                'dice': round(self.od_dice / self.number, 5),
                'precision': round(self.od_precision / self.number, 5),
                'recall': round(self.od_recall / self.number, 5),
                'iou': round(self.od_iou / self.number, 5),  # 修改：miou -> iou
                'f1': round(self.od_f1 / self.number, 5)
            },
            'oc_metrics': {
                'dice': round(self.oc_dice / self.number, 5),
                'precision': round(self.oc_precision / self.number, 5),
                'recall': round(self.oc_recall / self.number, 5),
                'iou': round(self.oc_iou / self.number, 5),  # 修改：miou -> iou
                'f1': round(self.oc_f1 / self.number, 5)
            }
        })

        return result_dict

