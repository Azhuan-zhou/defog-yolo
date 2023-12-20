from __future__ import division
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import cfg
from utils import util
from utils.dataset_fog import dataset_fog
from utils.util import read_class_names
from model.FY import YoloDIP
from matplotlib import pyplot as plt
from evaluate import _evaluate
from utils.loss import compute_loss
import torch.optim as optim


def train():
    # 确定使用GPU还是CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = read_class_names(cfg.YOLO.CLASSES)  # 获得类别名
    # 建立模型并初始化
    model_def = cfg.model_def

    model = YoloDIP(cfg_file=model_def, weights_path_yolov3=cfg.YOLO.pretrained_weights).to(
        device)  # 导入配置文件建立模型，并放入GPU中

    # Get dataloader
    dataset = dataset_fog('train')
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.n_cpu,
        collate_fn=dataset.collate_fn
    )

    dataset_eval = dataset_fog('eval')
    dataloader_eval = torch.utils.data.DataLoader(
        dataset_eval,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.n_cpu,
        collate_fn=dataset.collate_fn
    )

    anchors = cfg.YOLO.ANCHORS
    anchors = util.get_anchors(anchors)

    params = [p for p in model.parameters() if p.requires_grad]

    if model.yolov3.hyperparams['optimizer'] in [None, "adam"]:
        optimizer = optim.Adam(
            params,
            lr=model.yolov3.hyperparams['learning_rate'],
            weight_decay=model.yolov3.hyperparams['decay'],
        )
    elif model.yolov3.hyperparams['optimizer'] == "sgd":
        optimizer = optim.SGD(
            params,
            lr=model.yolov3.hyperparams['learning_rate'],
            weight_decay=model.yolov3.hyperparams['decay'],
            momentum=model.yolov3.hyperparams['momentum'])
    else:
        print("Unknown optimizer. Please choose between (adam, sgd).")

    map_epoch = []
    loss_epoch = []
    for epoch in range(cfg.epochs):

        # 训练
        #train_loss = train_epoch(epoch, model, optimizer, dataloader, device)
        # 一个epoch结束
        #print('Loss:{}'.format(train_loss / len(dataloader)))
        #loss_epoch.append(train_loss / len(dataloader))

        # 评估
        if (epoch + 1) % cfg.evaluation_interval == 0:
            # 并不是每个epoch结束后都进行评价，而是若干个epoch结束后做一次评价
            AP = eval_epoch(model, dataloader_eval, device, class_names)
            map_epoch.append(AP.mean())

        # 保存
        if epoch % cfg.checkpoint_interval == 0:
            torch.save(model.state_dict(),
                       "./checkpoints/FY_ckpt_{}.pth".format(epoch))
    # 画图
    plt.figure(figsize=(8, 8))
    plt.plot(range(1, cfg.epochs + 1), loss_epoch)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True)
    plt.savefig('./checkpoint/training process/epoch_loss.png')
    plt.close()
    plt.figure(figsize=(8, 8))
    plt.plot(range(1, cfg.epochs + 1), map_epoch)
    plt.xlabel('epoch')
    plt.ylabel('mAP')
    plt.grid(True)
    plt.savefig('./checkpoint/training process/epoch_mAP.png')
    plt.close()


def train_epoch(epoch, model, optimizer, dataloader, device):
    train_loss = 0
    print('epoch:{}'.format(epoch + 1))
    model.train()  # 切换到训练模式
    print('----Training Model----')
    for batch_i, loader in enumerate(tqdm(dataloader, desc="Training")):
        # 图片和标签做成变量
        images = loader[1].to(device)
        targets = loader[2].to(device)
        outputs = model(images)  # 前向传播

        loss, loss_components = compute_loss(outputs, targets, model)
        loss.requires_grad_(True)

        loss.backward()  # 根据损失函数更新梯度
        if batch_i % cfg.gradient_accumulations == 0:
            # 这里并非每次得到梯度就更新，而是累积若干次梯度才进行更新
            optimizer.step()
            optimizer.zero_grad()  # 梯度信息清零
        train_loss += loss.item()
    return train_loss


def eval_epoch(model, dataloader, device, class_names):
    # 并不是每个epoch结束后都进行评价，而是若干个epoch结束后做一次评价
    print("\n---- Evaluating Model ----")
    # Evaluate the model on the validation set 将模型放在验证集上进行评价
    precision, recall, AP, f1, ap_class = _evaluate(
        model=model,
        dataloader=dataloader,
        device=device,
        class_names=class_names,
        img_size=cfg.TEST.INPUT_SIZE,
        iou_thres=cfg.TEST.iou_threshold,
        conf_thres=cfg.TEST.conf_threshold,
        nms_thres=cfg.TEST.nms_threshold,
        verbose=True
    )
    return AP


if __name__ == '__main__':
    train()
