from __future__ import division

import os.path
from config import cfg
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset_fog import dataset_fog
from utils.utils import read_class_names, get_anchors
from model.FY import YoloDIP
from matplotlib import pyplot as plt
from evaluate import _evaluate
from utils.loss import compute_loss
import torch.optim as optim


def train():
    if not os.path.exists("./checkpoints/{}".format(cfg.time)):
        os.makedirs("./checkpoints/{}".format(cfg.time))
    if not os.path.exists("./checkpoints/training process/{}".format(cfg.time)):
        os.makedirs("./checkpoints/training process/{}".format(cfg.time))
    # 确定使用GPU还是CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = read_class_names(cfg.YOLO.CLASSES)  # 获得类别名
    # 建立模型并初始化
    model_def = cfg.MODEL.model_def

    model = YoloDIP(cfg_file=model_def).to(device)
    if cfg.TRAIN.scratch:
        model.init_weights(weights_path_yolov3=cfg.YOLO.pretrained_weights)
        print('train from scratch')
    else:
        weight = cfg.weight
        print(weight)
        model.load_state_dict(torch.load(weight, map_location=device))
        print('init with .pth')

    # parameters = model.named_parameters()
    # 打印参数及其梯度情况
    # for name,param in parameters:
    #    print(f"Parameter Name: {name}")
    #    print(f"Requires Gradient: {param.requires_grad}")
    #    print("---")
    # Get dataloader
    dataset = dataset_fog('train')
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.TRAIN.n_cpu,
        collate_fn=dataset.collate_fn
    )

    dataset_eval = dataset_fog('test')
    dataloader_eval = torch.utils.data.DataLoader(
        dataset_eval,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.TEST.n_cpu,
        collate_fn=dataset_eval.collate_fn
    )

    anchors = cfg.YOLO.ANCHORS
    anchors = get_anchors(anchors)

    params = [p for p in model.parameters() if p.requires_grad]

    if model.yolov3.hyperparams['optimizer'] in [None, "adam"]:
        optimizer = optim.Adam(
            params,
            lr=model.yolov3.hyperparams['learning_rate'],
            weight_decay=model.yolov3.hyperparams['decay']
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
    for epoch in range(cfg.TRAIN.epochs):

        # 训练
        train_loss = train_epoch(epoch, model, optimizer, dataloader, device)
        # 一个epoch结束
        print('Loss:{}'.format(train_loss / len(dataloader)))
        loss_epoch.append(train_loss / len(dataloader))

        # 评估
        if (epoch + 1) % cfg.evaluation_interval == 0:
            # 并不是每个epoch结束后都进行评价，而是若干个epoch结束后做一次评价
            AP = eval_epoch(model, dataloader_eval, device, class_names)
            map_epoch.append(AP.mean())

        # 保存
        if epoch % cfg.TEST.checkpoint_interval == 0:

            torch.save(model.state_dict(),
                       "./checkpoints/{}/FY_ckpt_{}_{}.pth".format(cfg.time,AP.mean(), epoch))
    # 画图
    plt.figure(figsize=(8, 8))
    plt.plot(range(1, cfg.TRAIN.epochs + 1), loss_epoch)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True)
    plt.savefig('./checkpoints/training process/{}/epoch_loss.png'.format(cfg.time))
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.plot(range(1, cfg.TRAIN.epochs + 1), map_epoch)
    plt.xlabel('epoch')
    plt.ylabel('mAP')
    plt.grid(True)
    plt.savefig('./checkpoints/training process/{}/epoch_mAP.png'.format(cfg.time))
    plt.close()


def train_epoch(epoch, model, optimizer, dataloader, device):
    train_loss = 0
    print('epoch:{}'.format(epoch + 1))
    model.train()  # 切换到训练模式
    print('----Training Model----')
    for batch_i, loader in enumerate(tqdm(dataloader, desc="Training")):
        batches_done = len(dataloader) * epoch + batch_i

        # 图片和标签做成变量
        images = loader[1].to(device)
        targets = loader[2].to(device)
        outputs, _ = model(images)  # 前向传播

        loss, loss_components = compute_loss(outputs, targets, model)
        loss.requires_grad_(True)

        loss.backward()  # 根据损失函数更新梯度

        if batches_done % model.yolov3.hyperparams['subdivisions'] == 0:
            # Adapt learning rate
            # Get learning rate defined in cfg
            lr = model.yolov3.hyperparams['learning_rate']
            if batches_done < model.yolov3.hyperparams['burn_in']:
                # Burn in
                lr *= (batches_done / model.yolov3.hyperparams['burn_in'])
            else:
                # Set and parse the learning rate to the steps defined in the cfg
                for threshold, value in model.yolov3.hyperparams['lr_steps']:
                    if batches_done > threshold:
                        lr *= value

            for g in optimizer.param_groups:
                g['lr'] = lr

            # Run optimizer
            optimizer.step()
            # Reset gradients
            optimizer.zero_grad()
            print('backward:{},{}'.format(epoch + 1, batches_done))
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
    print(cfg)
    train()
