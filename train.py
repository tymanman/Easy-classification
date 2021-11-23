import argparse
import os
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as m
from dataset import train_dataloader,train_datasets, val_dataloader, val_datasets
import cfg
from lr_scheduler import adjust_learning_rate_cosine, adjust_learning_rate_step

#####build the network model
model = m.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(in_features=1280, out_features=cfg.NUM_CLASSES, bias=True)
print('****** Training {} ****** '.format(str(model)))
print('****** loading the Imagenet pretrained weights ****** ')
if cfg.FREEZE_LAYER_NUM>0:
    for index, child in enumerate(model.children()):
        if index <= cfg.FREEZE_LAYER_NUM:
            print(child)
            for param in child.parameters():
                param.requires_grad = False

###模型放置在gpu上进行计算
if torch.cuda.is_available():
    model.cuda()


##定义优化器与损失函数
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.LR)
# optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
optimizer = optim.SGD(model.parameters(), lr=cfg.LR,
                      momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)

criterion = nn.CrossEntropyLoss()
# criterion = nn.BCEWithLogitsLoss()
# criterion = LabelSmoothSoftmaxCE()
# criterion = LabelSmoothingCrossEntropy()

lr = cfg.LR

#每一个epoch含有多少个batch
batch_per_epoch = len(train_datasets)//cfg.BATCH_SIZE
## 训练max_epoch个epoch
max_iter = cfg.MAX_EPOCH * batch_per_epoch

epoch = 0

# cosine学习率调整
# warmup_epoch=5
# warmup_steps = warmup_epoch * batch_per_epoch
global_step = 0

# step 学习率调整参数
stepvalues = (10 * batch_per_epoch, 20 * batch_per_epoch, 30 * batch_per_epoch)
step_index = 0

model.train()
for iteration in range(max_iter):
    global_step += 1

    ##更新迭代器
    if iteration % batch_per_epoch == 0:
        # create batch iterator
        batch_iterator = iter(train_dataloader)
        loss = 0
        epoch += 1
        ###保存模型
        if epoch % cfg.EVAL_PERIOD == 0 and epoch > 0:
            model.eval()
            val_correct = 0
            for val_batch, val_labels in tqdm.tqdm(val_dataloader):
                out = model(val_batch.cuda())
                prediction = torch.max(out, 1)[1]
                val_correct += (prediction == val_labels.cuda()).sum()
            val_acc = (val_correct.float()) / len(val_datasets)
            torch.save(model.state_dict(), os.path.join(cfg.SAVE_FOLDER, 'epoch_{}_val_acc_{}.pth'.format(epoch, val_acc.cpu().data)))
            model.train()

    if iteration in stepvalues:
        step_index += 1
    lr = adjust_learning_rate_step(optimizer, cfg.LR, 0.1, epoch, step_index, iteration, batch_per_epoch)

    ## 调整学习率
    # lr = adjust_learning_rate_cosine(optimizer, global_step=global_step,
    #                           learning_rate_base=cfg.LR,
    #                           total_steps=max_iter,
    #                           warmup_steps=warmup_steps)


    ## 获取image 和 label
    # try:
    images, labels = next(batch_iterator)
    # except:
    #     continue

    ##在pytorch0.4之后将Variable 与tensor进行合并，所以这里不需要进行Variable封装
    if torch.cuda.is_available():
        images, labels = images.cuda(), labels.cuda()

    out = model(images)
    loss = criterion(out, labels)

    optimizer.zero_grad()  # 清空梯度信息，否则在每次进行反向传播时都会累加
    loss.backward()  # loss反向传播
    optimizer.step()  ##梯度更新

    prediction = torch.max(out, 1)[1]
    train_correct = (prediction == labels).sum()
    ##这里得到的train_correct是一个longtensor型，需要转换为float
    # print(train_correct.type())
    train_acc = (train_correct.float()) / cfg.BATCH_SIZE

    if iteration % 10 == 0:
        print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % batch_per_epoch) + '/' + repr(batch_per_epoch)
              + '|| Totel iter ' + repr(iteration) + ' || Loss: %.6f||' % (loss.item()) + 'ACC: %.3f ||' %(train_acc * 100) + 'LR: %.8f' % (lr))
