import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
import torch
import torch.nn as nn
# import visdom
import random
import sys
import torch.optim as optim
Root_Dir = os.path.abspath("../../")  # obtain the absolute path of a directory relative to the current working directory
sys.path.append(Root_Dir)
from SSIM1 import SSIM_loss
from model.model1 import MCNN_new
from load_data import CrowdDataset

if __name__ == "__main__":
    torch.backends.cudnn.enabled = True  # cuDNN is not available
    # vis = visdom.Visdom(env='MAN')
    # gpu_ids = [0,1]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # *************************
    # 原有代码书写方式
    # MAN = MCNN_new().cuda()  # 会将所有模型参数移动到GPU上进行计算
    # MAN = MC.to(device)

    # 此为修改代码，保证无论时GPU还是CPU均不会产生错误
    MAN = MCNN_new().to(device)
    # ***************************

    # if len(gpu_ids) > 1:
    #     MAN = torch.nn.DataParallel(MAN,device_ids=gpu_ids)

    # 修改如上，且两者需要统一，即参数计算、loss计算等均需要统一（GPU/CPU）计算设备
    # 原
    # criterion = nn.MSELoss(size_average=False).cuda()
    # criterion1 = SSIM_loss().cuda(device)
    # 修改
    criterion = nn.MSELoss(size_average=False).to(device)  #python1.2之前存在，size_average已被reduction参数所替代，意思就是返回所有样本的总损失
    # criterion = nn.MSELoss(reduction='sum').to(device)
    criterion1 = SSIM_loss().to(device)

    # 随机梯度下降 momentum用于加速下降并减少震荡  动量类似与惯性 表示用了95%的过去梯度信息来加速当前更新
    # 原来使用SGD with momentum
    # optimizer = torch.optim.SGD(MAN.parameters(), lr=1e-4, momentum=0.95)
    optimizer = torch.optim.Adam(MAN.parameters(), lr=1e-4, weight_decay=1e-4)  # 增加权重衰减，防止过拟合  1e-4公共数据集的验证集不算很好，有点过拟合
    # 动态调整学习率
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6) #减少T_max，原来100，改为20
    img_root = '/media/admin1/hdd1/Projects/crowdcount-mcnn-master/data/formatted_trainval/shanghaitech_part_A_patches_9/train'
    # img_root = 'data/train_data/images'
    gt_dmap_root = '/media/admin1/hdd1/Projects/crowdcount-mcnn-master/data/formatted_trainval/shanghaitech_part_A_patches_9/train_den'
    # gt_dmap_root = 'data/train_data/ground_truth'
    dataset = CrowdDataset(img_root, gt_dmap_root, 4)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    val_img_root = '/media/admin1/hdd1/Projects/crowdcount-mcnn-master/data/formatted_trainval/shanghaitech_part_A_patches_9/val'
    # val_img_root = 'data/val_data/images'
    val_gt_dmap_root = '/media/admin1/hdd1/Projects/crowdcount-mcnn-master/data/formatted_trainval/shanghaitech_part_A_patches_9/val_den'
    # val_gt_dmap_root = 'data/val_data/ground_truth'
    val_dataset = CrowdDataset(val_img_root, val_gt_dmap_root, 4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)
    if not os.path.exists('./checkpoints_SH'):
        os.mkdir('./checkpoints_SH')

    min_mae = 4000
    min_epoch = 0
    train_loss_list = []
    train_loss_txt = "train_loss.txt"
    f_train = open(os.path.join("gt_dmap/", train_loss_txt), 'w')
    epoch_list = []
    val_loss_list = []
    val_error_list = []
    val_loss_txt = "val_loss.txt"
    f_val = open(os.path.join("gt_dmap/", val_loss_txt), 'w')
    # checkpoint = torch.load('./checkpoints/epoch_4999.param')
    # MAN.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # start_epoch = checkpoint['epoch']
    start_epoch=0

    for epoch in range(start_epoch, 2000):

        MAN.train()
        epoch_loss = 0
        epoch_val_loss = 0
        for i, (img, gt_dmap) in enumerate(dataloader):
            img = img.cuda()
            gt_dmap = gt_dmap.cuda()
            # forward propagation
            et_dmap = MAN(img)
            # calculate loss
            loss = criterion(et_dmap, gt_dmap) + criterion1(gt_dmap,et_dmap)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        print("epoch:", epoch, "loss:", epoch_loss / len(dataloader))
        epoch_list.append(epoch)
        train_loss_list.append(epoch_loss / len(dataloader))

        state = {'model': MAN.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, './checkpoints_SH/epoch_' + str(epoch) + ".param")

        MAN.eval()
        mae = 0
        for i, (img, gt_dmap) in enumerate(val_dataloader):
            img = img.cuda()
            gt_dmap = gt_dmap.cuda()
            # forward propagation
            et_dmap = MAN(img)
            val_loss = criterion(et_dmap, gt_dmap) + criterion1(gt_dmap,et_dmap)
            epoch_val_loss += val_loss.item()
            mae += abs(et_dmap.data.sum() - gt_dmap.data.sum()).item()
            del img, gt_dmap, et_dmap
        if mae / len(val_dataloader) < min_mae:
            min_mae = mae / len(val_dataloader)
            min_epoch = epoch
        val_error_list.append(mae / len(val_dataloader))
        val_loss_list.append(epoch_val_loss / len(val_dataloader))

        # print("epoch:", epoch, "val_loss:", epoch_val_loss/len(val_dataloader))
        print("epoch:" + str(epoch) + "error:" + str(mae / len(val_dataloader)) + "min_mae:" + str(min_mae)
              + "min_epoch:" + str(min_epoch))

        ##vis.line(win=1, X=epoch_list, Y=train_loss_list, opts=dict(title='train_loss'))
        # vis.line(win=2, X=epoch_list, Y=val_loss_list, opts=dict(title='val_loss'))
        ##vis.line(win=3, X=epoch_list, Y=val_error_list, opts=dict(title='val_error'))
        # show an image
        ##index = random.randint(0, len(val_dataloader) - 1)
        ##img, gt_dmap = val_dataset[index]
        ##vis.image(win=4, img=img, opts=dict(title='img'))
        ##vis.image(win=5, img=gt_dmap / (gt_dmap.max()) * 255, opts=dict(title='gt_dmap(' + str(gt_dmap.sum()) + ')'))
        ##img = img.unsqueeze(0).to(device)
        ##gt_dmap = gt_dmap.unsqueeze(0)
        ##et_dmap = res_mcnn(img)
        ##et_dmap = et_dmap.squeeze(0).detach().cpu().numpy()
        ##vis.image(win=6, img=et_dmap / (et_dmap.max()) * 255, opts=dict(title='et_dmap(' + str(et_dmap.sum()) + ')'))
    f_train.write(str(train_loss_list) + '\n')
    f_val.write(str(val_loss_list) + '\n')
    f_train.close()
    f_val.close()
    import time

    print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
