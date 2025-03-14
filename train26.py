import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
import torch
import torch.nn as nn
# import visdom
import random
import sys
Root_Dir = os.path.abspath("../../")
sys.path.append(Root_Dir)
import model
from SSIM1 import SSIM_loss
from model.attention_mcnn_model15 import Attention_mcnn
from load_data import CrowdDataset

if __name__ == "__main__":
    torch.backends.cudnn.enabled = False

    # vis = visdom.Visdom(env='MAN')
    # gpu_ids = [0,1]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MAN = Attention_mcnn().cuda()
    # if len(gpu_ids) > 1:
    #     MAN = torch.nn.DataParallel(MAN,device_ids=gpu_ids)

    MAN = MAN.to(device)
    criterion = nn.MSELoss(size_average=False).cuda()
    criterion1 = SSIM_loss().cuda(device)
    optimizer = torch.optim.SGD(MAN.parameters(), lr=1e-6, momentum=0.95)
    img_root = 'data1/train_data/images'
    gt_dmap_root = 'data1/train_data/ground_truth'
    dataset = CrowdDataset(img_root, gt_dmap_root, 4)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    val_img_root = 'data1/val_data/images'
    val_gt_dmap_root = 'data1/val_data/ground_truth'
    val_dataset = CrowdDataset(val_img_root, val_gt_dmap_root, 4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)

    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')

    min_mae = 4000
    min_epoch = 0
    train_loss_list = []
    train_loss_txt = "train_loss.txt"
    f_train = open(os.path.join("./", train_loss_txt), 'w')
    epoch_list = []
    val_loss_list = []
    val_error_list = []
    val_loss_txt = "val_loss.txt"
    f_val = open(os.path.join("./", val_loss_txt), 'w')
    checkpoint = torch.load('./checkpoints/epoch_4999.param')
    MAN.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']

    for epoch in range(start_epoch, 10000):

        MAN.train()
        epoch_loss = 0
        epoch_val_loss = 0
        for i, (img, gt_dmap) in enumerate(dataloader):
            img = img.cuda()
            gt_dmap = gt_dmap.cuda()
            # forward propagation
            et_dmap = MAN(img)
            # calculate loss
            loss = criterion(et_dmap, gt_dmap) +  criterion1( gt_dmap, et_dmap)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("epoch:", epoch, "loss:", epoch_loss/len(dataloader))
        epoch_list.append(epoch)
        train_loss_list.append(epoch_loss/len(dataloader))
        
        state = {'model':MAN.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
        torch.save(state, './checkpoints/epoch_' + str(epoch) + ".param")

        MAN.eval()
        mae= 0
        for i, (img, gt_dmap) in enumerate(val_dataloader):
            img = img.cuda()
            gt_dmap = gt_dmap.cuda()
            # forward propagation
            et_dmap = MAN(img)
            val_loss = criterion(et_dmap, gt_dmap) +  criterion1( gt_dmap, et_dmap)
            epoch_val_loss  += val_loss.item()
            mae += abs(et_dmap.data.sum() - gt_dmap.data.sum()).item()
            del img, gt_dmap, et_dmap
        if mae / len(val_dataloader) < min_mae:
            min_mae = mae / len(val_dataloader)
            min_epoch = epoch
        val_error_list.append(mae / len(val_dataloader))
        val_loss_list.append(epoch_val_loss/len(val_dataloader))
        
        #print("epoch:", epoch, "val_loss:", epoch_val_loss/len(val_dataloader))
        print("epoch:" + str(epoch) + "error:" + str(mae / len(val_dataloader)) + "min_mae:" +str(min_mae)
              + "min_epoch:"+ str(min_epoch))

        ##vis.line(win=1, X=epoch_list, Y=train_loss_list, opts=dict(title='train_loss'))
        #vis.line(win=2, X=epoch_list, Y=val_loss_list, opts=dict(title='val_loss'))
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
    f_train.write(str(train_loss_list)+ '\n')
    f_val.write(str(val_loss_list)+ '\n')
    f_train.close()
    f_val.close()
    import time

    print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
