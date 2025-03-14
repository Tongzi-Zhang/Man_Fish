#%%
import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib.cm as CM
import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# from model.attention_mcnn_model15 import Attention_mcnn
from load_data import CrowdDataset
from model.model1 import MCNN_new

def cal_mae(img_root,gt_dmap_root,model_param_path):
    '''
    Calculate the MAE of the test data.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    '''
    device=torch.device("cuda")
    res_mcnn = MCNN_new().to(device)
    checkpoint = torch.load(model_param_path)
    res_mcnn.load_state_dict(checkpoint['model'])
    dataset=CrowdDataset(img_root,gt_dmap_root,4)
    print(dataset.img_names)
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    res_mcnn.eval()
    mae=0
    precision =  0
    gt_num_arr = 0
    et_num_arr = 0
    with torch.no_grad():
        for i,(img,gt_dmap) in enumerate(dataloader):
            img=img.to(device)
            gt_dmap=gt_dmap.to(device)
            gt_num = gt_dmap.data.sum().cpu()
            gt_num = gt_num.numpy()
            
            gt_dmap = gt_dmap.squeeze(0).squeeze(0).cpu().numpy()
            np.save('gt_dmap/gt_dmap'+str(i)+'.npy',gt_dmap)
            plt.imshow(gt_dmap, cmap=CM.jet)
            plt.savefig('gt_dmap/gt_dmap' + str(i) + '.png')
            # forward propagation
            et_dmap = res_mcnn(img).detach()
            et_num = et_dmap.data.sum().cpu()
            et_num = et_num.numpy()

            et_dmap = et_dmap.squeeze(0).squeeze(0).cpu().numpy()
            np.save('et_dmap/et_dmap'+str(i)+'.npy',et_dmap)
            plt.imshow(et_dmap, cmap=CM.jet)
            plt.savefig('et_dmap/et_dmap' + str(i) + '.png')
            # gt_num = gt_dmap.data.sum()
            # et_num = et_dmap.data.sum()
            mae+=abs(et_num -gt_num).item()
            precision += gt_num.item()

            a = np.array(gt_num)
            b = np.array(et_num)
            gt_num_arr = np.vstack((gt_num_arr, a))
            et_num_arr = np.vstack((et_num_arr, b))

            del img,gt_dmap,et_dmap
        np.save( 'gt_dmap/gt_num_arr.npy', gt_num_arr)
        np.save('et_dmap/et_num_arr.npy', et_num_arr)
        np.save('mae.npy', mae/len(dataloader))
        np.save('presicion.npy', 1-(mae)/(precision))


    print("model_param_path:"+model_param_path+" MAE:"+str(mae/len(dataloader)) + "presicion:" + str(1-(mae)/(precision)))


if __name__=="__main__":
    torch.backends.cudnn.enabled=False
    # img_root='data/test_data/images'
    # gt_dmap_root='data/test_data/ground_truth'
    img_root = '/media/admin1/hdd1/Projects/crowdcount-mcnn-master/data/formatted_trainval/shanghaitech_part_A_patches_9/images'
    gt_dmap_root = '/media/admin1/hdd1/Projects/crowdcount-mcnn-master/data/formatted_trainval/shanghaitech_part_A_patches_9/test_den'
    # model_param_path='checkpoints/epoch_8.param'
    model_param_path='checkpoints_SH/epoch_604.param'
    cal_mae(img_root,gt_dmap_root,model_param_path)