import csv
import os
import argparse
import datetime
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms

from sn8dataset import SN8Dataset
import models.pytorch_zoo.unet as unet
from models.other.unet import UNetSiamese
from models.other.siamunetdif import SiamUnet_diff
from models.other.siamnestedunet import SNUNet_ECAM

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv",
                        type=str,
                        required=True)
    parser.add_argument("--val_csv",
                         type=str,
                         required=True)
    parser.add_argument("--save_dir",
                         type=str,
                         required=True)
    parser.add_argument("--model_name",
                         type=str,
                         required=True)
    parser.add_argument("--lr",
                         type=float,
                        default=0.0001)
    parser.add_argument("--batch_size",
                         type=int,
                        default=2)
    parser.add_argument("--n_epochs",
                         type=int,
                         default=50)
    parser.add_argument("--gpu",
                        type=int,
                        default=0)
    
    args = parser.parse_args()
    return args

def write_metrics_epoch(epoch, fieldnames, train_metrics, val_metrics, training_log_csv):
    epoch_dict = {"epoch":epoch}
    merged_metrics = {**epoch_dict, **train_metrics, **val_metrics}
    with open(training_log_csv, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(merged_metrics)
        
def save_model_checkpoint(model, checkpoint_model_path): 
    torch.save(model.state_dict(), checkpoint_model_path)
        
def save_best_model(model, best_model_path):
    torch.save(model.state_dict(), best_model_path)

models = {
    'pseudo_resnet18_siamese': unet.Resnet18_pseudosiamese_upsample,
    'resnet18_siamese': unet.Resnet18_siamese_upsample,
    'resnet34_siamese': unet.Resnet34_siamese_upsample,
    'resnet34': unet.Resnet34_upsample,
    'resnet50': unet.Resnet50_upsample,
    'resnet101': unet.Resnet101_upsample,
    'seresnet50': unet.SeResnet50_upsample,
    'seresnet101': unet.SeResnet101_upsample,
    'seresnet152': unet.SeResnet152_upsample,
    'seresnext50': unet.SeResnext50_32x4d_upsample,
    'seresnext101': unet.SeResnext101_32x4d_upsample,
    'unet_siamese':UNetSiamese,
    'unet_siamese_dif':SiamUnet_diff,
    'nestedunet_siamese':SNUNet_ECAM
}

if __name__ ==  "__main__":
    # args = parse_args()
    # train_csv = args.train_csv
    # val_csv = args.val_csv
    # save_dir = args.save_dir
    # model_name = args.model_name
    # initial_lr = args.lr
    # batch_size = args.batch_size
    # n_epochs = args.n_epochs
    # gpu = args.gpu
    random_seeds = [4353, 7845, 1297, 6184, 2134, 8967, 1023, 5348, 7621, 4976]

    for seed in random_seeds:

        for typ in ['mix','lou']:

            train_csv = f"E:/code/Py_workplace/spacenet8/Data/train_val_csv/train_data_{typ}_seed_{seed}.csv"
            val_csv = f"E:/code/Py_workplace/spacenet8/Data/train_val_csv/val_data_{typ}_seed_{seed}.csv"
            save_dir = f"E:/code/Py_workplace/spacenet8/Data/model_output"
    
            model_name = "pseudo_resnet18_siamese"
            initial_lr = 0.0001
            batch_size = 2
            n_epochs = 20
            # gpu = -1
            gpu = 1


            
            now = datetime.now() 
            date_total = str(now.strftime("%d-%m-%Y-%H-%M"))

            # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)


            soft_dice_loss_weight = 0.25
            focal_loss_weight = 0.75
            num_classes=3
            # 0 background
            # 1 non-flooded road
            # 2 flooded road
            class_weights = None

            road_loss_weight = 0.5
            building_loss_weight = 0.5

            img_size = (512,512)

            if gpu>=0:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

            SEED=12
            torch.manual_seed(SEED)
            
            assert(os.path.exists(save_dir))
            save_dir = os.path.join(save_dir, f"{model_name}_{typ}_{seed}_lr{'{:.2e}'.format(initial_lr)}_bs{batch_size}_{date_total}")

            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
                os.chmod(save_dir, 0o777)
            checkpoint_model_path = os.path.join(save_dir, "model_checkpoint_modis.pth")
            best_model_path = os.path.join(save_dir, "best_model_modis.pth")
            training_log_csv = os.path.join(save_dir, "log.csv")

            # init the training log
            with open(training_log_csv, 'w', newline='') as csvfile:
                fieldnames = ['epoch', 'lr', 'train_tot_loss',
                                            'val_tot_loss']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

            train_dataset = SN8Dataset(train_csv,
                                    data_to_load=["preimg","pre_modis","post_modis","flood_road"],
                                    img_size=img_size)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, num_workers=2, batch_size=batch_size)
            val_dataset = SN8Dataset(val_csv,
                                    data_to_load=["preimg","pre_modis","post_modis","flood_road"],
                                    img_size=img_size)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, num_workers=2, batch_size=batch_size)

            #model = models["resnet34"](num_classes=5, num_channels=6)
            if model_name == "unet_siamese":
                model = UNetSiamese(3, num_classes, bilinear=True)
            elif model_name == "pseudo_resnet18_siamese":
                model = models[model_name](num_classes=num_classes, num_channels_1=3, num_channels_2=4)
            else:
                model = models[model_name](num_classes=num_classes, num_channels_1=3)


            if gpu>=0:
                model.cuda()

            optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
            
            if class_weights is None:
                celoss = nn.CrossEntropyLoss()
            else:
                celoss = nn.CrossEntropyLoss(weight=class_weights)

            best_loss = np.inf
            for epoch in range(n_epochs):
                print(f"EPOCH {epoch}")

                ### Training ##
                model.train()
                train_loss_val = 0
                train_focal_loss = 0
                train_soft_dice_loss = 0
                train_bce_loss = 0
                train_road_loss = 0
                train_building_loss = 0
                for i, data in enumerate(train_dataloader):
                    optimizer.zero_grad()

                    preimg, _, _, _, _, _, flood_road, _, pre_modis, post_modis = data
                    # "preimg","postimg","building","road","roadspeed","flood","flood_road","flood_building","pre_modis","post_modis"
                    flood = flood_road
                    # modis_img = np.concatenate((pre_modis, post_modis), axis=0)
                    # np.concatenate((image1, image1), axis=0)
                    combinedimg = torch.cat((pre_modis, post_modis), dim=1)
                    

                    if gpu>=0:
                        preimg = preimg.cuda().float()
                        modis_img = combinedimg.cuda().float()
                    else:
                        preimg = preimg.float()
                        modis_img = combinedimg.float()

                    flood = flood.numpy()
                    flood_shape = flood.shape
                    flood = np.append(np.zeros(shape=(flood_shape[0],1,flood_shape[2],flood_shape[3])), flood, axis=1)
                    flood = np.argmax(flood, axis = 1) # this is needed for cross-entropy loss. 

                    if gpu>=0:
                        flood = torch.tensor(flood).cuda()
                    else:
                        flood = torch.tensor(flood)

                    # flood_pred = model(combinedimg) # this is for resnet34 with stacked preimg+postimg input
                    flood_pred = model(preimg, modis_img) # this is for siamese resnet34 with stacked preimg+postimg input

                    #y_pred = F.sigmoid(flood_pred)
                    #focal_l = focal(y_pred, flood)
                    #dice_soft_l = soft_dice_loss(y_pred, flood)
                    #loss = (focal_loss_weight * focal_l + soft_dice_loss_weight * dice_soft_l)
                    loss = celoss(flood_pred, flood.long())

                    train_loss_val+=loss
                    #train_focal_loss += focal_l
                    #train_soft_dice_loss += dice_soft_l
                    loss.backward()
                    optimizer.step()

                    print(f"    {str(np.round(i/len(train_dataloader)*100,2))}%: TRAIN LOSS: {(train_loss_val*1.0/(i+1)).item()}", end="\r")
                print()
                train_tot_loss = (train_loss_val*1.0/len(train_dataloader)).item()
                #train_tot_focal = (train_focal_loss*1.0/len(train_dataloader)).item()
                #train_tot_dice = (train_soft_dice_loss*1.0/len(train_dataloader)).item()
                current_lr = scheduler.get_last_lr()[0]
                scheduler.step()
                train_metrics = {"lr":current_lr, "train_tot_loss":train_tot_loss}

                # validation
                model.eval()
                val_loss_val = 0
                val_focal_loss = 0
                val_soft_dice_loss = 0
                val_bce_loss = 0
                val_road_loss = 0
                with torch.no_grad():
                    for i, data in enumerate(val_dataloader):
                        preimg, _, _, _, _, _, flood_road, _, pre_modis, post_modis = data

                        #combinedimg = torch.cat((preimg, postimg), dim=1)
                        #combinedimg = combinedimg.cuda().float()
                        flood = flood_road
                        # modis_img = np.concatenate((pre_modis, post_modis), axis=0)
                        # # np.concatenate((image1, image1), axis=0)
                        # if gpu>=0:
                        #     preimg = preimg.cuda().float()
                        #     modis_img = modis_img.cuda().float()
                        # else:
                        #     preimg = preimg.float()
                        #     modis_img = modis_img.float()
                        
                        combinedimg = torch.cat((pre_modis, post_modis), dim=1)
                        

                        if gpu>=0:
                            preimg = preimg.cuda().float()
                            modis_img = combinedimg.cuda().float()
                        else:
                            preimg = preimg.float()
                            modis_img = combinedimg.float()


                        flood = flood.numpy()
                        flood_shape = flood.shape
                        flood = np.append(np.zeros(shape=(flood_shape[0],1,flood_shape[2],flood_shape[3])), flood, axis=1)
                        flood = np.argmax(flood, axis = 1) # for crossentropy
                        
                        #temp = np.zeros(shape=(flood_shape[0],6,flood_shape[2],flood_shape[3]))
                        #temp[:,:4] = flood
                        #temp[:,4] = np.max(flood[:,:2], axis=1)
                        #temp[:,5] = np.max(flood[:,2:], axis=1)
                        #flood = temp

                        if gpu>=0:
                            flood = torch.tensor(flood).cuda()
                        else:
                            flood = torch.tensor(flood)

                        # flood_pred = model(combinedimg) # this is for resnet34 with stacked preimg+postimg input
                        flood_pred = model(preimg, modis_img) # this is for siamese resnet34 with stacked preimg+postimg input


                        #y_pred = F.sigmoid(flood_pred)
                        #focal_l = focal(y_pred, flood)
                        #dice_soft_l = soft_dice_loss(y_pred, flood)
                        #loss = (focal_loss_weight * focal_l + soft_dice_loss_weight * dice_soft_l)

                        loss = celoss(flood_pred, flood.long())

                        #val_focal_loss += focal_l
                        #val_soft_dice_loss += dice_soft_l
                        val_loss_val += loss

                        print(f"    {str(np.round(i/len(val_dataloader)*100,2))}%: VAL LOSS: {(val_loss_val*1.0/(i+1)).item()}", end="\r")

                print()        
                val_tot_loss = (val_loss_val*1.0/len(val_dataloader)).item()
                #val_tot_focal = (val_focal_loss*1.0/len(val_dataloader)).item()
                #val_tot_dice = (val_soft_dice_loss*1.0/len(val_dataloader)).item()
                val_metrics = {"val_tot_loss":val_tot_loss}

                write_metrics_epoch(epoch, fieldnames, train_metrics, val_metrics, training_log_csv)

                save_model_checkpoint(model, checkpoint_model_path)

                epoch_val_loss = val_metrics["val_tot_loss"]
                if epoch_val_loss < best_loss:
                    print(f"    loss improved from {np.round(best_loss, 6)} to {np.round(epoch_val_loss, 6)}. saving best model...")
                    best_loss = epoch_val_loss
                    save_best_model(model, best_model_path)