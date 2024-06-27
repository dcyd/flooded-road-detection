import os
import time
import argparse
import glob

import json
from osgeo import gdal
from osgeo import osr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import torch
import torch.nn as nn
import pandas as pd

import models.pytorch_zoo.unet as unet
from sn8dataset import SN8Dataset
from models.other.unet import UNetSiamese
from models.other.siamunetdif import SiamUnet_diff
from models.other.siamnestedunet import SNUNet_ECAM
from utils.utils import write_geotiff

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                         type=str,
                         required=True)
    parser.add_argument("--model_name",
                         type=str,
                         required=True)
    parser.add_argument("--in_csv",
                       type=str,
                       required=True)
    parser.add_argument("--save_fig_dir",
                        help="saves model predictions as .pngs for easy viewing.",
                         type=str,
                         required=False,
                         default=None)
    parser.add_argument("--save_preds_dir",
                        help="saves model predictions as .tifs",
                         type=str,
                         required=False,
                         default=None)
    parser.add_argument("--gpu",
                         type=int,
                         required=False,
                         default=0)
    args = parser.parse_args()
    return args

def make_prediction_png(image, postimage, gt, prediction, save_figure_filename):
    #raw_im = image[:,:,:3]
    #raw_im = np.asarray(raw_im[:,:,::-1], dtype=np.float32)
    raw_im = np.moveaxis(image, 0, -1) # now it is channels last
    raw_im = raw_im/np.max(raw_im)
    post_im = np.moveaxis(postimage, 0, -1)
    post_im = post_im/np.max(post_im)
        
    #gt = np.asarray(gt*255., dtype=np.uint8)
    #pred = np.asarray(prediction*255., dtype=np.uint8)
    
    combined_mask_cmap = colors.ListedColormap(['black', 'red', 'blue', 'green', 'yellow'])

    grid = [[raw_im, gt, prediction],[post_im, 0, 0]]

    fig, axs = plt.subplots(2, 3, figsize=(12,8))
    for row in range(2):
        for col in range(3):
            ax = axs[row][col]
            ax.axis('off')
            if row==0 and col == 0:
                theim = ax.imshow(grid[row][col])
            elif row==1 and col == 0:
                if grid[row][col].ndim == 3 and grid[row][col].shape[2] == 3:
                    theim = ax.imshow(grid[row][col])
                else:
                    gray_image = grid[row][col][:, :, 0]
                    theim = ax.imshow(gray_image, cmap='gray')
            elif row==0 and col in [1,2]:
                ax.imshow(grid[row][col],
                          interpolation='nearest', origin='upper',
                          cmap=combined_mask_cmap,
                          norm=colors.BoundaryNorm([0, 1, 2, 3, 4, 5], combined_mask_cmap.N))
            elif row==1 and col == 1:
                ax.imshow(grid[0][0])
                mask = np.where(gt==0, np.nan, 1)
                ax.imshow(mask, cmap='gist_rainbow_r', alpha=0.6)
            elif row==1 and col == 2:
                ax.imshow(grid[0][0])
                mask = np.where(prediction==0, np.nan, 1)
                ax.imshow(mask, cmap='gist_rainbow_r', alpha=0.6)

    plt.subplots_adjust(hspace=0, wspace=0)
    plt.savefig(save_figure_filename, dpi=95)
    plt.close(fig)
    plt.close('all')
                
    
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

if __name__ == "__main__":
    # args = parse_args()
    # model_path = args.model_path
    # in_csv = args.in_csv
    # model_name = args.model_name
    # save_fig_dir = args.save_fig_dir
    # save_preds_dir = args.save_preds_dir
    # gpu = args.gpu
    random_seeds = [4353, 7845, 1297, 6184, 2134, 8967, 1023, 5348, 7621, 4976]
    ## for the mix dataset:
    # Splitting the data into training, validation, and test sets
    model_root_path = "E:\\code\\Py_workplace\\spacenet8\\Data\\model_output\\"
    in_csv = "E:\\code\\Py_workplace\\spacenet8\\Data\\train_val_csv\\modis_val.csv"
    model_name = "pseudo_resnet18_siamese"

    gpu = 0
    num_classes=3
    img_size = (512,512)

    perfor = dict()
    
    for typ in ['mix','lou']:
        perfor[typ] = {}
            
        for seed in random_seeds:
            perfor[typ][seed] = {}
            test_csv = f"E:/code/Py_workplace/spacenet8/Data/train_val_csv/test_data_{typ}_seed_{seed}.csv"
            save_dir = f"E:/code/Py_workplace/spacenet8/Data/model_output"
            save_fig_dir = f"E:\\code\\Py_workplace\\spacenet8\\Data\\eval_results\\png_file\\{model_name}_{typ}_{seed}"
            save_preds_dir = f"E:\\code\\Py_workplace\\spacenet8\\Data\\eval_results\\tif_file\\{model_name}_{typ}_{seed}"
 
            if not os.path.exists(save_preds_dir):
                os.mkdir(save_preds_dir)
                os.chmod(save_preds_dir, 0o777)

            if not os.path.exists(save_fig_dir):
                os.mkdir(save_fig_dir)
                os.chmod(save_fig_dir, 0o777)

            folder_prefix = f"{model_name}_{typ}_{seed}_lr"
            folders = glob.glob(os.path.join(model_root_path, f"{folder_prefix}*"))

            if folders:
                for folder in folders:
                    print(f"Found folder: {folder}")
            else:
                print("No matching folder found.")

            model_path = os.path.join(folders[0], "best_model_modis.pth")

            if gpu>=0:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

            test_dataset = SN8Dataset(test_csv,
                                    data_to_load=["preimg","pre_modis","post_modis","flood_road"],
                                    img_size=img_size)
            
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

            if model_name == "unet_siamese":
                model = UNetSiamese(3, num_classes, bilinear=True)
            elif model_name == "pseudo_resnet18_siamese":
                model = models[model_name](num_classes=num_classes, num_channels_1=3, num_channels_2=4)
            else:
                model = models[model_name](num_classes=num_classes, num_channels_1=3)

            if gpu>=0 and (torch.cuda.is_available()):
                model.cuda()
                
            model.load_state_dict(torch.load(model_path))


            #criterion = nn.BCEWithLogitsLoss()

            predictions = np.zeros((len(test_dataset),img_size[0],img_size[1]))
            gts = np.zeros((len(test_dataset),img_size[0],img_size[1]))

            # we need running numbers for each class: [no flood road, flood road]
            classes = ["non-flooded road", "flooded road"]
            running_tp = [0, 0]
            running_fp = [0, 0]
            running_fn = [0, 0]
            running_union = [0, 0]

            filenames = []
            precisions = [[],[]]
            recalls = [[],[]]
            f1s = [[],[]]
            ious = [[],[]]
            positives = [[],[]]

            model.eval()
            test_loss_test = 0
            with torch.no_grad():
                for i, data in enumerate(test_dataloader):
                    
                    current_image_filename = test_dataset.get_image_filename(i)
                    print("evaluating: ", i, os.path.basename(current_image_filename))
                    preimg, _, _, _, _, _, flood_road, _, pre_modis, post_modis = data

                    flood = flood_road
                    combinedimg = torch.cat((pre_modis, post_modis), dim=1)

                    if gpu>=0 and (torch.cuda.is_available()):
                        preimg = preimg.cuda().float()
                        modis_img = combinedimg.cuda().float()
                    else:
                        preimg = preimg.float()
                        modis_img = combinedimg.float()
                    
                    flood = flood.numpy()
                    flood_shape = flood.shape
                    flood = np.append(np.zeros(shape=(flood_shape[0],1,flood_shape[2],flood_shape[3])), flood, axis=1)
                    flood = np.argmax(flood, axis = 1)
                    
                    if gpu>=0 and (torch.cuda.is_available()):
                        flood = torch.tensor(flood).cuda()
                    else:
                        flood = torch.tensor(flood)

                    flood_pred = model(preimg, modis_img) # siamese resnet34 with stacked preimg+postimg input
                    flood_pred = torch.nn.functional.softmax(flood_pred, dim=1).cpu().numpy()[0] # (5, H, W)
                    #for i in flood_pred:
                    #    plt.imshow(i)
                    #    plt.colorbar()
                    #    plt.show()
                    
                    flood_prediction = np.argmax(flood_pred, axis=0) # (H, W)
                    #plt.imshow(flood_pred)
                    #plt.colorbar()
                    #plt.show()
                    
                    #flood_pred = torch.softmax(flood_pred)
                    #flood_pred = torch.sigmoid(flood_pred)
                    
                    #print(flood_pred.shape)
                    
                    ### save prediction
                    if save_preds_dir is not None:
                        ds = gdal.Open(current_image_filename)
                        geotran = ds.GetGeoTransform()
                        xmin, xres, rowrot, ymax, colrot, yres = geotran
                        raster_srs = osr.SpatialReference()
                        raster_srs.ImportFromWkt(ds.GetProjectionRef())
                        ds = None
                        output_tif = os.path.join(save_preds_dir, os.path.basename(current_image_filename.replace(".tif","_floodpred.tif")))

                        nrows, ncols = flood_prediction.shape
                        write_geotiff(output_tif, ncols, nrows,
                                    xmin, xres, ymax, yres,
                                    raster_srs, [flood_prediction])
                    
                    preimg = preimg.cpu().numpy()[0] # index at 0 so we have (C,H,W)
                    post_modis = post_modis.cpu().numpy()[0]
                    
                    gt_flood = flood.cpu().numpy()[0] # index so building gt is (H, W)
                    
                    #flood_prediction = flood_pred.cpu().numpy()[0] # index so shape is (C,H,W) for buildings
                    #flood_prediction = np.append(np.zeros(shape=(1,flood_shape[2],flood_shape[3])), flood_prediction, axis=0) # for focal loss 
                    #flood_prediction = np.argmax(flood_prediction, axis=0)
                    #flood_prediction = np.rint(flood_prediction).astype(int)

                    for j in range(2): # there are 4 classes
                        gt = np.where(gt_flood==(j+1), 1, 0) # +1 because classes start at 1. background is 0
                        prediction = np.where(flood_prediction==(j+1), 1, 0)
                        
                        #gts[i] = gt_flood
                        #predictions[i] = flood_prediction

                        tp = np.rint(prediction * gt)
                        fp = np.rint(prediction - tp)
                        fn = np.rint(gt - tp)
                        union = np.rint(np.sum(prediction + gt - tp))

                        iou = np.sum(tp) / np.sum((prediction + gt - tp + 0.00001))
                        tp = np.sum(tp).astype(int)
                        fp = np.sum(fp).astype(int)
                        fn = np.sum(fn).astype(int)

                        running_tp[j]+=tp
                        running_fp[j]+=fp
                        running_fn[j]+=fn
                        running_union[j]+=union

                        # acc = np.sum(np.where(prediction == gt, 1, 0)) / (gt.shape[0] * gt.shape[1])
                        precision = tp / (tp + fp + 0.00001)
                        recall = tp / (tp + fn + 0.00001)
                        f1 = 2 * (precision * recall) / (precision + recall + 0.00001)
                        precisions[j].append(precision)
                        recalls[j].append(recall)
                        f1s[j].append(f1)
                        ious[j].append(iou)

                        if np.sum(gt) < 1:
                            positives[j].append("n")
                        else:
                            positives[j].append("y")
                        
                    current_image_filename = test_dataset.files[i]["preimg"]
                    filenames.append(current_image_filename)
                    if save_fig_dir != None:
                        save_figure_filename = os.path.join(save_fig_dir, os.path.basename(current_image_filename)[:-4]+"_pred.png")
                        make_prediction_png(preimg, post_modis, gt_flood, flood_prediction, save_figure_filename)
            

            print()
            for j in range(len(classes)):
                print(f"class: {classes[j]}")
                precision = running_tp[j] / (running_tp[j] + running_fp[j] + 0.00001)
                recall = running_tp[j] / (running_tp[j] + running_fn[j] + 0.00001)
                f1 = 2 * (precision * recall) / (precision + recall + 0.00001)
                iou = running_tp[j] / (running_union[j] + 0.00001)
                print("  precision: ", precision)
                print("  recall: ", recall)
                print("  f1: ", f1)
                print("  iou: ", iou)
                perfor[typ][seed][f'class_{j}_iou'] = iou
                perfor[typ][seed][f'class_{j}_f1'] = f1
                perfor[typ][seed][f'class_{j}_precision'] = precision
                perfor[typ][seed][f'class_{j}_recall'] = recall
    
    # 转换为 DataFrame
    data = []

    # 遍历 perfor 字典
    for typ, seeds in perfor.items():
        for seed, metrics in seeds.items():
            # 将 typ 和 seed 添加到行数据中
            row = [typ, seed]
            # 将 metrics 字典中的值添加到行数据中
            row.extend([
                metrics['class_0_iou'], metrics['class_0_f1'], metrics['class_0_precision'], metrics['class_0_recall'],
                metrics['class_1_iou'], metrics['class_1_f1'], metrics['class_1_precision'], metrics['class_1_recall']
            ])
            # 将行数据添加到数据列表中
            data.append(row)

    # 定义列名
    columns = [
        'typ', 'seed',
        'class_0_iou', 'class_0_f1', 'class_0_precision', 'class_0_recall',
        'class_1_iou', 'class_1_f1', 'class_1_precision', 'class_1_recall'
    ]

    # 创建 DataFrame
    df = pd.DataFrame(data, columns=columns)

    # 将 DataFrame 保存为 Excel 文件
    excel_path = 'perfor_results_modis.xlsx'
    df.to_excel(excel_path, index=False)