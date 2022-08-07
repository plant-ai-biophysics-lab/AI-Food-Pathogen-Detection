#%% 
#============================================== Define local file path ==============================================#

## A path to input images
root_path = '/data2/ai-food-pathogen-data/'

## A path to saved results
results_path = '/data/jyyi/ai-food-pathogen-results/'

## Models trained on our bacterial datasets:
### Our model trained on E. coli monoculture
model_2_path = results_path+'best_models/Kaggle/best_model210831_tr2.pth'
### Our model trained on multiple species + fine-tuned on microbial mixture
model_33_path = results_path+'best_models/Kaggle/best_model211024_tr33.pth'

#====================================================================================================================# 

import torch
import cv2
import matplotlib.pyplot as plt
import random
import models
import os
from scipy import stats
import numpy as np
import pandas as pd
from dataset import MicrobeDataset, get_transform
from sklearn.metrics import mean_squared_error
from statsmodels.api import formula as smf

'''
Load test datasets (food and ag water)

'''
dataset_name = ['ccw_1', 'ccw_2-3',                 #0-1: coconut water with 10, 10^2-10^3 CFU/mL of E. coli 
                'spw_1', 'spw_2-3',                 #2-3: spinach wash water with 10, 10^2-10^3 CFU/mL of E. coli 
                'irw_0', 'irw_1', 'irw_2-3',        #4-6: irrigation water with 0, 10, 10^2-10^3 CFU/mL of E. coli 
                'irwEC_0', 'irwEC_1', 'irwEC_2-3'   #7-9: irrigation water + EC broth with 0, 10, 10^2-10^3 CFU/mL of E. coli 
                ]

## Initialize empty lists
imgpath_test = [[] for _ in range(len(dataset_name))] 
annotpath_test = [[] for _ in range(len(dataset_name))]
dataset_test = [[] for _ in range(len(dataset_name))]

## Load test datasets via MicrobeDataset
for i in range(len(dataset_name)): # for each sample
    imgpath_test[i] = root_path+'test/'+dataset_name[i]+'/'
    if dataset_name[i].split('_')[-1] == '0':
        annotpath_test[i] = None
    else:
        annotpath_test[i] = root_path+dataset_name[i]+'_annotations.xml'
    dataset_test[i] = MicrobeDataset(root=imgpath_test[i], annot_path=annotpath_test[i], 
                                     transforms=get_transform(False), n_anchors=100)
    
#%%
'''
Load model and define functions

'''
## Select a model - uncomment one of the following two lines
model_path, model_num = model_2_path, 'model2'
# model_path, model_num = model_33_path, 'model33'

## Load trained model for inference
device = torch.device('cuda')
model_loader = models.ModelLoader()
model = model_loader.get_model(model_name='faster_rcnn', num_classes=2,
                               pretrained_backbone=True, # returns a model with backbone pre-trained on ImageNet
                               )
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

def get_pred(img, height=1024, width=1344):
    """Gets predicted bounding boxes (bboxes)

    Args:
        img (Tensor): A resized input image loaded via 'MicrobeDataset' s.t. img.shape=[C, H, W].
        height (int, optional): Height of an original input image. Defaults to 1024.
        width (int, optional): Width of an original input image. Defaults to 1344.
        
    Returns:
        pred_bboxes (array): Predicted bbox coordinates for a resized input image.
        pred_bboxes_cnv (list): Predicted bbox coordinates for an original input image.
    """
    ## Get predicted bboxes
    img = img.to(device)
    pred = model([img])
    pred_bboxes = pred[0]['boxes'].detach().cpu().numpy()
    
    ## Convert bbox coordinates to match the original image dimensions
    max_px = min(height, width)
    for x in range(len(pred_bboxes)):
        pred_bboxes[x][0] = pred_bboxes[x][0]*width/max_px
        pred_bboxes[x][1] = pred_bboxes[x][1]*height/max_px
        pred_bboxes[x][2] = pred_bboxes[x][2]*width/max_px
        pred_bboxes[x][3] = pred_bboxes[x][3]*height/max_px
    pred_bboxes_cnv = [[(j[0], j[1]), (j[2], j[3])] for j in list(pred_bboxes)]
    
    return pred_bboxes, pred_bboxes_cnv

def object_detection_api(imgpath, imgfile, dataset, savepath, rect_th=5):
    """Draws predicted bounding boxes on an original input image & saves to 'savepath'

    Args:
        imgpath (string): A root folder containing original input images
        imgfile (string): An original input image. File names should be in the format of 'Acquired-00.jpg'
        dataset (datset.MicrobeDataset): A dataset loaded via MicrobeDataset.
        savepath (string): A destination folder to save output images.
        rect_th (int, optional): Thickness of bboxes. Defaults to 5.        
    """
    img_orig = cv2.imread(imgpath+imgfile)
    height, width = img_orig.shape[0:2]
    img_filename = imgfile.split('.')[0]
    img_num = int(img_filename.split('-')[-1])
    for i in range(len(dataset)): # for each image
        
        ## Find a dataset that matches img_orig from dataset
        if img_num == dataset[i][3]:
            img = dataset[i][0]
            pred_bboxes_cnv = get_pred(img, height, width)[1]
            for k in range(len(pred_bboxes_cnv)):
                cv2.rectangle(img_orig, pred_bboxes_cnv[k][0], pred_bboxes_cnv[k][1],
                              color=(random.randint(0,255), random.randint(0,255), random.randint(0,255)), 
                              thickness=rect_th)
            plt.imsave(savepath+imgfile, img_orig)

def object_detection_gt(imgpath, imgfile, dataset, savepath, rect_th=5):
    """Draws GT bounding boxes on an original input image & saves to 'savepath'

    Args:
        imgpath (string): A root folder containing original input images.
        imgfile (string): An original input image. File names should be in the format of 'Acquired-00.jpg'
        dataset (datset.MicrobeDataset): A dataset loaded via MicrobeDataset.
        savepath (string): A destination folder to save output images.
        rect_th (int, optional): Thickness of bboxes. Defaults to 5.        
    """
    img_orig = cv2.imread(imgpath+imgfile)
    height, width = img_orig.shape[0:2]
    max_px = min(height, width)
    img_filename = imgfile.split('.')[0]
    img_num = int(img_filename.split('-')[-1])
    for i in range(len(dataset)): # for each image
        
        ## Find a dataset that matches img_orig from dataset
        if img_num == dataset[i][3]:
            annotations = dataset[i][1]
            gt_bboxes = annotations['boxes'].detach().cpu().numpy()
            
            ## Convert bbox coordinates to match the original image dimensions
            for x in range(len(gt_bboxes)): 
                gt_bboxes[x][0] = gt_bboxes[x][0]*width/max_px
                gt_bboxes[x][1] = gt_bboxes[x][1]*height/max_px
                gt_bboxes[x][2] = gt_bboxes[x][2]*width/max_px
                gt_bboxes[x][3] = gt_bboxes[x][3]*height/max_px
            gt_bboxes_cnv = [[(j[0], j[1]), (j[2], j[3])] for j in list(gt_bboxes)]
            for k in range(len(gt_bboxes_cnv)):
                cv2.rectangle(img_orig, gt_bboxes_cnv[k][0], gt_bboxes_cnv[k][1],
                              color=(random.randint(0,255), random.randint(0,255), random.randint(0,255)),
                              thickness=rect_th)
            plt.imsave(savepath+imgfile, img_orig)

#%%
'''
Compute observed and predicted results & save as a csv file
- Run this cell to get datasets for plotting
- Make sure to uncomment 'box_score_thresh=0.4' in 'models.py' to run this cell

'''
## Initialize empty lists & dataframe
obs_counts = [[] for _ in range(len(dataset_name))] # observed instances (GT)
pred_counts = [[] for _ in range(len(dataset_name))] # predicted instances
zero_counts = [[] for _ in range(len(dataset_name))] # observed zero instances
tn = [[] for _ in range(len(dataset_name))] # true negative (zero count)
fp = [[] for _ in range(len(dataset_name))] # false positive (non-zero counts)
fn = [[] for _ in range(len(dataset_name))] # false negative
tp = [[] for _ in range(len(dataset_name))] # true positive
df_count_total = pd.DataFrame([])

## Get observed and predicted instances from the loaded model
for i in range(len(dataset_name)): # for each sample

    for k in range(len(dataset_test[i])): # for each dataset or img
        
        ## Observed instances (GT)
        annotations = dataset_test[i][k][1]
        gt_bbox = annotations['boxes'].numpy()
        obs_counts[i].append(gt_bbox.shape[0]) # number of gt_bboxes per image

        ## Predicted instances
        img = dataset_test[i][k][0]
        pred_bboxes = get_pred(img)[0]
        pred_count = 0 # reset per img
        for j in range(len(pred_bboxes)): # for each bbox or instance
            pred_bbox = pred_bboxes[j]
            pred_count += 1
        pred_counts[i].append(pred_count)
        
    ## Count the number of images that falls into each category
    tn_count, fp_count, fn_count, tp_count = 0, 0, 0, 0 # reset per sample
    for j in range(len(obs_counts[i])): # for each image 
        if obs_counts[i][j] == 0 and pred_counts[i][j] == 0: tn_count += 1 # TN: obs & pred are zeros
        if obs_counts[i][j] == 0 and pred_counts[i][j] != 0: fp_count += 1 # FP: pred is non-zero, but obs is zero
        if obs_counts[i][j] != 0 and pred_counts[i][j] == 0: fn_count += 1 # FN: pred is zero, but obs is non-zero
        if obs_counts[i][j] != 0 and pred_counts[i][j] != 0: tp_count += 1 # TP: obs & pred are non-zeros
    tn[i], fp[i], fn[i], tp[i] = tn_count, fp_count, fn_count, tp_count
    
    ## Count the number of images with observed zero intances
    zero_count = 0 # reset per sample
    for j in range(len(obs_counts[i])): # for each image
        if obs_counts[i][j] == 0:
            zero_count += 1
    zero_counts[i] = zero_count
    if zero_counts[i] == 0:
        print('There is no zero count for %s' % dataset_name[i])
    elif (len(obs_counts[i])-zero_counts[i]) == 0:
        print('There is no nonzero count for %s' % dataset_name[i])
    else:
        print(dataset_name[i]+' GT_zero = %i/%i' % (zero_counts[i], len(obs_counts[i])))
    
print('All done!')

## Add results to dataframe
for i in range(len(dataset_name)): # for each sample
    area = 84 * 64 # area captured by microscope
    df_count = pd.DataFrame({"sample": dataset_name[i].split('_')[0], 
                             "initial load": dataset_name[i].split('_')[-1],
                             "obs": obs_counts[i], "pred": pred_counts[i],                             
                            #  "obs_instance": obs_counts[i], "pred_instance": pred_counts[i],
                             "TN": tn[i], "FP": fp[i], "FN": fn[i], "TP": tp[i]
                             })
    # df_count["obs"] = df_count["obs_instance"] / area * 10000
    # df_count["pred"] = df_count["pred_instance"] / area * 10000
    df_count_total = pd.concat([df_count_total, df_count], axis=0)

## Write a dataframe as a delimited text file (.csv)
df_count_total.to_csv('./data_count_'+model_num+'.csv', index=False, header=True)

#%%
'''
Read data for instances plot / density plot (Figures 4/5/S1/S2 - B, C, E, F)

'''
## Select a model - uncomment one of the following two lines
model_path, model_num = model_2_path, 'model2'
# model_path, model_num = model_33_path, 'model33'

## Select test dataset - uncomment one of the following four lines
sample_name, sample_color = 'ccw', 'gray'
# sample_name, sample_color = 'spw', 'olive'
# sample_name, sample_color = 'irw', 'sienna'
sample_name, sample_color = 'irwEC', 'sienna'

## Read dataframe from .csv file
df_counts = pd.read_csv('./data_count_'+model_num+'.csv', sep=",")  
df_count = df_counts[df_counts["sample"] == sample_name]
obs = df_count["obs"]
pred = df_count["pred"]

## Fit linear model between predicted and observed values
plot_x = np.linspace(np.min(obs), np.max(obs), 100)
slope, intc, rvalue, pvalue, stderr = stats.linregress(obs, pred)
plot_y = plot_x * slope + intc

## Estimate prediction intervals
df = pd.DataFrame({"obs": np.arange(0, np.max(obs), step=0.01)})
lm = smf.ols('pred ~ obs', df_count)
results = lm.fit()
a = .05 # 95% prediction interval
predictions = results.get_prediction(df).summary_frame(a)

## Fit linear model to upper and lower prediction intervals
slope_l, intc_l, rvalue_l, pvalue_l, stderr_l = stats.linregress(df['obs'], predictions['obs_ci_lower'])
slope_u, intc_u, rvalue_u, pvalue_u, stderr_u = stats.linregress(df['obs'], predictions['obs_ci_upper'])

## Calculate the 95% prediction zone thresholds
xintc_l = (max(0,-intc_l/slope_l)) # x-intercept of lower 95% PI
yintc_u = (intc_u) # y-intercept of upper 95% PI

## Calculate accuracy = (tp+tn)/(tp+tn+fp+fn)
tn, fp, fn, tp = sum(df_count.TN.unique()), sum(df_count.FP.unique()), sum(df_count.FN.unique()), sum(df_count.TP.unique())
accr = (tp+tn)/(tp+tn+fp+fn)    

#%%
'''
Draw instances plot (Figures 4/5/S1/S2 - B, E)

'''
plt.clf()
plt.rcParams['axes.linewidth'] = 1

## Generate plot
fig, ax = plt.subplots(figsize=(7.87,3.54))
plt.scatter(obs, pred, s=30, color=sample_color, linewidth=1)
plt.plot(plot_x, plot_y, color='black', alpha=0.8, linestyle='dashed', label='Linear regression', linewidth=1)
plt.fill_between(df['obs'], predictions['obs_ci_lower'], predictions['obs_ci_upper'], 
                 color=sample_color, alpha=.3, label='95% prediction interval')
plt.axvline(x=xintc_l, color='darkcyan', linestyle='dashed', 
            label='95% '+'prediction threshold \nfor a zero count', linewidth=1)
plt.axhline(y=yintc_u, color='mediumvioletred', linestyle='dashed', 
            label='95% '+'prediction threshold \nfor true non-zero counts', linewidth=1)

## Add equation and evaluation metrics
rmse = mean_squared_error(obs, pred, squared=False)
text_lm = 'y = %0.2fx + %0.2f\n$R^2$ = %0.2f\nRMSE = %0.1f\n$Acc_{zero}$ = %0.2f' % (slope, intc, rvalue ** 2, rmse, accr)
plt.text(x=1.05, y=0, s=text_lm, ha='left', va='bottom', fontsize=12, transform=ax.transAxes)

## Plot formatting
plt.xlabel('Observed $\it{E. coli}$ counts', fontsize=12)
plt.ylabel('Predicted $\it{E. coli}$ counts', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
xrange = np.max(obs)-np.min(obs)
plt.xlim(left=-xrange*0.01, right=np.max(obs)+xrange*0.01)
yrange = np.max(pred)-np.min(pred)
plt.ylim(bottom=-yrange*0.01, top=max(np.max(pred)+yrange*0.01,max(predictions['obs_ci_upper'])))
plt.legend(fontsize=12, bbox_to_anchor=(1, 1))
if xintc_l != 0:
    plt.text(x=xintc_l, y=plt.gca().get_ylim()[1], s='obs ≤ %0.1f ' % xintc_l, 
             color='darkcyan', ha='right', va='top', fontsize=12, rotation='90')
plt.text(x=plt.gca().get_xlim()[1], y=yintc_u, s='pred > %0.1f ' % yintc_u,
         color='mediumvioletred', ha='right', va='bottom', fontsize=12)
plt.tight_layout()

## Save plot
save_path = results_path+'instances/'+sample_name+'/'
if not os.path.exists(save_path):
            os.makedirs(save_path)
plt.savefig(save_path+model_num+'.png')

#%%
'''
Draw density plot with 95% prediction threshold for true non-zero counts (Figures 4/5/S1/S2 - C, F)

'''
plt.clf()
plt.rcParams['axes.linewidth'] = 1

## Read data
df_counts = pd.read_csv('./data_count_'+model_num+'.csv', sep=",")
df_count = df_counts[df_counts["sample"] == sample_name]
df_test = df_count.pivot(columns="initial load", values="pred")
if min(df_test) == '0':
    del df_test["0"]

## Get area under the curve to the right of the 95% prediction threshold (A_nonzero)
dlines = df_test.plot.density().get_lines()
a_nonzero = []
linex = [[] for _ in range(len(dlines))]
liney = [[] for _ in range(len(dlines))]
for j in range(len(dlines)):
    xy = dlines[j].get_xydata()
    for i in range(len(xy)):
        if xy[i][0] >= yintc_u:
            linex[j].append(xy[i][0])
            liney[j].append(xy[i][1])
    a_nonzero.append(np.trapz(liney[j],linex[j]))
    
## Add density plot
ax = df_test.plot.density(figsize=(5.31,3.54), fontsize=12, linewidth=1)  
ymax = plt.gca().get_ylim()[1]
xmax = np.nanmax(df_test.values)+10
yrange = plt.gca().get_ylim()[1]-plt.gca().get_ylim()[0]
xrange = plt.gca().get_xlim()[1]-plt.gca().get_xlim()[0]

## Add 95% prediction threshold & A_nonzero
plt.axvline(x=yintc_u, color='k', linestyle='dashed', linewidth=1)

## Fill A_nonzero to the right of the 95% precition threshold
for j in range(len(dlines)):
    plt.fill_between(linex[j], 0, liney[j], alpha=.3)

## Plot formatting & add text
conf = (1 - a) * 100
# ax.annotate('pred ≥ %0.1f' % yintc_u,
#             xy=(yintc_u, ymax-yrange*0.3), xycoords='data', 
#             xytext=(yintc_u+xrange*0.05, ymax-yrange*0.5), textcoords='data', color='k', fontsize=12,
#             arrowprops=dict(arrowstyle="->", color='k', connectionstyle="arc3,rad=-0.3", linewidth=1),
#             )

plt.text(x=yintc_u+xrange*0.02, y=ymax-yrange*0.05,
         s='95%'+' prediction threshold \nfor true non-zero counts \n(pred > %0.1f)' % yintc_u,
         color='k', ha='left', va='top', fontsize=12)
        #  color='k', ha='right', va='top', fontsize=12) # for Figure 5C

plt.text(x=1.05, y=0.25, s='$A_{non-zero}$:', ha='left', va='top', color='k', fontsize=12, transform=ax.transAxes)
plt.text(x=1.05, y=0.25, s='\n%0.2f (10 CFU/mL)' % a_nonzero[0],
         ha='left', va='top', fontsize=12, color='tab:blue', transform=ax.transAxes)
plt.text(x=1.05, y=0.25, s='\n\n%0.2f ($10^{2-3}$ CFU/mL)' % a_nonzero[1], 
         ha='left', va='top', fontsize=12, color='tab:orange',transform=ax.transAxes)

plt.xlabel("Predicted $\it{E. coli}$ counts", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.xlim(left=-10, right=np.nanmax(df_test.values)+10)
plt.ylim(bottom=0)
plt.legend(['10 CFU/mL', '$10^{2-3}$ CFU/mL','95% '+'prediction threshold'], fontsize=11, bbox_to_anchor=(1, 1), loc='upper left')

## Save plot
save_path = results_path+'instances/'+sample_name+'/'
if not os.path.exists(save_path):
            os.makedirs(save_path)
plt.savefig(save_path+sample_name+'_density_TP.png')

#%%
'''
Draw predicted bboxes on original input images (Figures 4/5/S1/S2 - A, D)

'''
# for i in range(0,4): # uncomment this line for 'ccw', 'spw' test dataset with model 'tr2'
# for i in range(4,10): # uncomment this line for 'irw', 'irwEC' test dataset with model 'tr33'
for i in range(len(dataset_name)): # for each sample
    save_path = results_path+'bbox/test/final/'+dataset_name[i]+'/'
    if not os.path.exists(save_path):
            os.makedirs(save_path)
    for imgfile in list(sorted(os.listdir(imgpath_test[i]))): # for each image
        if imgfile != 'Delete':
            object_detection_api(imgpath=imgpath_test[i], imgfile=imgfile, dataset=dataset_test[i], savepath=save_path)
    print(dataset_name[i]+' done!')
print('All done!')

#%%
'''
Draw GT bboxes on image files (Figures 4/5/S1/S2 - A, D)

'''
for i in range(len(dataset_name)): # for each sample
    save_path = results_path+'bbox/gt'+dataset_name[i]+'/'
    if not os.path.exists(save_path):
            os.makedirs(save_path)
    for imgfile in list(sorted(os.listdir(imgpath_test[i]))): # for each image
        if imgfile != 'Delete':
            object_detection_gt(imgpath=imgpath_test[i], imgfile=imgfile, dataset=dataset_test[i], savepath=save_path)
    print(dataset_name[i]+' done!')
print('All done!')

#%%