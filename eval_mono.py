#%%
#============================================= Define local file path =============================================#

## A path to input images
root_path = '/data2/ai-food-pathogen-data/'

## A path to saved results
results_path = '/data/jyyi/ai-food-pathogen-results/'

## Models trained on our bacterial datasets:
### Our model trained on E. coli monoculture (split dataset)
model_2s_path = results_path+'best_models/Kaggle/best_model211225_tr2s.pth'

#==================================================================================================================# 

import os
import torch
import matplotlib.pyplot as plt
import models
from scipy import stats
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
import random
from dataset import MicrobeDataset, get_transform
from scipy.interpolate import make_interp_spline, BSpline
from sklearn.metrics import mean_squared_error
from statsmodels.api import formula as smf

'''
Load test datasets (E. coli monoculture)

'''
dataset_name = ['mono_1', 'mono_2', 'mono_3'] # E. coli monoculture (10, 10^2, 10^3 CFU/mL)

## Initialize empty lists for data loading
imgpath_test = [[] for _ in range(len(dataset_name))] 
annotpath_test = [[] for _ in range(len(dataset_name))]
dataset_test = [[] for _ in range(len(dataset_name))]

## Load test datasets via MicrobeDataset
for i in range(len(dataset_name)):
    imgpath_test[i] = root_path+'test/'+dataset_name[i]+'/'
    if 'mono_2' in dataset_name[i] or 'mono_3' in dataset_name[i]:
        annotpath_test[i] = root_path+'mono_2-3_annotations.xml'
    else:
        annotpath_test[i] = root_path+dataset_name[i]+'_annotations.xml'
    dataset_test[i] = MicrobeDataset(root=imgpath_test[i], annot_path=annotpath_test[i], 
                                     transforms=get_transform(False), n_anchors=100)

## Initialize empty lists for plotting
num_thresh = 11 # the number of pred_score thresholds to be explored
thresholds = [p/10 for p in range(num_thresh)]
obs_ls = [[] for _ in range(len(dataset_name))] # GT
pred_ls = [[[] for _ in range(num_thresh)] for _ in range(len(dataset_name))]

#%%
'''
Load model and functions

'''
## Load trained model for inference
model_path, model_num = model_2s_path, 'model2s'
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
        img (Tensor): A resized input image loaded via MicrobeDataset s.t. img.shape=[C, H, W].
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
    pred_scores = pred[0]['scores'].detach().cpu().numpy()
    
    ## Convert bbox coordinates to match the original image dimensions
    max_px = min(height, width)
    for x in range(len(pred_bboxes)):
        pred_bboxes[x][0] = pred_bboxes[x][0]*width/max_px
        pred_bboxes[x][1] = pred_bboxes[x][1]*height/max_px
        pred_bboxes[x][2] = pred_bboxes[x][2]*width/max_px
        pred_bboxes[x][3] = pred_bboxes[x][3]*height/max_px
    pred_bboxes_cnv = [[(j[0], j[1]), (j[2], j[3])] for j in list(pred_bboxes)]
    
    return pred_bboxes, pred_bboxes_cnv, pred_scores

def object_detection_api(imgpath, imgfile, dataset, savepath, rect_th=5):
    """Draws predicted bounding boxes on an original input image & saves to 'savepath'

    Args:
        imgpath (string): A root folder containing original input images
        imgfile (string): An original input image. File names should be 'Acquired-00.jpg'
        dataset (datset.MicrobeDataset): A dataset loaded via MicrobeDataset.
        savepath (string): A destination folder to save output images.
        rect_th (int, optional): Thickness of bboxes. Defaults to 5.        
    """
    img_orig = cv2.imread(imgpath+imgfile)
    height, width = img_orig.shape[0:2]
    img_filename = imgfile.split('.')[0]
    if '-' in img_filename:
        img_num = int(img_filename.split('-')[-1])
    if '_' in img_filename:
        img_num = int(img_filename.split('_')[-1])
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
        imgfile (string): An original input image. File names should be 'Acquired-00.jpg'
        dataset (datset.MicrobeDataset): A dataset loaded via MicrobeDataset.
        savepath (string): A destination folder to save output images.
        rect_th (int, optional): Thickness of bboxes. Defaults to 5.        
    """
    img_orig = cv2.imread(imgpath+imgfile)
    height, width = img_orig.shape[0:2]
    max_px = min(height, width)
    img_filename = imgfile.split('.')[0]
    if '-' in img_filename:
        img_num = int(img_filename.split('-')[-1])
    if '_' in img_filename:
        img_num = int(img_filename.split('_')[-1])
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
Compute GT and predicted results & save as a csv file
- Run this cell to get datasets for plotting
- Make sure to comment 'box_score_thresh=0.4' in 'models.py' to run this part

'''
## Get GT and predicted results
for i in range(len(dataset_name)): # for each sample
    
    for k in range(len(dataset_test[i])): # for each dataset or img
        
        ## Observed instances (GT)
        annotations = dataset_test[i][k][1]
        gt_bbox = annotations['boxes'].numpy()
        obs_ls[i].append(gt_bbox.shape[0]) # number of gt_bbox per image

        ## Predicted results
        im = dataset_test[i][k][0]
        pred_bboxes = get_pred(im)[0]
        pred_scores = get_pred(im)[2]
        for p in range(len(thresholds)): # for each pred_score threshold value
            pred_counts = 0 # reset
            for j in range(pred_bboxes.shape[0]): # for each bbox or instance
                if pred_scores[j] > thresholds[p]:
                    pred_counts += 1
            pred_ls[i][p].append(pred_counts)  
              
    print(dataset_name[i]+' done!')
    
print('All done!')

## Initialize an empty dataframe
df_count_total = pd.DataFrame([])

## Create dataframe
for i in range(len(dataset_name)): # for each sample
    area = 84 * 64 # area captured by microscope
    for p in range(len(thresholds)): 
        df_count = pd.DataFrame({"sample": dataset_name[i].split('_')[0],
                                "threshold": p/10,
                                "initial load": dataset_name[i].split('_')[1],
                                "obs": obs_ls[i], "pred": pred_ls[i][p]                                
                                # "obs_instance": obs_ls[i], "pred_instance": pred_ls[i][p]
                                })
        df_count_total = pd.concat([df_count_total, df_count], axis=0)

## Write dataframe as a delimited text file (.csv)
df_count_total.to_csv('./data_count_mono.csv', index=False, header=True)

#%%
'''
Plot pred score vs linregress eval metrics (Figure 3B)

'''
plt.clf()
plt.rcParams['axes.linewidth'] = 1

## Read dataframe from .csv file
df_count = pd.read_csv('./data_count_mono.csv', sep=",")

## Calculate RMSE, slope, R^2 for each subset as a functino of threshold
rmses = [[] for _ in range(num_thresh)]
slopes = [[] for _ in range(num_thresh)]
r_sqs = [[] for _ in range(num_thresh)]

for p in range(len(thresholds)):
    df_count_p = df_count[df_count["threshold"] == p/10]
    obs_p = df_count_p["obs"]
    pred_p = df_count_p["pred"]
    rmses[p] = mean_squared_error(obs_p, pred_p, squared=False)
    slope, intc, rvalue, pvalue, stderr = stats.linregress(obs_p, pred_p)
    slopes[p] = slope
    r_sqs[p] = rvalue ** 2

## Plot slope and R^2
fig, ax1 = plt.subplots(figsize=(5.31,3.54))
ax1.scatter(thresholds, slopes, s=30, facecolors='none', edgecolors='tab:orange', linewidth=1)
ax1.scatter(thresholds, r_sqs, s=30, facecolors='none', edgecolors='tab:green', linewidth=1)
plot_x = np.linspace(np.min(thresholds),np.max(thresholds),100)
spl_slopes = make_interp_spline(thresholds, slopes, k=1)  # type: BSpline
spl_rsq = make_interp_spline(thresholds, r_sqs, k=1)  # type: BSpline
ln1 = ax1.plot(plot_x, spl_slopes(plot_x), color='tab:orange', linestyle='dashed', label='Linear regression slope', linewidth=1)
ln2 = ax1.plot(plot_x, spl_rsq(plot_x), color='tab:green', linestyle='dotted', label='Linear regression $R^2$', linewidth=1)

## Plot RMSE
ax2 = ax1.twinx()
ax2.scatter(thresholds, rmses, s=30, facecolors='none', edgecolors='tab:blue', linewidth=1)
spl_rmse = make_interp_spline(thresholds, rmses, k=1)  # type: BSpline
ln3 = ax2.plot(plot_x, spl_rmse(plot_x), color='tab:blue', linestyle='dashdot', label='RMSE', linewidth=1)

## Plot formatting
ax1.set_xlabel('Confidence score threshold', fontsize=12)
ax1.set_ylabel('RMSE', fontsize=12)
ax2.set_ylabel('Linear regression slope or $R^2$', fontsize=12)
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)
ax2.tick_params(axis='y', labelsize=12)
lns = ln1+ln2+ln3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='best', fontsize=11)
plt.tight_layout()

## Save plot
save_path = results_path+'instances/mono/'
if not os.path.exists(save_path):
            os.makedirs(save_path)
plt.savefig(save_path+model_num+'_linregress.png')

# %%
'''
Draw instance plot (Figure 3A)

'''
## Read dataframe from .csv file
df_count = pd.read_csv('./data_count_mono.csv', sep=",")

for p in range(4,5):
# for p in range(len(thresholds)-1):
    plt.clf()
    plt.rcParams['axes.linewidth'] = 1
    df_count_p = df_count[df_count["threshold"] == p/10]
    obs_p = df_count_p["obs"]
    pred_p = df_count_p["pred"]

    ## Divide dataframe into subsets based on the initial load -> for figure legends
    df_count_1 = df_count_p[df_count_p["initial load"] == 1]
    obs_1 = df_count_1["obs"]
    pred_1 = df_count_1["pred"]
    df_count_2 = df_count_p[df_count_p["initial load"] == 2]
    obs_2 = df_count_2["obs"]
    pred_2 = df_count_2["pred"]
    df_count_3 = df_count_p[df_count_p["initial load"] == 3]
    obs_3 = df_count_3["obs"]
    pred_3 = df_count_3["pred"]
    
    ## Fit linear model between predicted and observed values
    plot_x = np.linspace(np.min(obs_p), np.max(obs_p), 100)
    slope, intc, rvalue, pvalue, stderr = stats.linregress(obs_p, pred_p)
    plot_y = plot_x * slope + intc

    ## Estimate prediction intervals
    df = pd.DataFrame({"obs": np.arange(0, np.max(obs_p), step=0.01)})
    lm = smf.ols('pred ~ obs', df_count_p)
    results = lm.fit()
    alpha = .05 # 95% prediction interval
    predictions = results.get_prediction(df).summary_frame(alpha)

    ## Fit linear model to upper and lower prediction intervals
    slope_l, intc_l, rvalue_l, pvalue_l, stderr_l = stats.linregress(df['obs'], predictions['obs_ci_lower'])
    slope_u, intc_u, rvalue_u, pvalue_u, stderr_u = stats.linregress(df['obs'], predictions['obs_ci_upper'])

    ## Calculate the confidence zone thresholds for plot
    xintc_l = (-intc_l/slope_l) # x-intercept of lower 95% PI
    yintc_u = (intc_u) # y-intercept of upper 95% PI
    
    ## Generate a plot
    fig, ax = plt.subplots(figsize=(7.08,3.54))
    plt.scatter(obs_1, pred_1, s=30, color='tab:blue', label='10 CFU/mL', linewidth=1)
    plt.scatter(obs_2, pred_2, s=30, color='tab:orange', label='$10^2$ CFU/mL', linewidth=1)
    plt.scatter(obs_3, pred_3, s=30, color='tab:green', label='$10^3$ CFU/mL', linewidth=1)
    plt.plot(plot_x, plot_y, color='black', linestyle='dashed', label='Linear regression', linewidth=1)
    plt.fill_between(df['obs'], predictions['obs_ci_lower'], predictions['obs_ci_upper'],
                     alpha=.3, label='95% prediction interval')
    plt.axvline(x=xintc_l, color='teal', linestyle='dashed', 
                label='95% '+'prediction threshold \nfor a zero count', linewidth=1)
    plt.axhline(y=yintc_u, color='mediumvioletred', linestyle='dashed',
                label='95% '+'prediction threshold \nfor true non-zero counts', linewidth=1)

    ## Add equation and error values
    rmse = mean_squared_error(obs_p, pred_p, squared=False)
    text_lm = 'y = %0.2fx + %0.2f\n$R^2$ = %0.2f; RMSE = %0.1f' % (slope, intc, rvalue ** 2, rmse)
    plt.text(x=1.05, y=0, s=text_lm, ha='left', va='bottom', fontsize=11, transform=ax.transAxes)

    ## Plot formatting
    plt.xlabel('Observed $\it{E. coli}$ counts', fontsize=12)
    plt.ylabel('Predicted $\it{E. coli}$ counts', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    xrange = np.max(obs_p)-np.min(obs_p)
    plt.xlim(left=-xrange*0.01, right=np.max(obs_p)+xrange*0.01)
    yrange = np.max(pred_p)-np.min(pred_p)
    plt.ylim(bottom=-yrange*0.01, top=max(np.max(pred_p)+yrange*0.01,max(predictions['obs_ci_upper'])))
    plt.text(x=xintc_l, y=plt.gca().get_ylim()[1], s='obs ≤ %0.1f ' % xintc_l, 
             color='teal', ha='right', va='top', fontsize=12, rotation='90')
    plt.text(x=plt.gca().get_xlim()[1], y=yintc_u, s='pred > %0.1f ' % yintc_u, 
             color='mediumvioletred', ha='right', va='bottom', fontsize=12)
    legend_order = [3,4,5,0,1,2,6]
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend([handles[idx] for idx in legend_order], [labels[idx] for idx in legend_order],
            fontsize=11, bbox_to_anchor=(1, 1))
    plt.tight_layout()

    ## Save plot
    save_path = results_path+'instances/mono/'
    if not os.path.exists(save_path):
                os.makedirs(save_path)
    plt.savefig(save_path+model_num+'thr'+str(p)+'.png')

#%%
'''
E. coli monoculture density plot (omitted)

'''
plt.clf()
plt.rcParams['axes.linewidth'] = 1

## Read data
df_count = pd.read_csv('./data_count_mono.csv', sep=",")
df_mono_p4 = df_count[df_count["threshold"] == 4/10]
df_mono = df_mono_p4.pivot(columns="initial load", values="pred")

## Estimate prediction intervals
obs_p4 = df_mono_p4["obs"]
pred_p4 = df_mono_p4["pred"]
df = pd.DataFrame({"obs": np.arange(0, np.max(obs_p4), step=0.01)})
lm4 = smf.ols('pred ~ obs', df_mono_p4)
results4 = lm4.fit()
alpha = .05 # 95% prediction interval
predictions4 = results4.get_prediction(df).summary_frame(alpha)

## Fit linear model to upper and lower prediction intervals
slope_l4, intc_l4, rvalue_l4, pvalue_l4, stderr_l4 = stats.linregress(df['obs'], predictions4['obs_ci_lower'])
slope_u4, intc_u4, rvalue_u4, pvalue_u4, stderr_u4 = stats.linregress(df['obs'], predictions4['obs_ci_upper'])

## Calculate the confidence zone thresholds for plot
xintc_l4 = (-intc_l4/slope_l4) # x-intercept of lower 95% PI
yintc_u4 = (intc_u4) # y-intercept of upper 95% PI

## Get area under the curve to the right of the 95% TP prediction threshold
dlines = df_mono.plot.density().get_lines()
auc = []
linex = [[] for _ in range(len(dlines))]
liney = [[] for _ in range(len(dlines))]
for j in range(len(dlines)):
    xy = dlines[j].get_xydata()
    for i in range(len(xy)):
        if xy[i][0] >= yintc_u4:
            linex[j].append(xy[i][0])
            liney[j].append(xy[i][1])
    auc.append(np.trapz(liney[j],linex[j]))
    
## Density plot
ax = df_mono.plot.density(figsize=(5.31,3.5), fontsize=11, linewidth=1)
ymax = plt.gca().get_ylim()[1]
yrange = plt.gca().get_ylim()[1]-plt.gca().get_ylim()[0]
xrange = plt.gca().get_xlim()[1]-plt.gca().get_xlim()[0]

## Add 95% TP prediction threshold and AUC
plt.axvline(x=yintc_u4, color='k', linestyle='dashed', linewidth=1)

## Fill the area under the curve to the right of the threshold
for j in range(len(dlines)):
    plt.fill_between(linex[j], 0, liney[j], alpha=.3)
    
## Plot formatting & add text
ax.annotate('\n95% '+'confidence \ntrue positive prediction (pred ≥ %0.1f)' % yintc_u4,
            xy=(yintc_u4, ymax-yrange*0.3), xycoords='data', 
            xytext=(yintc_u4+xrange*0.05, ymax-yrange*0.4), textcoords='data', color='k', fontsize=12,
            arrowprops=dict(arrowstyle="->", color='k', connectionstyle="arc3,rad=-0.3", linewidth=1),
            )
plt.text(x=yintc_u4+xrange*0.05, y=ymax-yrange*0.5, s='AUC:', ha='left', va='top', color='k', fontsize=12)
plt.text(x=yintc_u4+xrange*0.15, y=ymax-yrange*0.5, s='%0.2f (10 CFU/mL)' % auc[0],
         ha='left', va='top', fontsize=12, color='tab:blue')
plt.text(x=yintc_u4+xrange*0.15, y=ymax-yrange*0.5, s='\n%0.2f ($10^2$ CFU/mL)' % auc[1],
         ha='left', va='top', fontsize=12, color='tab:orange')
plt.text(x=yintc_u4+xrange*0.15, y=ymax-yrange*0.5, s='\n\n%0.2f ($10^3$ CFU/mL)' % auc[2],
         ha='left', va='top', fontsize=12, color='tab:green')
plt.xlabel("Predicted $\it{E. coli}$ counts", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.xlim(left=-1, right=max(pred_p4))
plt.ylim(bottom=0)
plt.legend(['10 CFU/mL', '$10^2$ CFU/mL', '$10^3$ CFU/mL'], fontsize=11, bbox_to_anchor=(1, 1))

## Save plot
save_path = results_path+'instances/mono/'
if not os.path.exists(save_path):
            os.makedirs(save_path)
plt.savefig(save_path+'_density_TP.png')

#%%
'''
Draw predicted bboxes on original input images (Figure 3C)

'''
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
Draw GT bboxes on image files (Figure 3C)

'''
for i in range(len(dataset_name)): # for each sample
    save_path = results_path+'bbox/gt/'+dataset_name[i]+'/'
    if not os.path.exists(save_path):
            os.makedirs(save_path)
    for imgfile in list(sorted(os.listdir(imgpath_test[i]))): # for each image
        if imgfile != 'Delete':
            object_detection_gt(imgpath=imgpath_test[i], imgfile=imgfile, dataset=dataset_test[i], savepath=save_path)
    print(dataset_name[i]+' done!')
print('All done!')

#%%