import math
# import sys
# import time
import torch
# import torchvision.models.detection.mask_rcnn
from torch.utils.tensorboard import SummaryWriter
# from coco_utils import get_coco_api_from_dataset
# from coco_eval import CocoEvaluator
import utils

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    data_loader.dataset.transform = True
    data_loader.dataset.mosaic = False
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    # if epoch == 0:
    #     warmup_factor = 1. / 1000
    #     warmup_iters = min(1000, len(data_loader) - 1)

        # lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    c=0
    writer = SummaryWriter(log_dir='/data/jyyi/ai-food-pathogen-results/runs')

    for (images, targets, image_id, img_num) in metric_logger.log_every(data_loader, print_freq, header):


        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        batch_size=len(images)
        writer.add_scalar("train loss", losses/batch_size, epoch)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        writer.flush()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            # with open("logs.txt", "a") as myfile:
            #     myfile.write('\n')
            #     myfile.write("Loss is {}, stopping training".format(loss_value))
            #     myfile.write('\n')
            #     myfile.write(loss_dict_reduced)
            # sys.exit(1)

        # losses.backward()

        # if c%3==0:
        #     optimizer.step()
        #     optimizer.zero_grad()
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        # if c%6==0:
        #     data_loader.dataset.mosaic = True
        # else:
        #     data_loader.dataset.mosaic = False
        # if data_loader.dataset.dataset.mosaic == True:
        #     data_loader.dataset.dataset.mosaic = False
        # if data_loader.dataset.dataset.mosaic == False:
        #     data_loader.dataset.dataset.mosaic = True
            
        # c+=1
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def validate(model, data_loader, device, best_val_loss):
    data_loader.dataset.transform = False
    data_loader.dataset.mosaic = False
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    total_val_loss=0
    for (images, targets, image_id, img_num) in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        total_val_loss=total_val_loss+losses

    if total_val_loss<best_val_loss:
        best_val_loss=total_val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print('Best Model Saved! ' + 'loss: ' +str(best_val_loss))
        with open("logs.txt", "a") as myfile:
            myfile.write('\n')
            myfile.write('Best Model Saved! ' + 'loss: ' +str(best_val_loss))
    else:
        print('Not Saved, Current Loss: '+ str(total_val_loss)+ ', Best Loss: '+ str(best_val_loss))
        with open("logs.txt", "a") as myfile:
            myfile.write('\n')
            myfile.write('Not Saved, Current Loss: '+ str(total_val_loss)+ ', Best Loss: '+ str(best_val_loss))

    return total_val_loss, best_val_loss

# def evaluate(model, data_loader, device):
#     data_loader.dataset.transform = False
#     data_loader.dataset.mosaic = False
#     n_threads = torch.get_num_threads()
#     # FIXME remove this and make paste_masks_in_image run on the GPU
#     torch.set_num_threads(1)
#     cpu_device = torch.device("cpu")
#     model.eval()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Test:'
#     coco = get_coco_api_from_dataset(data_loader.dataset)
#     #iou_types = _get_iou_types(model)
#     iou_types = ["bbox"]
#     coco_evaluator = CocoEvaluator(coco, iou_types)
#     for images, targets, image_id in metric_logger.log_every(data_loader, 10, header):
#         images = list(img.to(device) for img in images)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

#         torch.cuda.synchronize()
#         model_time = time.time()
#         outputs = model(images)

#         outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
#         model_time = time.time() - model_time

#         res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
#         evaluator_time = time.time()
#         coco_evaluator.update(res)
#         evaluator_time = time.time() - evaluator_time
#         metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
    
#     print("Averaged stats:", metric_logger)
#     with open("logs.txt", "a") as myfile:
#         myfile.write('\n')
#         myfile.write("Averaged stats:" + str(metric_logger))
#     coco_evaluator.synchronize_between_processes()

#     # accumulate predictions from all images
#     coco_evaluator.accumulate()
#     coco_evaluator.summarize()
#     torch.set_num_threads(n_threads)
#     return coco_evaluator