import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

class ModelLoader:

    def __init__(self):
        None

    def get_model(self, model_name, num_classes=None, pretrained_backbone=False, pretrained_model_path=None):
        
        ## Load faster rcnn model from torchvision
        if model_name == 'faster_rcnn':

            anchor_sizes = ((5,), (10,), (20,), (25,), (35,))                                  
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, # returns a model pre-trained on COCOtrain2017
                                                                         box_detections_per_img=1000, 
                                                                         box_batch_size_per_image=2048, rpn_batch_size_per_image=1024, 
                                                                         rpn_anchor_generator=anchor_generator,
                                                                         box_score_thresh=0.4
                                                                         )
            print('Loaded pre-trained backbone.')

            ## Customize final output layer to correspond with number of specified classes
            input_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(input_features, num_classes)

            if pretrained_model_path is not None:
                model.load_state_dict(torch.load(pretrained_model_path))
                print('Loaded pre-trained model.')

            ## Determine if GPU available
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

            ## Move model to GPU/CPU
            model.to(device)
            if str(device)=='cuda':
                print("Model loaded on GPU")

            return model

        else:
            print('Could not find model with that name!')
            