import os
import torch
from torchvision.transforms import v2
from torchvision.ops import batched_nms
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import UCF24, HMDB21

from baseline_model import Baseline

import torch.nn.functional as F
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy
from util.misc import reduce_dict
from torchmetrics.detection import MeanAveragePrecision
import numpy as np
#import cv2 #disable for headless

from pprint import pprint

import argparse
parser = argparse.ArgumentParser(description="Evaluate f-mAP@0.5 for UCF-101-24 and JHMDB")
parser.add_argument("-ucf101", type=str,
                    help="path to ucf101")
parser.add_argument("-jhmdb", type=str,
                    help="path to jhmdb")
parser.add_argument("-fusion", type=float, default=0,
                    help="global-local vision embedding fusion ratio")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

################
# Load Dataset #
################
mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
unnormalize = v2.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

video_input_num_frames = 9

num_workers = 4
tfs = v2.Compose([v2.Resize((240, 320)), v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = torch.stack(batch[0])
    return tuple(batch)

argsUCF24 = args.ucf101
argsANNOUCF24 = 'anno/UCF101v2-GT.pkl'
argsRATEUCF24 = 7
test_ucf = UCF24(argsUCF24, argsANNOUCF24, transforms = tfs, frames=video_input_num_frames, rate=argsRATEUCF24, split = 'test')

argsHMDB21 = args.jhmdb
argsANNOHMDB21 = 'anno/JHMDB-GT.pkl'
test_hmdb = HMDB21(argsHMDB21, argsANNOHMDB21, transforms = tfs, frames = video_input_num_frames, split = 'test')

ucf_captions = test_ucf.classes
ucf_captionstoidx = {v: k for k, v in enumerate(ucf_captions)}
jhmdb_captions = test_hmdb.classes
jhmdb_captionstoidx = {v: k for k, v in enumerate(jhmdb_captions)}

#################
# Load Pipeline #
#################
from transformers import AutoImageProcessor, AutoModelForObjectDetection
image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")
from bytetrack.byte_tracker import BYTETracker
from viclip import get_viclip
import torchvision.transforms.functional as visionF
vlm = get_viclip('b', 'viclip/ViCLIP-B_InternVid-FLT-10M.pth')['viclip']
_ = vlm.eval()

if args.fusion > 0:
    base = Baseline(model, vlm, fusion=True, fusion_ratio=args.fusion)
    print('global-local fusion enabled with ratio:', args.fusion)
else:
    base = Baseline(model, vlm)

_ = base.to(device)

############
# Test UCF #
############
metric = MeanAveragePrecision(iou_type='bbox',
                              box_format='xyxy',
                              iou_thresholds=[0.5, ],
                              backend='faster_coco_eval')
for i in tqdm(range(len(test_ucf))):
    preds, tgts = [], []
    captions = ucf_captions

    sample, gt = test_ucf[i]
    H, W = sample.shape[-2:]

    # captionstoidx
    captionstoidx = {v: k for k, v in enumerate(captions)}
    temp_num_classes = len(captions)
    gt['boxes'] = gt['boxes'].to(device)
    intlabels = [list(map(lambda x:captionstoidx[x], ele)) for ele in gt['text_labels']]
    gt['labels'] = intlabels
    
    out = base(sample.unsqueeze(0).to(device), captions)[0]
    
    pred_boxes = out['pred_boxes']
    pred_classes = out['pred_classes']
    pred_scores = np.array(out['pred_scores'])
    if len(pred_boxes) == 0:
        result_dict = None
    else:
        cls = torch.tensor(pred_classes).to(device)
        scores = torch.tensor(pred_scores).to(device)
        boxes = torch.tensor(pred_boxes).to(device)
        result_dict = {'boxes': boxes, 'labels': cls, 'scores': scores}
    
    rawboxes = gt['boxes']
    rawboxes = box_cxcywh_to_xyxy(rawboxes)
    rawcls = gt['labels']
    boxes = []
    cls = []
    for j in range(len(rawcls)):
        tmpcls = rawcls[j]
        cls.extend(tmpcls)
        for _ in range(len(tmpcls)):
            boxes.append(rawboxes[j])
    boxes = torch.stack(boxes)
    boxes[:,0::2] = boxes[:,0::2] * W
    boxes[:,1::2] = boxes[:,1::2] * H
    cls = torch.tensor(cls).to(boxes.device)
    ground_truth = {'boxes': boxes, 'labels': cls}

    if result_dict != None:
        preds.append(result_dict)
        tgts.append(ground_truth)
    metric.update(preds, tgts)
        
pprint(metric.compute())

##############
# Test JHMDB #
##############
metric = MeanAveragePrecision(iou_type='bbox',
                              box_format='xyxy',
                              iou_thresholds=[0.5, ],
                              backend='faster_coco_eval')
for i in tqdm(range(len(test_hmdb))):
    preds, tgts = [], []
    captions = jhmdb_captions

    sample, gt = test_hmdb[i]
    H, W = sample.shape[-2:]

    # captionstoidx
    captionstoidx = {v: k for k, v in enumerate(captions)}
    temp_num_classes = len(captions)
    gt['boxes'] = gt['boxes'].to(device)
    intlabels = [list(map(lambda x:captionstoidx[x], ele)) for ele in gt['text_labels']]
    gt['labels'] = intlabels
    
    out = base(sample.unsqueeze(0).to(device), captions)[0]
    
    pred_boxes = out['pred_boxes']
    pred_classes = out['pred_classes']
    pred_scores = np.array(out['pred_scores'])
    if len(pred_boxes) == 0:
        result_dict = None
    else:
        cls = torch.tensor(pred_classes).to(device)
        scores = torch.tensor(pred_scores).to(device)
        boxes = torch.tensor(pred_boxes).to(device)
        result_dict = {'boxes': boxes, 'labels': cls, 'scores': scores}
    
    rawboxes = gt['boxes']
    rawboxes = box_cxcywh_to_xyxy(rawboxes)
    rawcls = gt['labels']
    boxes = []
    cls = []
    for j in range(len(rawcls)):
        tmpcls = rawcls[j]
        cls.extend(tmpcls)
        for _ in range(len(tmpcls)):
            boxes.append(rawboxes[j])
    boxes = torch.stack(boxes)
    boxes[:,0::2] = boxes[:,0::2] * W
    boxes[:,1::2] = boxes[:,1::2] * H
    cls = torch.tensor(cls).to(boxes.device)
    ground_truth = {'boxes': boxes, 'labels': cls}

    if result_dict != None:
        preds.append(result_dict)
        tgts.append(ground_truth)
    metric.update(preds, tgts)
        
pprint(metric.compute())
