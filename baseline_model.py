import torch
from torch import nn
import torchvision.transforms.functional as visionF
import torch.nn.functional as F
import numpy as np
from bytetrack.byte_tracker import BYTETracker
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from util.box_ops import box_cxcywh_to_xyxy

class Baseline(nn.Module):
    def __init__(self, yolos, vlm, fusion=False, fusion_ratio=0.5):
        super().__init__()
        self.yolos = yolos
        self.vlm = vlm
        
        self.human_id = 1
        self.thresh = 0.9
        self.fusion = fusion #disable/enable global-local fusion
        self.fusion_ratio = fusion_ratio
        
    def forward(self, x, captions):
        H, W = x.shape[-2:]
        text_embeds = F.normalize(self.vlm.encode_text(captions), dim=-1)
        
        results = []
        for B in range(x.shape[0]): # process each video independently
            # get global feature if fusion is enabled
            if self.fusion:
                global_embed = F.normalize(self.vlm.encode_vision(visionF.resize(x[B], (224,224))), dim=-1)
            # reinitialized tracker for each video
            tracker = BYTETracker(track_thresh=0.5,
                                  track_buffer=9,
                                  match_thresh=0.8,
                                  fuse_score=False,
                                  frame_rate=30)
            outputs = self.yolos(pixel_values=x[B])
            logits = outputs.logits
            bboxes = outputs.pred_boxes
            H, W = x[B].shape[-2:]
            
            
            # detect humans in all frames and build tubelets
            tubelets = {}
            keyframe_ids = []
            keyframe_boxes = []
            for t in range(x.shape[1]):
                class_id = 1
                
                logits = outputs.logits
                bboxes = outputs.pred_boxes
                
                probs = logits.softmax(-1)[t, :, :]
                selected_indices = (probs.argmax(-1) == class_id).nonzero(as_tuple=True)[0]
                
                filtered_boxes = bboxes[t, selected_indices]
                filtered_scores = probs[selected_indices, class_id]
                
                selected_indices = (filtered_scores > 0.8)
                filtered_boxes = filtered_boxes[selected_indices]
                filtered_scores = filtered_scores[selected_indices]
                
                filtered_boxes = box_cxcywh_to_xyxy(filtered_boxes)
                filtered_boxes = filtered_boxes * torch.tensor([W, H, W, H]).to(filtered_boxes.device)
                
                output_postprocessed = torch.cat([filtered_boxes,filtered_scores.unsqueeze(1)], dim=1).cpu().detach()
                online_targets = tracker.update(output_postprocessed, [H, W], (H, W))
                for tgt in online_targets:
                    tlwh = tgt.tlwh.clip(0)
                    tlwh[2:] += tlwh[:2]
                    tlwh[2].clip(W)
                    tlwh[3].clip(H)
                    tlwh = tlwh.astype(int)
                    tid = tgt.track_id
                    if tid not in tubelets:
                        tubelets[tid] = [(t,tlwh)]
                    else:
                        tubelets[tid].append((t,tlwh))

                    if t == x.shape[1] // 2:
                        keyframe_ids.append(tid)
                        keyframe_boxes.append(tlwh)
                        
            # classify tubelets with VLM (optionally + global-local feature fusion)
            tubelets_pred = {}
            tubelets_scores = {}
            for tube_id in tubelets:
                if tube_id in keyframe_ids:
                    boxes = tubelets[tube_id]
                    vid = []
                    for b in range(len(boxes)):
                        t, (x1,y1,x2,y2) = boxes[b]
                        vid.append(visionF.resize(visionF.crop(x[B,t], x1, y1, x2, y2), (224,224)))
                    vid = torch.stack(vid).unsqueeze(0)
                    vision_embeds = F.normalize(self.vlm.encode_vision(vid), dim=-1)
                    if self.fusion:
                        vision_embeds = (1 - self.fusion_ratio) * vision_embeds + self.fusion_ratio * global_embed
                    pred = (vision_embeds @ text_embeds.T)[0].argmax().cpu().detach().item()
                    #score = (vision_embeds @ text_embeds.T)[0].softmax(dim=-1).max().cpu().detach().item()
                    score = (((vision_embeds @ text_embeds.T)[0]+1)/2).max().cpu().detach().item()
                    tubelets_pred[tube_id] = pred
                    tubelets_scores[tube_id] = score

            keyframe_pred = [tubelets_pred[ele] for ele in keyframe_ids]
            keyframe_scores = [tubelets_scores[ele] for ele in keyframe_ids]
            
            results.append({'pred_classes':keyframe_pred, 'pred_boxes':keyframe_boxes, 'pred_scores': keyframe_scores})
        
        return results
