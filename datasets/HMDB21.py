import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
import numpy as np
import pickle as pkl
import os
import decord
from decord import VideoReader
import json
import random

from util.box_ops import box_xyxy_to_cxcywh

novel25 = [['brush_hair', 'pick', 'catch', 'golf', 'shoot_bow'],
           ['shoot_ball', 'push', 'climb_stairs', 'run', 'pick'],
           ['shoot_ball', 'throw', 'sit', 'climb_stairs', 'shoot_bow'],
           ['pour', 'shoot_ball', 'push', 'shoot_gun', 'stand'],
           ['clap', 'sit', 'wave', 'throw', 'pullup', 'catch']]

novel50 = [['brush_hair', 'pick', 'catch', 'golf', 'shoot_bow', 'throw', 'sit', 'walk', 'kick_ball', 'stand', 'pour'],
           ['shoot_ball', 'push', 'climb_stairs', 'run', 'pick', 'golf', 'clap', 'walk', 'throw', 'pullup', 'swing_baseball'],
           ['shoot_ball', 'throw', 'sit', 'climb_stairs', 'shoot_bow', 'pullup', 'walk', 'clap', 'kick_ball', 'swing_baseball', 'jump'],
           ['pour', 'shoot_ball', 'push', 'shoot_gun', 'stand', 'brush_hair', 'swing_baseball', 'clap', 'climb_stairs', 'shoot_bow', 'kick_ball'],
           ['clap', 'sit', 'wave', 'throw', 'pullup', 'catch', 'stand', 'climb_stairs', 'shoot_bow', 'swing_baseball', 'shoot_gun']]

decord.bridge.set_bridge('torch')

class HMDB21(Dataset):
    def __init__(self, video_dir, anno_pth,
                 transforms = None,
                 imgsize = (240,320),
                 frames = 8,
                 split = 'train',
                 base2novel = 25,
                 base2novelsplit = 0):
        assert base2novel in (25, 50), 'choose either 25 or 50 percent'
        self.video_dir = video_dir
        self.anno = pkl.load(open(anno_pth, 'rb'), encoding='latin1')
        self.gttubes = self.anno['gttubes']
        self.nframes = self.anno['nframes']
        
        self.transforms = transforms
        self.imgsize = imgsize
        self.frames = frames
        
        if split == 'train':
            self.video_list = self.anno['train_videos'][0]
            self.classes = self.anno['labels']
            self.classes = list(map(lambda x:x.replace('_', ' '), self.classes))
        elif split == 'test':
            self.video_list = self.anno['test_videos'][0]
            self.classes = self.anno['labels']
            self.classes = list(map(lambda x:x.replace('_', ' '), self.classes))
        elif split == 'base':
            if base2novel == 25:
                novel_classes = novel25[base2novelsplit]
            else:
                novel_classes = novel50[base2novelsplit]
                
            self.classes = list(set(self.anno['labels']) - set(novel_classes))
            self.classes = list(map(lambda x:x.replace('_', ' '), self.classes))
            
            self.video_list = self.anno['train_videos'][0] + self.anno['test_videos'][0]
            self.video_list = list(filter(lambda v:v.split('/')[0] not in novel_classes, self.video_list))
            
        elif split == 'novel':
            if base2novel == 25:
                novel_classes = novel25[base2novelsplit]
            else:
                novel_classes = novel50[base2novelsplit]
                    
            self.classes = novel_classes
            self.classes = list(map(lambda x:x.replace('_', ' '), self.classes))
            
            self.video_list = self.anno['test_videos'][0] #+ self.anno['train_videos'][0]
            self.video_list = list(filter(lambda v:v.split('/')[0] in novel_classes, self.video_list))
            
        self.bboxes = {}
        for idx in range(len(self.video_list)):
            # get bboxes per frame
            filepth = self.video_list[idx]
            gttubes = self.gttubes[self.video_list[idx]]
            for action in gttubes:
                gttube_set = gttubes[action]
                for tube in gttube_set:
                    for bbox in tube:
                        img_id, x1, y1, x2, y2 = bbox
                        img_id, x1, y1, x2, y2 = int(img_id), int(x1), int(y1), int(x2), int(y2)

                        file_dir = os.path.join(video_dir, filepth)
                        if file_dir not in self.bboxes:
                            self.bboxes[file_dir] = {}
                        
                        if img_id not in self.bboxes[file_dir]:
                            self.bboxes[file_dir][img_id] = [torch.tensor([x1,y1,x2,y2])]
                        else:
                            self.bboxes[file_dir][img_id].append(torch.tensor([x1,y1,x2,y2]))
            
        print(split, 'dataset (hmdb21) loaded!')
        
    def __len__(self):
        return len(self.video_list)
        

    def __getitem__(self, idx):
        v = self.video_list[idx]

        v = os.path.join(self.video_dir, v)
        vr = VideoReader(v + '.avi')
        nframe = len(vr)

        label = v.split('/')[-2].replace('_', ' ')

        start = 0
        end = nframe
        rate = nframe//self.frames
        clip_idx = list(range(start, end, rate)[:self.frames])
    
        # load clip
        clip = vr.get_batch(clip_idx).permute(0, 3, 1, 2) / 255 # TCHW, scaled between 0 and 255

        # load anno
        mididx = clip_idx[self.frames // 2] + 1
        try:
            bboxes = self.bboxes[v][mididx]
        except KeyError:
            bboxes = None
        
        if bboxes != None:
            bboxes = torch.stack(bboxes).float()
            bboxes[:, 0::2].clamp_(min=0, max=self.imgsize[1])
            bboxes[:, 1::2].clamp_(min=0, max=self.imgsize[0])
        else:
            bboxes = torch.empty((0, 4), dtype=torch.float32)

        bboxes = tv_tensors.BoundingBoxes(bboxes, format="XYXY", canvas_size=self.imgsize)

        if self.transforms != None:
            clip, bboxes = self.transforms(clip, bboxes)
            #clip = self.transforms(clip)

        canvas_size = bboxes.canvas_size
        # XYXY to CXCYWH
        bboxes = box_xyxy_to_cxcywh(bboxes)

        # raw to normal
        bboxes[:, 0::2] = bboxes[:,0::2] / canvas_size[1]
        bboxes[:, 1::2] = bboxes[:,1::2] / canvas_size[0]

        target = {'boxes': bboxes,
                  'text_labels': [[label]] * bboxes.shape[0],
                  'keyframe': mididx,
                  'video': v}
        
        return clip, target
        
def tubelet_in_tube(tube, i, K):
    # True if all frames from i to (i + K - 1) are inside tube
    # it's sufficient to just check the first and last frame.
    # return (i in tube[: ,0] and i + K - 1 in tube[:, 0])
    return all([j in tube[:, 0] for j in range(i, i + K)])


def tubelet_out_tube(tube, i, K):
    # True if all frames between i and (i + K - 1) are outside of tube
    return all([not j in tube[:, 0] for j in range(i, i + K)])


def tubelet_in_out_tubes(tube_list, i, K):
    # Given a list of tubes: tube_list, return True if
    # all frames from i to (i + K - 1) are either inside (tubelet_in_tube)
    # or outside (tubelet_out_tube) the tubes.
    return all([tubelet_in_tube(tube, i, K) or tubelet_out_tube(tube, i, K) for tube in tube_list])

def tubelet_has_gt(tube_list, i, K):
    # Given a list of tubes: tube_list, return True if
    # the tubelet starting spanning from [i to (i + K - 1)]
    # is inside (tubelet_in_tube) at least a tube in tube_list.
    return any([tubelet_in_tube(tube, i, K) for tube in tube_list])
    
'''
hmdb21textaug = json.load(open('gpt/GPT_HMDB21.json'))

def textaughmdb21(text_lst):
    aug_text_lst = []
    for text in text_lst:
        if text not in hmdb21textaug:
            aug_text_lst.append(text)
        else:
            aug_text_lst.append(random.choice(hmdb21textaug[text]))
    return aug_text_lst
'''
