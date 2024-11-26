import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
import numpy as np
import pickle as pkl
import os
import decord
from decord import VideoReader
import json
import regex as re
import random

from util.box_ops import box_xyxy_to_cxcywh

novel25 = [['TennisSwing', 'RopeClimbing', 'VolleyballSpiking', 'HorseRiding', 'LongJump', 'SkateBoarding'],
           ['VolleyballSpiking', 'GolfSwing', 'Diving', 'SalsaSpin', 'TrampolineJumping', 'WalkingWithDog'],
           ['HorseRiding', 'LongJump', 'SkateBoarding', 'CliffDiving', 'RopeClimbing', 'PoleVault'],
           ['Basketball', 'Biking', 'SalsaSpin', 'IceDancing', 'GolfSwing', 'HorseRiding'],
           ['FloorGymnastics', 'PoleVault', 'CliffDiving', 'TrampolineJumping', 'HorseRiding', 'SalsaSpin']]

novel50 = [['TennisSwing', 'RopeClimbing', 'VolleyballSpiking', 'HorseRiding', 'LongJump', 'SkateBoarding', 'PoleVault', 'SoccerJuggling', 'Diving', 'Skiing', 'CricketBowling', 'Skijet'],
           ['VolleyballSpiking', 'GolfSwing', 'Diving', 'SalsaSpin', 'TrampolineJumping', 'WalkingWithDog', 'FloorGymnastics', 'PoleVault', 'Skijet', 'Surfing', 'SkateBoarding', 'HorseRiding'],
           ['HorseRiding', 'LongJump', 'SkateBoarding', 'CliffDiving', 'RopeClimbing', 'PoleVault', 'VolleyballSpiking', 'Diving', 'TrampolineJumping', 'Biking', 'TennisSwing', 'SalsaSpin'],
           ['Basketball', 'Biking', 'SalsaSpin', 'IceDancing', 'GolfSwing', 'HorseRiding', 'Skijet', 'VolleyballSpiking', 'CliffDiving', 'Skiing', 'RopeClimbing', 'BasketballDunk'],
           ['FloorGymnastics', 'PoleVault', 'CliffDiving', 'TrampolineJumping', 'HorseRiding', 'SalsaSpin', 'Basketball', 'CricketBowling', 'BasketballDunk', 'Surfing', 'SkateBoarding', 'TennisSwing']]

decord.bridge.set_bridge('torch')

class UCF24(Dataset):
    def __init__(self, video_dir, anno_pth,
                 transforms = None,
                 imgsize = (240,320),
                 frames = 8,
                 rate = 8,
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
        self.rate = rate
        self.T = frames * rate
        
        if split == 'train':
            self.video_list = self.anno['train_videos'][0]
            self.classes = self.anno['labels']
            self.classes = list(map(lambda x:' '.join(re.findall('[A-Z][a-z]*', x)), self.classes))
        elif split == 'test':
            self.video_list = self.anno['test_videos'][0]
            self.classes = self.anno['labels']
            self.classes = list(map(lambda x:' '.join(re.findall('[A-Z][a-z]*', x)), self.classes))
        elif split == 'base':
            if base2novel == 25:
                novel_classes = novel25[base2novelsplit]
            else:
                novel_classes = novel50[base2novelsplit]
                
            self.classes = list(set(self.anno['labels']) - set(novel_classes))
            self.classes = list(map(lambda x:' '.join(re.findall('[A-Z][a-z]*', x)), self.classes))
            
            self.video_list = self.anno['train_videos'][0] + self.anno['test_videos'][0]
            self.video_list = list(filter(lambda v:v.split('/')[0] not in novel_classes, self.video_list))
            
        elif split == 'novel':
            if base2novel == 25:
                novel_classes = novel25[base2novelsplit]
            else:
                novel_classes = novel50[base2novelsplit]
                    
            self.classes = novel_classes
            self.classes = list(map(lambda x:' '.join(re.findall('[A-Z][a-z]*', x)), self.classes))
            
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
        
        if split == 'base' or split == 'novel':
            split = split + '_' + str(base2novel) + '_' + str(base2novelsplit)
            
        self.indices = []
        if str(frames)+'x'+str(rate)+'_'+split + 'indices_ucf24.json' in os.listdir('/'.join(anno_pth.split('/')[:-1])):
            self.indices = json.load(open(os.path.join('/'.join(anno_pth.split('/')[:-1]), str(frames)+'x'+str(rate)+'_'+split + 'indices_ucf24.json')))
        else:
            for v in self.video_list:
                vtubes = sum(self.gttubes[v].values(), [])
                self.indices += [(v, i) for i in range(1, self.nframes[v] + 2 - self.T) if tubelet_in_out_tubes(vtubes, i, self.T) and tubelet_has_gt(vtubes, i, self.T)]
            json.dump(self.indices, open(os.path.join('/'.join(anno_pth.split('/')[:-1]), str(frames)+'x'+str(rate)+'_'+split + 'indices_ucf24.json'), 'w'))
            
        print(split, 'dataset (ucf24) loaded!')
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        v, frame = self.indices[idx]
        v = os.path.join(self.video_dir, v)
        vr = VideoReader(v + '.avi')

        label = ' '.join(re.findall('[A-Z][a-z]*', v.split('/')[-2]))
        
        # sample random segment from video
        start = frame - 1
        end = start + self.T
        clip_idx = list(range(start, end, self.rate))

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
ucf24textaug = json.load(open('gpt/GPT_UCF24.json'))

def textaugucf24(text_lst):
    aug_text_lst = []
    for text in text_lst:
        if text not in ucf24textaug:
            aug_text_lst.append(text)
        else:
            aug_text_lst.append(random.choice(ucf24textaug[text]))
    return aug_text_lst
'''
