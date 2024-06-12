import os
import math
import json
import torch
import config
import random
from tqdm import tqdm
from torch.utils.data import Dataset
from caption_pipeline.code2string import triple2string, triple2label

################################################################################
## POSESCRIPT with Semantic Mask
################################################################################
class PoseScriptSM(Dataset):

    def __init__(self, cache=None, version='default', split='train', mask_prob=0.25) -> None:
        super().__init__()

        if cache is None:
            cache = os.path.join(config.POSESCRIPT_LOCATION, f'%s/PoseScript_{version}_split_{split}.pt')

        if os.path.exists(cache):
            self.data = torch.load(cache)
        else:
            self.cache_data(cache)

        self.version = version
        self.mask_prob = mask_prob

        with open(config.file_posescript_split[version]%split, 'r') as f:
            dataIDs = json.load(f) # split dependent
            self.data[3] = {k:self.data[3][k] for k in dataIDs if k in self.data[3]}
            self.dataIDs = list(self.data[3].keys())
    
    def cache_data(self, cache):

        from caption_pipeline.code2string import code2triple, pose2code, mirror_caption

        captions = []
        bp_labels = []
        print("cacheing data to {}".format(cache))
        
        poses = torch.cat(poses, dim=0)

        poses, ids = torch.load(config.supported_datasets["PoseScript"])
        batch_size = 1000

        for t in tqdm(range(math.ceil(poses.shape[0] / batch_size))):
            code = pose2code(poses[t*batch_size:(t+1)*batch_size])
            code = code2triple(code)
            label = triple2label(code)
            captions += code
            bp_labels += label

        with open(config.ACTION_LABEL_FILE, 'r') as f:
            action_labels = json.load(f)
            keys = list(action_labels.keys())
            for key in keys:
                action_labels['M'+key] = mirror_caption(action_labels[key])
        
        if cache:
            torch.save([poses, captions, ids, action_labels, bp_labels], cache)
        self.data = [poses, captions, ids, action_labels, bp_labels]

    def bp2mask(self, bps):
        mask = [0] * 5
        if 'left leg' in bps:
            mask[0] = 1
        if 'right leg' in bps:
            mask[1] = 1
        if 'left arm' in bps:
            mask[3] = 1
        if 'right arm' in bps:
            mask[4] = 1
        mask[2] = 1
        return mask

    def __len__(self) -> int:
        return len(self.dataIDs)
    
    def __getitem__(self, index):
        key = self.dataIDs[index]
        index = self.data[2].index(key)

        pose = self.data[0][index]
        posecodes = self.data[1][index]
        bp_label = self.data[4][index]
        
        action_label = self.data[3][key]

        bps_all = list(set([bp for bps in bp_label for bp in bps]))

        # remove some pose codes
        mask = torch.rand(len(bps_all)) < self.mask_prob
        mask_bps = [bps_all[i] for i in range(len(bps_all)) if mask[i]]
        bps = set(bps_all) - set(mask_bps)
        if len(bps) == 0:
            bps = [random.choice([bps_all])[0]]
            mask_bps.remove(bps[0])
        bp_mask = self.bp2mask(bps)

        posecodes_masked = []
        for i in range(len(posecodes)):
            mask = False
            for bp in mask_bps:
                if bp in bp_label[i]:
                    mask = True
                    break
            if not mask:
                posecodes_masked.append(posecodes[i])

        caption = action_label[0] + ' ' + triple2string([posecodes_masked])[0]
        
        return pose, caption, bp_mask