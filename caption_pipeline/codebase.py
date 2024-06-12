import torch
import random
from caption_pipeline.grammar import REPEATS

class Posecode1D:
    """
    Generic posecode class.
    """
    def __init__(self) -> None:
        # define subjects
        self.category_subjects = None
        # define verbs (list)
        self.category_verb = None
        # define categories (list)
        self.category_strings = None
        # thresholds for each categories
        self.category_thresholds = None
        # random offsets
        self.category_random = 0
        # body part label
        self.category_labels = []

        self.setup()
    
    def setup(self):
        raise NotImplementedError
    
    def totriple(self, code):
        bs = code.shape[0]
        descriptions = []
        for i in range(bs):
            description = []
            for j in range(code[i].shape[0]):
                description.append((self.category_subjects[j], self.__class__.__name__, code[i][j].item()))
            descriptions.append(description)
        return descriptions
    
    def tostring(self, code, **kargs):
        raise NotImplementedError

    def quantize(self, x):
        # apply random offsets
        x += (torch.ones_like(x) - 0.5) * 2 * self.category_random
        ret = torch.ones(x.shape) * len(self.category_thresholds)
        for i in range(len(self.category_thresholds)-1, -1, -1):
            ret[x<=self.category_thresholds[i]] = i
        return ret.int()
    
    def label(self, code):
        return self.category_labels[self.category_subjects.index(code[0])]
