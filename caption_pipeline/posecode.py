import os
import json
import torch
import random
from caption_pipeline.codebase import Posecode1D
from caption_pipeline.rotation_conversions import matrix_to_euler_angles
from libs.body_model.joint_names import SMPLH_JOINT_NAMES

class PosecodeAngle(Posecode1D):
    """
    Angle posecode for knees and elbows
    """ 
    def __init__(self) -> None:
        super().__init__()
        self.joints = ["left_knee", "right_knee", "left_elbow", "right_elbow"]
        self.joints_id = [SMPLH_JOINT_NAMES.index(label) for label in self.joints]
    
    def setup(self):
        self.category_subjects = ["left knee", "right knee", "left elbow", "right elbow"]
        self.category_labels = [["left leg"], ["right leg"], ["left arm"], ["right arm"]]
        self.renamed_subjects = {
            "left knee": "left leg", 
            "right knee": "right leg", 
            "left elbow": "left arm", 
            "right elbow": "right arm",
            "both knees": "both legs",
            "both elbows": "both arms"
        }
        self.category_strings = [
            'straight',
            'slightly bent',
            'partially bent',
            'bent at right angle',
            'almost completely bent',
            'completely bent',
        ]
        self.category_thresholds = [20, 45, 85, 115, 135]
        self.category_random = 2

    def eval(self, smpl):
        joints = smpl.full_pose[:, self.joints_id].view(-1, 3).cpu()
        rot_ang = torch.norm(joints, dim=-1)
        return rot_ang * 180.0 / torch.pi
    
    def tostring(self, code, **kargs):
        if code[2] == 0:
            return (random.choice([self.renamed_subjects[code[0]], code[0]]), self.category_strings[code[-1]])
        return (code[0], self.category_strings[code[-1]])
    
class PosecodeVertical(Posecode1D):
    """
    Vertical and Horizontal posecode for thigh, calf, forearm, upper arm
    """
    def __init__(self) -> None:
        super().__init__()
        self.vertical_vec = torch.tensor([0.0, 0.0, 1.0])
        self.joints_pair = [
            ["left_hip","left_knee"], 
            ["right_hip", "right_knee"], 
            ["left_shoulder", "left_elbow"], 
            ["right_shoulder", "right_elbow"],
            ["left_knee", "left_ankle"],
            ["right_knee", "right_ankle"],
            ["left_elbow", "left_wrist"],
            ["right_elbow", "right_wrist"],
            ['spine3', 'pelvis'],
        ]
        self.joints_id = [[SMPLH_JOINT_NAMES.index(sublist[0]), SMPLH_JOINT_NAMES.index(sublist[1])] for sublist in self.joints_pair]
    
    def setup(self):
        self.category_subjects = ["left thigh", "right thigh", "left upper arm", "right upper arm", "left calf", "right calf", "left forearm", "right forearm", "upper body"]
        self.category_labels = [["left leg"], ["right leg"], ["left arm", "body"], ["right arm", "body"], ["left leg"], ["right leg"], ["left arm", "body"], ["right arm", "body"], ["body"]]
        self.category_strings = [
            'vertical',
            '',
            'horizontal',
        ]
        self.category_random = 2
        self.category_thresholds = [15, 80]
    
    def eval(self, smpl):
        joints = smpl.Jtr[:, self.joints_id].view(-1, 2, 3).cpu()
        body_part_vec = torch.nn.functional.normalize(joints[:, 0] - joints[:, 1], dim=-1)
        cos_tensor = (self.vertical_vec*body_part_vec).sum(-1).abs() # cos(theta)
        ang = torch.acos(cos_tensor)
        return ang * 180.0 / torch.pi
    
    def tostring(self, code, **kargs):
        if self.category_strings[code[2]] != '':
            return (code[0], self.category_strings[code[2]])
        return None
    
class PosecodeYaw(Posecode1D):
    """
    Direction posecode for hip, shoulder, head
    """
    def __init__(self) -> None:
        super().__init__()
        self.vertical_vec = torch.tensor([0.0, 0.0, 1.0])
        self.ignore_threshold = torch.pi/9
        self.joints_pair = [
            ["left_shoulder", "left_wrist"], 
            ["right_shoulder", "right_wrist"],
            ["left_hip","left_foot"], 
            ["right_hip", "right_foot"]
        ]
        self.pair_id = [[SMPLH_JOINT_NAMES.index(sublist[0]), SMPLH_JOINT_NAMES.index(sublist[1])] for sublist in self.joints_pair]
        self.joints = ['spine2', 'head']
        self.joints_id = [SMPLH_JOINT_NAMES.index(joint) for joint in self.joints]

    def setup(self):
        self.category_subjects = ["left arm", "right arm", "left leg", "right leg", "upper body", 'head']
        self.category_labels = [["left arm"], ["right arm"], ["left leg"], ["right leg"], ["body"], ['head']]
        self.category_verb = lambda x: 'stretched ' if 'arm' in x else ('stepped ' if 'leg' in x else 'faced ')
        self.category_strings = [
            '',
            'right',
            'right front',
            'front',
            'left front',
            'left',
            'left back',
            'back',
            'right back',
            'right',
        ]
        self.category_random = 5
        self.category_thresholds = [-180,-180+22.5, -180+80, -180+100, -180+157.5, -180+202.5, -180+260, -180+280, -180+337.5]
    
    def eval(self, smpl):
        bs = smpl.Jtr.shape[0]
        joints = smpl.Jtr[:, self.pair_id].view(-1, 2, 3).cpu()
        body_part_vec = torch.nn.functional.normalize(joints[:, 0] - joints[:, 1], dim=-1)
        cos_tensor = (self.vertical_vec*body_part_vec).sum(-1).abs() # cos(theta)
        ang = torch.acos(cos_tensor)
        ignore_mask = ang < self.ignore_threshold

        directions = joints[:, 1] - joints[:, 0]
        theta_x = torch.atan2(directions[:, 1], directions[:, 0])
        theta_x[ignore_mask] = - torch.pi * 2

        x_rotmat = smpl.Jrot[:, self.joints_id].view(-1, 3, 3).cpu()
        yaw = torch.atan2(x_rotmat[:,1,-1], x_rotmat[:,0,-1])
        ang_1 = torch.acos(x_rotmat[:,2,-1].abs())
        ignore_mask = ang_1 < self.ignore_threshold
        yaw[ignore_mask] = - torch.pi * 2

        theta_x = torch.cat([theta_x.reshape(bs, -1), yaw.reshape(bs, -1)],dim=1).flatten()

        return theta_x * 180.0 / torch.pi
    
    def tostring(self, code, **kargs):
        if self.category_strings[code[2]] == '':
            return None
        else:
            if 'arm' not in code[0] and 'leg' not in code[0]:
                verb = self.category_verb(code[0]) if code[2] == 3 else random.choice([self.category_verb(code[0]), 'turned '])
                return (code[0], verb + 'to the ' + self.category_strings[code[2]])
            return (code[0], self.category_verb(code[0]) + 'to the ' + self.category_strings[code[2]])
    
class PosecodePitch(Posecode1D):
    def __init__(self) -> None:
        super().__init__()
        self.joints_pair = [
            ["left_shoulder", "left_wrist"], 
            ["right_shoulder", "right_wrist"],
            ["left_hip","left_foot"], 
            ["right_hip", "right_foot"],
        ]
        self.pair_id = [[SMPLH_JOINT_NAMES.index(sublist[0]), SMPLH_JOINT_NAMES.index(sublist[1])] for sublist in self.joints_pair]
        self.joints = ['head']
        self.joints_id = [SMPLH_JOINT_NAMES.index(joint) for joint in self.joints]
        self.vertical_vec = torch.tensor([0.0, 0.0, 1.0])

    def setup(self):
        self.category_subjects = ["left arm", "right arm", "left leg", "right leg", 'head']
        self.category_labels = [["left arm"], ["right arm"], ["left leg"], ["right leg"], ['head']]
        self.category_thresholds = torch.tensor([[30, 135], [30, 135], [80, 190], [80, 190], [45, 135]])
        self.category_strings = ['upwards', '' , 'downwards']
        self.category_random = 5

    def eval(self, smpl):
        bs = smpl.Jtr.shape[0]
        joints = smpl.Jtr[:, self.pair_id].view(-1, 2, 3).cpu()
        body_part_vec = torch.nn.functional.normalize(joints[:, 1] - joints[:, 0], dim=-1)
        cos_tensor = (self.vertical_vec*body_part_vec).sum(-1) # cos(theta)
        ang = torch.acos(cos_tensor)

        x_rotmat = smpl.Jrot[:, self.joints_id].view(-1, 3, 3).cpu()
        ang_1 = torch.acos(x_rotmat[:,2,-1])

        theta_x = torch.cat([ang.reshape(bs, -1), ang_1.reshape(bs, -1)],dim=1)

        return theta_x * 180.0 / torch.pi
    
    def quantize(self, x):
        # apply random offsets
        x += (torch.ones_like(x) - 0.5) * 2 * self.category_random
        ret = torch.ones(x.shape) * 2
        for i in range(1, -1, -1):
            ret[x<=self.category_thresholds[:, i]] = i
        return ret.int()
    
    def tostring(self, code, **kargs):
        if self.category_strings[code[2]] == '':
            return None
        else:
            if 'arm' in code[0]:
                if code[2] < 1:
                    return (code[0], random.choice(['raised up', 'put up', 'up']))
                else:
                    return (code[0], random.choice(['hanging down', 'put down']))
            elif 'leg' in code[0]:
                if code[2] < 1:
                    return (code[0], random.choice(['raised upwards', 'extended upwards']))
            else:
                if code[2] < 1:
                    return (code[0], random.choice(['raised up', 'put up']))
                else:
                    return (code[0], random.choice(['down', 'faced down', 'turned down']))
        
class PosecodeHead(Posecode1D):
    def __init__(self) -> None:
        super().__init__()
        self.joints = ['spine2', 'head']
        self.joints_id = [SMPLH_JOINT_NAMES.index(joint) for joint in self.joints]

    def setup(self):
        self.category_subjects = ['head']
        self.category_labels = [['head']]
        self.category_thresholds = [135]
        self.category_strings = ["", "upside down"]
        self.category_random = 5

    def eval(self, smpl):
        bs = smpl.Jrot.shape[0]
        x_rotmat = smpl.Jrot[:, self.joints_id].view(-1, 3, 3).cpu()

        ang = torch.acos(x_rotmat[:,2,1])

        return ang * 180.0 / torch.pi
    
    def tostring(self, code, **kargs):
        if self.category_strings[code[2]] == '':
            return None
        else:
            return (code[0], self.category_strings[code[2]])
    
class PosecodeHand(Posecode1D):
    def __init__(self) -> None:
        super().__init__()
        self.joints = ['left_wrist', 'right_wrist']
        self.targets = ['spine', 'spine1', 'spine2', 'hips', 'head', 'left_knee', 'right_knee', 'leftUpLeg', 'rightUpLeg', 'leftLeg', 'rightLeg', 'leftToeBase', 'leftFoot', 'rightToeBase', 'rightFoot',]
        self.joints_id = [SMPLH_JOINT_NAMES.index(joint) for joint in self.joints]
        with open(os.path.join(os.path.dirname(__file__), 'smpl_vert_segmentation.json')) as f:
            self.j2v = json.load(f)
            self.v2j = {i:j for j in self.j2v.keys() for i in self.j2v[j] if j in self.targets}
        self.candidate_v = torch.tensor(list(self.v2j.keys()))
        self.distance_threshold = 0.2

    def setup(self):
        self.category_subjects = ["left hand", "right hand"]
        self.category_label_sub = [["left arm"], ["right arm"]]
        self.category_objects = ['stomache', 'chest', 'chest', 'hips', 'head', 'left knee', 'right knee', 'left thigh', 'right thigh', 'left calf', 'right calf', 'left foot', 'left foot', 'right foot', 'right foot',]
        self.category_label_obj = [["body"], ["body"], ["body"], ["body"], ['head'], ['left leg'], ['right leg'], ['left leg'], ['right leg'], ['left leg'], ['right leg'], ['left leg'],['left leg'], ['right leg'], ['right leg']]
        self.category_verb = []
        self.category_strings = [
            ""
        ]

    # def totriple(self, code):
    #     bs = code.shape[0]
    #     descriptions = []
    #     for i in range(bs):
    #         description = []
    #         for j in range(code[i].shape[0]):
    #             description.append((self.category_subjects[j], self.__class__.__name__, code[i][j].item()))
    #         descriptions.append(description)
    #     return descriptions
    
    def eval(self, smpl):
        # find the nearest joint
        bs = smpl.Jtr.shape[0]
        joints = smpl.Jtr[:, self.joints_id].cpu()
        # compute the distance to the target vertices
        candidate_v = smpl.v[:, self.candidate_v].cpu()

        directions = joints.clone()
        directions = torch.zeros(bs, len(self.joints), dtype=torch.long)
        related_labels = []

        for i in range(len(self.joints)):
            dist = torch.nn.functional.pairwise_distance(joints[:, [i]].repeat(1, len(self.candidate_v), 1), candidate_v)
            # min_index, min_dist = torch.min(dist, dim=1, keepdim=True)
            min_index = torch.argmin(dist, dim=1)
            labels = []
            for j in range(bs):
                
                related_label = self.v2j[self.candidate_v[min_index[j]].item()]
                labels.append(self.targets.index(related_label))

                # get bounding box for the body part
                top = smpl.v[j, self.j2v[related_label], 2].max() # 1
                bottom = smpl.v[j, self.j2v[related_label], 2].min()
                left = smpl.v[j, self.j2v[related_label], 0].max() # 0
                right = smpl.v[j, self.j2v[related_label], 0].min()
                front = smpl.v[j, self.j2v[related_label], 1].min() # 2
                back = smpl.v[j, self.j2v[related_label], 1].max()

                if self.targets.index(related_label) > 4:
                    directions[j, i] = 000 if dist[j, min_index[j]] < self.distance_threshold else 444
                else:
                    directions[j, i]  = (joints[j, i, 2]>top)*10+(joints[j, i, 2]>bottom)*10+(joints[j, i, 0]>right)*100+(joints[j, i, 0]>left)*100+(joints[j, i, 1]>back)*1+(joints[j, i, 1]>front)*1

            related_labels.append(labels)
        related_labels = torch.tensor(related_labels).T
        
        return directions, related_labels

    def quantize(self, x):
        directions, labels = x
        ret = directions
        ret += labels * 1000
        return ret.long()
    
    def tostring(self, code, **kargs):
        obj = self.category_objects[code[2] // 1000]
        value = str(code[2]%1000).zfill(3)

        if obj == 'head':
            direction = 'upper' if value[1] == '2' else ('lower' if value[1] == '0' else "")
            direction += ' left' if value[0] == '2' else (' right' if value[0] == '0' else "")
            if direction == "":
                return None
            if value[2] == 0:
                return (code[0], random.choice([
                    f' located at the {direction} side of the head, in front of the head',
                    f' located at the {direction} front of the head',
                ]))
            elif value[2] == 2:
                return (code[0], random.choice([
                    f' located at the {direction} side of the head, behind the head',
                    f' located at the {direction} back of the head',
                ]))
            else:
                return (code[0], f' located at the {direction} side of the head',)
        elif obj == 'hips':
            if value[0] == '2':
                obj = 'left hips'
            elif value[0] == '0':
                obj = 'right hips'
            direction = ' left' if value[0] == '2' else ('right' if value[0] == '0' else "")
            direction += ' front' if value[2] == '0' else (' back' if value[2] == '2' else "")
            if direction == "":
                return (code[0], ' near the ' + obj)
            return (code[0], f'located at the {direction} side of ' + obj)
        elif obj in ['chest', 'stomache']:
            if value[0] != '1':
                direction = ' left' if value[0] == '2' else (' right' if value[0] == '0' else "")
                if value[1] == '2':
                    return (code[0], f'raised higher than ' + obj + f' , at the {direction} side')
                elif value[1] == '0':
                    return (code[0], f'raised lower than ' + obj + f' , at the {direction} side')
                else:
                    return (code[0], f'raised at the level of ' + obj + f' , at the {direction} side')
            else:
                if value[2] == '0':
                    return (code[0], f'located in front of ' + obj)
                elif value[2] == '2':
                    return (code[0], f'located behind the body')
                else:
                    return None
        else:
            if value == '000':
                return (code[0], ' near the ' + obj)
            return None
    
    def label(self, code):
        obj = self.category_objects[code[2] // 1000]
        return self.category_label_sub[self.category_subjects.index(code[0])] + self.category_label_obj[self.category_objects.index(obj)]

class PosecodeDist(Posecode1D):
    def __init__(self) -> None:
        super().__init__()
        self.joints_pair = [
            ['left_wrist', 'right_wrist'], 
            ["left_ankle", "right_ankle"],
            ["left_shoulder","right_shoulder"],
        ]
        self.pair_id = [[SMPLH_JOINT_NAMES.index(sublist[0]), SMPLH_JOINT_NAMES.index(sublist[1])] for sublist in self.joints_pair]

    def setup(self):
        self.category_subjects = ["both hands", "both feet"]
        self.category_labels = [['left arm', 'right arm'], ['left leg', 'right leg']]
        self.category_strings = [
            "close to each other",
            "",
            "shoulder width apart",
            ""
        ]
        self.category_thresholds = [0.4, 0.9, 1.1]
        self.category_random = 0


    def eval(self, smpl):
        joints = smpl.Jtr[:, self.pair_id].cpu()

        dist = torch.nn.functional.pairwise_distance(joints[:, :, 0], joints[:, :, 1])
        value = dist / dist[:, [-1]]

        return value[:, :-1]
    
    def tostring(self, code, **kargs):
        if self.category_strings[code[2]] == '':
            return None
        else:
            return (code[0], self.category_strings[code[2]])

class PosecodeGround(Posecode1D):
    def __init__(self) -> None:
        super().__init__()
        self.joints = [['leftToeBase'], ['rightToeBase'], ['leftFoot'], ['rightFoot'], ['leftHand', 'leftHandIndex1'], ['rightHand', 'rightHandIndex1'], ['hips']]
        self.candidate_v = []
        with open(os.path.join(os.path.dirname(__file__), 'smpl_vert_segmentation.json')) as f:
            self.j2v = json.load(f)
        for keys in self.joints:
            candidate_v = []
            for key in keys:
                candidate_v += self.j2v[key]
            self.candidate_v.append(candidate_v)
        self.distance_threshold = 0.2

    def setup(self):
        self.category_subjects = ['left toes', 'right toes', 'left heel', 'right heel', 'left hand', 'right hand', 'hips']
        self.category_labels = [["left leg"], ["right leg"], ["left leg"], ["right leg"], ["left arm"], ["right arm"], ["body"]]
        self.category_thresholds = [0.10, 0.25]
        self.category_strings_foot = ["fixed on the ground", "slightly off the ground", "in the air"]
        self.category_strings_hand = ["placed on the ground", "near the ground"]
        self.category_strings_hip = ["on the ground", "near the ground"]
        self.category_random = 0.02
    
    def eval(self, smpl):
        bs = smpl.v.shape[0]
        ground = smpl.v[..., 2].min(1)[0].cpu().unsqueeze(1)
        values = []
        for candidate in self.candidate_v:
            candidate_v = smpl.v[:, candidate].cpu()
            dist = (candidate_v[..., 2] - ground)
            values.append(dist.min(1)[0].reshape(bs, 1))
        return torch.cat(values, dim=1)
    
    def tostring(self, code, **kargs):
        if code[0] in ['left toes', 'right toes', 'left heel', 'right heel']:
            return (code[0], self.category_strings_foot[code[2]])
        else:
            if code[2] > 1:
                return None
            if code[0] in ['left hand', 'right hand']:
                return (code[0], self.category_strings_hand[code[2]])
            return (code[0], self.category_strings_hip[code[2]])
        
class PosecodeFoot(Posecode1D):
    def __init__(self) -> None:
        super().__init__()
        self.joints_pair = [
            ['left_foot', 'pelvis'],
            ['right_foot', 'pelvis'],
            ['left_foot', 'right_foot'],
        ]
        self.pair_id = [[SMPLH_JOINT_NAMES.index(sublist[0]), SMPLH_JOINT_NAMES.index(sublist[1])] for sublist in self.joints_pair]
    
    def setup(self):
        self.category_subjects = ['left foot', 'right foot', 'feet_x', 'feet_y']
        self.category_thresholds = [-0.15, -0.05, 0.05, 0.15]
        self.category_random = 0.02
    
    def eval(self, smpl):
        joints = smpl.Jtr[:, self.pair_id].cpu()
        loc = joints[:, :, 0] - joints[:, :, 1]
        ans = torch.cat([loc[:, :, 1], loc[:, [2], 0]], dim=1)
        return ans
    
    def quantize(self, x):
        ans = super().quantize(x)
        if ans[:, 0] > 2 and ans[:, 1] < 2:
            ans[:, 2] = 2
        elif ans[:, 0] < 2 and ans[:, 1] > 2:
            ans[:, 2] = 2
        return ans
    
    def tostring(self, code, **kargs):
        if code[0] in ['left foot', 'right foot']:
            if code[2] == 0:
                return (code[0], 'in front of the torso')
            elif code[2] == 1:
                return (code[0], 'slightly stepped forward')
            elif code[2] == 3:
                return (code[0], 'slightly stepped backward')
            elif code[2] == 4:
                return (code[0], 'behind the torso')
        elif code[0] == 'feet_x':
            if code[2] < 2:
                return random.choice([('left foot', 'in front of the right foot'), ('right foot', 'behind the left foot')])
            elif code[2] > 2:
                return random.choice([('left foot', 'behind the right foot'), ('right foot', 'in front of the left foot')])
        elif code[0] == 'feet_y':
            if code[2] >= 2:
                return
            else:
                return random.choice([('left foot', 'to the right of the right foot'), ('right foot', 'to the left of the left foot')])
        return

class PosecodeBent(Posecode1D):
    def __init__(self) -> None:
        super().__init__()
        self.joints_pair = [
            ['left_ankle', 'neck'],
            ['right_ankle', 'neck'],
            ['neck', 'pelvis'],
        ]
        self.pair_id = [[SMPLH_JOINT_NAMES.index(sublist[0]), SMPLH_JOINT_NAMES.index(sublist[1])] for sublist in self.joints_pair]

    def setup(self):
        self.category_subjects = ['torso']
        self.category_thresholds = [-0.15, -0.075, 0.075, 0.15]
        self.category_random = 0.01
        
    def eval(self, smpl):
        joints = smpl.Jtr[:, self.pair_id].cpu()
        return joints[:, :, 0] - joints[:, :, 1]
    
    def quantize(self, x):
        ret = torch.ones(x.shape) * len(self.category_thresholds)
        for i in range(len(self.category_thresholds)-1, -1, -1):
            ret[x<=self.category_thresholds[i]] = i

        quant = ret[:, 0, 2] * 10000 + ret[:, 1, 2] * 1000 + ret[:, 2, 0] * 100 + ret[:, 2, 1] * 10 + ret[:, 2, 2]
        # below = ret[:, [0,1], 2].view(ret.shape[0], -1)
        # lrfb = ret[:, 2, [0,1]].view(ret.shape[0], -1)

        return quant
    
    def label(self, code):
        return ['body']

    def tostring(self, code, **kargs):
        if code[2] // 1000 > 0:
            return
        else:
            templete = "leaning "
                
            code_2 = code[2] % 1000
            if (code_2 // 10) % 10 <= 1:
                if code_2 % 10 > 2:
                    templete = 'leaning forward '
                else:
                    templete = 'bent forward '
            elif (code_2 // 10) % 10 >= 3:
                if code_2 % 10 > 2:
                    templete = 'leaning backward '
                else:
                    templete = 'bent backward '
            
            if (code_2 // 10) % 10 == 1 or (code_2 // 10) % 10 == 3:
                templete = 'slightly ' + templete

            if code_2 // 100 == 4:
                templete += 'to the left'
            elif code_2 // 100 == 0:
                templete += 'to the right'
            
            if templete == 'leaning ':
                return 

            return (code[0], templete.replace('leaning', random.choice(['leaning', 'tilted', 'inclined'])))            