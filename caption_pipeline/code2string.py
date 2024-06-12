import re
import random
import config
from tqdm import tqdm
from libs.body_model import BodyModel
from collections import defaultdict
from caption_pipeline import caption_pipeline
from caption_pipeline.grammar import *

merge_rules1 = {
    'left leg': {'left thigh', 'left calf'},
    'right leg': {'right thigh', 'right calf'},
    'left arm': {'left upper arm', 'left forearm'},
    'right arm': {'right upper arm', 'right forearm'},
    'left foot': {'left toes', 'left heel'},
    'right foot': {'right toes', 'right heel'},
}
merge_rules2 = {
    'both arms': {'left arm', 'right arm'},
    'both knees': {'left knee', 'right knee'},
    'both elbows': {'left elbow', 'right elbow'},
    'both thighs': {'left thighs', 'right thighs'},
    'both calves': {'left calf', 'right calf'},
    'both forearms': {'left forearm', 'right forearm'},
    'both legs': {'left leg', 'right leg'},
    'head and upper body': {'upper body', 'head'},
    'both feet': {'left foot', 'right foot'},
    'both hands': {'left hand', 'right hand'},
    'both upper arms': {'left upper arm', 'right upper arm'},
}

# special_rules = {
    
# }

random_rules = {
    'bent slightly': ['bent slightly', "slightly bent",'bent a bit'],
    'bent at right angle': ['bent at right angle', 'bent at near 90 degrees', 'bent at 90 degrees', 'bent at an angle of almost 90 degrees'],
    'in front of the chest': ['in front of the chest', 'in front of the body'],
    'completely bent': ['completely bent', 'bent completely'],
    'located': ['located', '', 'situated', 'positioned', 'placed'],
    'upwards': ['upwards', 'up'],
    'vertical': ['vertical', 'erect'],
}

joint_labels = {
    "left knee": "legs",
    "right knee": "legs",
    "left calf": "legs",
    "right calf": "legs",
    "left thigh": "legs",
    "right thigh": "leg",
    "left leg": "legs", 
    "right leg": "legs",
    "left foot": "legs", 
    "right foot": "legs",
    "left heel": "legs", 
    "right heel": "legs",
    "left toes": "legs", 
    "right toes": "legs",

    "left elbow": "arms", 
    "right elbow": "arms",
    "left upper arm": "arms", 
    "right upper arm": "arms",
    "left forearm": "arms", 
    "right forearm": "arms",
    "left arm": "arms", 
    "right arm": "arms", 
    "left hand": "arms", 
    "right hand": "arms",

    "upper body": "body", 
    'head': "body",
    'head and upper body': "body",
    "hips": "body",
    "torso": "body",
    
    "both arms": "arms",
    "both hands": "arms",
    "both elbows": "arms",
    "both upper arms": "arms",
    "both forearms": "arms",
    "both knees": "legs",
    "both legs": "legs",
    "both thighs": "legs",
    "both calves": "legs",
    "both feet": "legs",
    "arms": "arms",
    "hands": "arms",
    "elbows": "arms",
    "upper arms": "arms",
    "forearms": "arms",
    "knees": "legs",
    "legs": "legs",
    "thighs": "legs",
    "calves": "legs",
    "feet": "legs",
}

# def simplify(codes):
#     bs = len(codes)
#     for i in range(bs):
#         code2joint = defaultdict(list)
#         simplfied = defaultdict(list)
#         for description in codes[i]:
#             code2joint[description[1]].append(description[0])
#         # merge left and right
#         for code, joints in code2joint.items():
#             for joint in joints:
#                 if 'left' in joint and joint.replace('left', 'right') in joints:
#                     simplfied[code].append(REPEATS[joint.split(' ', 1)[1]])
#                     continue
#                 if 'right' in joint and joint.replace('right', 'left') in joints:
#                     continue
#                 simplfied[code].append(joint)
#         codes[i] = [(joint, *code) for code, joints in simplfied.items() for joint in joints]
#     return codes

def mirror_caption(descriptions):
    mirror_descriptions = []
    for i in range(len(descriptions)):
        value_list = descriptions[i].lower()
        value_list = re.split(r'([.\s])', value_list)
        for j in range(len(value_list)):
            if 'left' in value_list[j] and 'right' in value_list[j]:
                print(descriptions[i])
                raise f"unexpected value_list" 
            elif 'left' in value_list[j]:
                value_list[j] = value_list[j].replace('left', 'right')
            elif 'right' in value_list[j]:
                value_list[j] = value_list[j].replace('right', 'left')
        mirror_descriptions.append(''.join(x.capitalize() for x in value_list))
    return mirror_descriptions

def generate_code(x, pipeline):
    y = pipeline.eval(x)
    code = pipeline.quantize(y).reshape(-1, len(pipeline.category_subjects))
    return code

def pose2code(pose):
    """
    pose : torch.tensor of shape (N, 22, 3)
    """
    bs = pose.shape[0]
    # setup body model
    body_model = BodyModel(config.SMPLH_BODY_MODEL_PATH, num_betas=16)
    body_model.eval()
    body_model.to(pose.device)

    smpl = body_model(root_orient=pose[:, 0], pose_body=pose[:, 1:22].view(bs, 63), pose_hand=pose[:, 22:].view(bs, -1))
    smpl.full_pose = smpl.full_pose.view(pose.shape[0], -1, 3)

    return {
        key: generate_code(smpl, caption_pipeline[key]) for key in caption_pipeline.keys()
    }

def code2string(code):
    triple = code2triple(code)
    descriptions = triple2string(triple)
    return descriptions

def code2triple(code):
    """
    code : dict
    """
    captions = [caption_pipeline[key].totriple(code[key]) for key in code.keys()]
    descriptions = [[] for _ in range(len(captions[0]))]
    for caption in captions:
        descriptions = [description + word for description, word in zip(descriptions, caption)]
    return descriptions

def triple2label(triples, verpose=False):
    bs = len(triples)
    labels = []

    bar = tqdm(range(bs)) if verpose else range(bs)
    for i in bar:
        label = [caption_pipeline[triple[1]].label(triple) for triple in triples[i]]
        labels.append(label)
    return labels

def triple2string(descriptions):
    bs = len(descriptions)
    for i in range(bs):
        code2joint = defaultdict(list)
        for description in descriptions[i]:
            code2joint[description[1:]].append(description[0])
        for key, value in merge_rules1.items():
            for code, joints in code2joint.items():
                if value.issubset(set(joints)):
                    code2joint[code] = list(set(joints)-value) + [key]
        for key, value in merge_rules2.items():
            for code, joints in code2joint.items():
                if value.issubset(set(joints)):
                    code2joint[code] = list(set(joints)-value) + [key]
        descriptions[i] = [(joint, *code) for code, joints in code2joint.items() for joint in joints]
        descriptions[i] = [caption_pipeline[code[1]].tostring(code, fullcode=descriptions[i]) for code in descriptions[i]]
        descriptions[i] = [des for des in descriptions[i] if des is not None]

        # fix subject
        subjects = {}
        for description in descriptions[i]:
            if description[0] not in subjects:
                subjects[description[0]] = random.choice([description[0], description[0].replace('both ', '')])
        descriptions[i] = [(subjects[description[0]], description[1]) for description in descriptions[i]]

        descriptions[i] = sorted(descriptions[i], key=lambda x: x[0])
    
    descriptions = random2string(descriptions)
    return descriptions

def random2string(code):
    bs = len(code)
    descriptions = []
    for i in range(bs):
        description = ""
        determiner = random.choices(DETERMINERS, weights=DETERMINERS_PROP)[0]
        # sort by body part
        labeled = defaultdict(list)
        for caption in code[i]:
            # randomization
            random_code = caption[1]
            for key, value in random_rules.items():
                if key in random_code:
                    random_code = random_code.replace(key, random.choice(value))
            labeled[joint_labels[caption[0]]].append((caption[0], random_code))
        # random.shuffle(labeled)
        keys = list(labeled.keys())
        random.shuffle(keys)
        for key in keys:
            values = labeled[key]
            transition = ""
            subject = ""
            for value in values:
                if transition == ' with ':
                    verb = ''
                else:
                    verb = ' are ' if is_repeats(value[0]) else ' is '

                if value[0] == subject:
                    transition = random.choice([' and ', ', '])
                    description += transition + ' ' + value[1]
                    continue

                subject = value[0]
                
                if 'both' in subject:
                    if determiner == 'the':
                        description += transition + subject + verb + ' ' + value[1]
                    else:
                        description += transition + random.choice(["", determiner]) + ' ' + subject + verb + ' ' + value[1]
                else:
                    description += transition + determiner + ' ' + subject + verb + ' ' + value[1]

                if transition in [' and ', ' with ', ' while ']:
                    transition = '. '
                else:
                    transition = random.choices(TEXT_TRANSITIONS, weights=TEXT_TRANSITIONS_PROP)[0]
        
            description += '. '

        description = re.sub("\s\s+", " ", description)
        description = '. '.join(x.capitalize() for x in description.split('. '))
        descriptions.append(description)
    return descriptions