#### GLOBAL PATH CONFIGURATION ####
import os
MAIN_DIR = os.path.realpath(__file__)
MAIN_DIR = os.path.dirname(MAIN_DIR)

# BODY MODEL
SMPLH_BODY_MODEL_PATH = MAIN_DIR + '/data/SMPL/smplh/SMPLH_NEUTRAL.npz'

# DATASET
AMASS_FILE_LOCATION = MAIN_DIR + "/data/AMASS/"
POSESCRIPT_LOCATION = MAIN_DIR + "/data/posescript"

supported_datasets = {
    "AMASS":AMASS_FILE_LOCATION,
    "PoseScript": POSESCRIPT_LOCATION + "/pose_100k.pt"
}

# ACTION_LABEL
ACTION_LABEL_FILE = POSESCRIPT_LOCATION + "/action_labels.json"

# TRAIN/TEST/VALIDATION SPLIT
file_posescript_split = {
    "default": f"{POSESCRIPT_LOCATION}/%s_ids_100k.json", # %s --> (train|val|test)
}