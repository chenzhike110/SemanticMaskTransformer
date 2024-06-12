#### GLOBAL PATH CONFIGURATION ####
import os
MAIN_DIR = os.path.realpath(__file__)
MAIN_DIR = os.path.dirname(MAIN_DIR)

# BODY MODEL
SMPLH_BODY_MODEL_PATH = MAIN_DIR + '/data/SMPL/smplh/SMPLH_NEUTRAL.npz'

# DATASET
AMASS_FILE_LOCATION = MAIN_DIR + f"/data/AMASS/"
supported_datasets = {"AMASS":AMASS_FILE_LOCATION}
