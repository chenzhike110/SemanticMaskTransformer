from .posecode import PosecodeAngle, PosecodeVertical, PosecodeYaw, PosecodePitch, PosecodeHand, PosecodeHead, PosecodeBent, PosecodeDist, PosecodeGround, PosecodeFoot

caption_pipeline = {
    "PosecodeYaw": PosecodeYaw(),
    "PosecodePitch": PosecodePitch(),
    "PosecodeHead": PosecodeHead(),
    "PosecodeHand": PosecodeHand(),
    "PosecodeBent": PosecodeBent(),
    "PosecodeGround": PosecodeGround(),
    "PosecodeDist": PosecodeDist(),
    "PosecodeVertical": PosecodeVertical(),
    "PosecodeAngle": PosecodeAngle(),
    "PosecodeFoot": PosecodeFoot(),
}