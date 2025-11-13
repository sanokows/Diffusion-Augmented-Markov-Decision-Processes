"""
Camera configurations for DM Control Suite environments.
Based on the camera configurations from dm_control_suite.ipynb.
"""

# Camera configurations for different environments
CAMERAS = {
    "AcrobotSwingup": "fixed",
    "AcrobotSwingupSparse": "fixed",
    "BallInCup": "cam0",
    "CartpoleBalance": "fixed",
    "CartpoleBalanceSparse": "fixed",
    "CartpoleSwingup": "fixed",
    "CartpoleSwingupSparse": "fixed",
    "CheetahRun": "side",
    "FingerSpin": "cam0",
    "FingerTurnEasy": "cam0",
    "FingerTurnHard": "cam0",
    "FishSwim": "fixed_top",
    "HopperHop": "cam0",
    "HopperStand": "cam0",
    "HumanoidStand": "side",
    "HumanoidWalk": "side",
    "HumanoidRun": "side",
    "PendulumSwingup": "fixed",
    "PointMass": "cam0",
    "ReacherEasy": "fixed",
    "ReacherHard": "fixed",
    "SwimmerSwimmer6": "tracking1",
    "WalkerRun": "side",
    "WalkerWalk": "side",
    "WalkerStand": "side",



    "AlohaHandOver": None,
    "AlohaSinglePegInsertion": None,
    "PandaPickCube": None,
    "PandaPickCubeOrientation": None,
    "PandaPickCubeCartesian": None,
    "PandaOpenCabinet": None,
    "PandaRobotiqPushCube": None,
    "LeapCubeReorient": None,
    "LeapCubeRotateZAxis": None
}


def get_camera_config(env_name: str) -> str:
    """
    Get the camera configuration for a given environment name.
    
    Args:
        env_name: Name of the environment
        
    Returns:
        str: Camera configuration name
    """
    # First check exact match
    if env_name in CAMERAS:
        return CAMERAS[env_name]

def get_render_config(env_name: str, width: int = 320, height: int = 240) -> dict:
    """
    Get render configuration including camera and dimensions.
    
    Args:
        env_name: Name of the environment
        width: Video width in pixels
        height: Video height in pixels
        
    Returns:
        dict: Render configuration
    """
    return {
        "camera": get_camera_config(env_name),
        "width": width,
        "height": height,
    }

# Predefined render qualities
RENDER_QUALITIES = {
    "low": {"width": 240, "height": 180},
    "medium": {"width": 320, "height": 240},
    "high": {"width": 640, "height": 480},
    "hd": {"width": 1280, "height": 720},
}

def get_render_quality_config(env_name: str, quality: str = "medium") -> dict:
    """
    Get render configuration with predefined quality settings.
    
    Args:
        env_name: Name of the environment
        quality: Quality setting ("low", "medium", "high", "hd")
        
    Returns:
        dict: Render configuration with quality settings
    """
    if quality not in RENDER_QUALITIES:
        quality = "medium"
    
    config = get_render_config(env_name)
    config.update(RENDER_QUALITIES[quality])
    return config