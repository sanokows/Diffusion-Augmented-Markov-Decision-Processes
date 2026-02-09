"""Custom G1 joystick env with an added box in the worldbody."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence
import xml.etree.ElementTree as ET

from etils import epath
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.g1 import base as g1_base
from mujoco_playground._src.locomotion.g1 import g1_constants as consts
from mujoco_playground._src.locomotion.g1 import joystick as g1_joystick


_BOX_BODY_NAME = "custom_box_body"
_BOX_GEOM_NAME = "custom_box_geom"


def default_config() -> config_dict.ConfigDict:
    return g1_joystick.default_config()


def _format_floats(values: Iterable[float]) -> str:
    return " ".join(f"{v:.6g}" for v in values)


def _ensure_box_in_worldbody(
    xml_text: str,
    *,
    box_pos: Sequence[float],
    box_size: Sequence[float],
    box_rgba: Sequence[float],
) -> str:
    """Insert a static box into the worldbody if it is not present."""
    root = ET.fromstring(xml_text)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Expected <worldbody> in MJCF but none found.")

    if worldbody.find(f".//body[@name='{_BOX_BODY_NAME}']") is not None:
        return xml_text
    if worldbody.find(f".//geom[@name='{_BOX_GEOM_NAME}']") is not None:
        return xml_text

    body = ET.SubElement(worldbody, "body", name=_BOX_BODY_NAME, pos=_format_floats(box_pos))
    ET.SubElement(
        body,
        "geom",
        name=_BOX_GEOM_NAME,
        type="box",
        size=_format_floats(box_size),
        rgba=_format_floats(box_rgba),
    )
    return ET.tostring(root, encoding="unicode")


class G1JoystickFlatTerrainWithBox(g1_joystick.Joystick):
    """G1 joystick env on flat terrain with a static box obstacle."""

    def __init__(
        self,
        *,
        config: Optional[config_dict.ConfigDict] = None,
        config_overrides: Optional[dict] = None,
        box_pos: Sequence[float] = (1.0, 0.0, 0.2),
        box_size: Sequence[float] = (0.2, 0.2, 0.2),
        box_rgba: Sequence[float] = (0.8, 0.2, 0.2, 1.0),
    ) -> None:
        if config is None:
            config = default_config()
        mjx_env.MjxEnv.__init__(self, config, config_overrides)

        xml_path = consts.task_to_xml("flat_terrain")
        xml_text = epath.Path(xml_path).read_text()
        xml_text = _ensure_box_in_worldbody(
            xml_text,
            box_pos=box_pos,
            box_size=box_size,
            box_rgba=box_rgba,
        )

        self._mj_model = mujoco.MjModel.from_xml_string(
            xml_text, assets=g1_base.get_assets()
        )
        self._mj_model.opt.timestep = self.sim_dt

        if self._config.restricted_joint_range:
            self._mj_model.jnt_range[1:] = consts.RESTRICTED_JOINT_RANGE
            self._mj_model.actuator_ctrlrange[:] = consts.RESTRICTED_JOINT_RANGE

        # Increase offscreen framebuffer size to render at higher resolutions.
        self._mj_model.vis.global_.offwidth = 3840
        self._mj_model.vis.global_.offheight = 2160

        self._mjx_model = mjx.put_model(self._mj_model)
        self._xml_path = f"{xml_path}#custom_box"

        self._post_init()
