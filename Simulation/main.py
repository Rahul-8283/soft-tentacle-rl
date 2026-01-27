import mujoco
import mujoco.viewer
import time
import numpy as np

model = mujoco.MjModel.from_xml_path("soft_gripper.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    for step in range(3000):

        # Phase 1: approach
        if step < 800:
            data.ctrl[0] = 0.2

        # Phase 2: wrapping
        elif step < 1800:
            data.ctrl[0] = 0.6

        # Phase 3: grasp (hold tension)
        else:
            data.ctrl[0] = 0.9

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.002)
