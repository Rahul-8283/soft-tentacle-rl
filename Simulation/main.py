import mujoco
import mujoco.viewer
import time
import numpy as np

model = mujoco.MjModel.from_xml_path("soft_gripper.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    for step in range(5000):

        # Use single actuator (index 0). Scale values per phase.
        # Phase 1: reach upward (bend forward)
        if step < 1000:
            data.ctrl[0] = 0.4

        # Phase 2: curl and wrap around object
        elif step < 2500:
            data.ctrl[0] = 0.9

        # Phase 3: grasp (hold tension)
        elif step < 4000:
            data.ctrl[0] = 1.0

        # Phase 4: retract
        else:
            data.ctrl[0] = -0.5

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.002)
