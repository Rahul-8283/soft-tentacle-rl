import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path("model.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:

    for step in range(4000):

        # Phase 1: approach
        if step < 1000:
            data.ctrl[0] = 0.2

        # Phase 2: wrap
        elif step < 2500:
            data.ctrl[0] = 0.6

        # Phase 3: grasp (hold tension)
        else:
            data.ctrl[0] = 0.9

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.002)
