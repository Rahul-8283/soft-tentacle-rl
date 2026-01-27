import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
import math

# --- 1. Design Function: Reversed for Broad Base Fixed ---
def generate_spirob_design(a=0.05, b=0.2, num_segments=40, max_theta=2.5*np.pi):
    """
    Generates the spiral from OUTSIDE (large) to INSIDE (small).
    """
    # CHANGE 1: We generate angles from Max down to 0 (Reversed order)
    thetas = np.linspace(max_theta, 0, num_segments + 1)
    
    # Calculate radius and coordinates as before
    r = a * np.exp(b * thetas)
    x = r * np.cos(thetas)
    y = r * np.sin(thetas)
    
    segments = []
    
    for i in range(num_segments):
        # The vector now points INWARDS (from large r to small r)
        p1 = np.array([x[i], y[i]])
        p2 = np.array([x[i+1], y[i+1]])
        
        length = np.linalg.norm(p2 - p1)
        abs_angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        
        segments.append({
            "index": i,
            "length": length,
            "abs_angle": abs_angle,
            # Width scales with length, so it naturally gets thinner
            "width": length * 0.6 
        })
        
    return segments

# --- 2. XML Generator (Same Logic, New Data) ---
def write_mujoco_xml(segments, filename="spirob_reversed.xml"):
    mujoco = ET.Element('mujoco', model="spirob_reversed")
    
    # Setup
    ET.SubElement(mujoco, 'option', timestep="0.002", gravity="0 0 -9.81")
    
    # Assets
    asset = ET.SubElement(mujoco, 'asset')
    ET.SubElement(asset, 'texture', name="grid", type="2d", builtin="checker", rgb1=".1 .2 .3", rgb2=".2 .3 .4", width="300", height="300")
    ET.SubElement(asset, 'material', name="grid", texture="grid", texrepeat="1 1", reflectance="0.2")
    
    # Defaults
    default = ET.SubElement(mujoco, 'default')
    ET.SubElement(default, 'geom', friction="0.5", solimp="0.9 0.95 0.001", solref="0.02 1")
    # Increased stiffness slightly to hold the heavier base structure up
    ET.SubElement(default, 'joint', type="hinge", axis="0 0 1", stiffness="2.0", damping="0.2", limited="true", range="-30 30")

    # World Body
    worldbody = ET.SubElement(mujoco, 'worldbody')
    ET.SubElement(worldbody, 'geom', name="floor", type="plane", size="2 2 0.1", material="grid")
    ET.SubElement(worldbody, 'light', diffuse=".5 .5 .5", pos="0 0 3", dir="0 0 -1")

    # --- Anchor Point ---
    # We lift it up higher (0.2) because the large outer loop might hit the floor otherwise
    current_parent = ET.SubElement(worldbody, 'body', name="base", pos="0 0 0.2")
    # No geom here, so it's invisible

    cable1_sites = []
    cable2_sites = []

    prev_angle = segments[0]['abs_angle']

    for seg in segments:
        i = seg['index']
        L = seg['length']
        W = seg['width']
        
        rel_angle_rad = seg['abs_angle'] - prev_angle
        rel_angle_deg = np.degrees(rel_angle_rad)
        prev_angle = seg['abs_angle']

        # Determine position relative to parent
        pos_str = f"{segments[i-1]['length'] if i > 0 else 0} 0 0"
        
        # Create Body
        body = ET.SubElement(current_parent, 'body', name=f"seg_{i}", pos=pos_str, euler=f"0 0 {rel_angle_deg}")
        ET.SubElement(body, 'joint', name=f"j_{i}")
        ET.SubElement(body, 'geom', type="box", size=f"{L/2} {W} 0.005", pos=f"{L/2} 0 0", rgba="0.2 0.6 0.8 1")
        
        # Sites
        offset_y = W * 1.2
        site1_name = f"s1_{i}"
        site2_name = f"s2_{i}"
        
        ET.SubElement(body, 'site', name=site1_name, pos=f"{L/2} {offset_y} 0", size="0.002", rgba="1 1 0 1")
        ET.SubElement(body, 'site', name=site2_name, pos=f"{L/2} {-offset_y} 0", size="0.002", rgba="1 1 0 1")
        
        cable1_sites.append(site1_name)
        cable2_sites.append(site2_name)
        current_parent = body

    # --- Tendons ---
    tendon = ET.SubElement(mujoco, 'tendon')
    spatial1 = ET.SubElement(tendon, 'spatial', name="cable_left", width="0.001", rgba="1 0 0 1")
    for site in cable1_sites:
        ET.SubElement(spatial1, 'site', site=site)

    spatial2 = ET.SubElement(tendon, 'spatial', name="cable_right", width="0.001", rgba="0 1 0 1")
    for site in cable2_sites:
        ET.SubElement(spatial2, 'site', site=site)

    # --- Actuators ---
    actuator = ET.SubElement(mujoco, 'actuator')
    # Because we reversed the direction, Left/Right roles might swap depending on spiral direction.
    # ctrlrange is negative because we pull (shorten) the tendon
    ET.SubElement(actuator, 'position', name="act_left", tendon="cable_left", kp="20", kv="1", gear="1", ctrlrange="-0.3 0")
    ET.SubElement(actuator, 'position', name="act_right", tendon="cable_right", kp="20", kv="1", gear="1", ctrlrange="-0.3 0")

    # Write File
    xml_str = minidom.parseString(ET.tostring(mujoco)).toprettyxml(indent="  ")
    with open(filename, "w") as f:
        f.write(xml_str)
    print(f"Successfully generated {filename}")

if __name__ == "__main__":
    design_data = generate_spirob_design()
    write_mujoco_xml(design_data)