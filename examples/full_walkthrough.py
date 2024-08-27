"""
================
Full Walkthrough
================
"""

from sunpy.map import Map

from smart.characterization.dB_dt import dB_dt, get_properties
from smart.differential_rotation import diff_rotation
from smart.IGM import indexed_grown_mask
from smart.map_processing import smart_prep

#####################################################
#

hmi_map = Map(
    "https://solmon.dias.ie/data/2024/06/06/HMI/fits/hmi.m_720s_nrt.20240606_230000_TAI.3.magnetogram.fits"
)
hmi_map_prev = Map(
    "https://solmon.dias.ie/data/2024/06/06/HMI/fits/hmi.m_720s_nrt.20240606_000000_TAI.3.magnetogram.fits"
)

threshold_map, cos_correction = smart_prep(hmi_map)
threshold_map_prev, cos_correction_prev = smart_prep(hmi_map_prev)

rotated_map = diff_rotation(hmi_map, hmi_map_prev)

sorted_labels = indexed_grown_mask(hmi_map, rotated_map)

dBdt, dt = dB_dt(hmi_map, hmi_map_prev)

properties = get_properties(hmi_map, dBdt, dt, sorted_labels)

for i in range(len(properties)):
    for prop, value in properties[i].items():
        print(prop, ":", value)
    print()
