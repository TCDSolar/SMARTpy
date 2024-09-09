"""
================
Full Walkthrough
================

Here we see the functions needed to quickly extract properties from a magnetogram.
"""

from sunpy.map import Map

from smart.calculate_properties import dB_dt, get_properties, smart_indentify_and_characterize
from smart.differential_rotation import diff_rotation
from smart.indexed_grown_mask import index_and_grow_mask
from smart.processing import smart_prep

#####################################################
#

hmi_map = Map(
    "http://jsoc.stanford.edu/data/hmi/fits/2024/06/06/hmi.M_720s.20240606_230000_TAI.fits"
    # "https://solmon.dias.ie/data/2024/06/06/HMI/fits/hmi.m_720s_nrt.20240606_230000_TAI.3.magnetogram.fits"
)
hmi_map_prev = Map(
    "http://jsoc.stanford.edu/data/hmi/fits/2024/06/06/hmi.M_720s.20240606_000000_TAI.fits"
    # "https://solmon.dias.ie/data/2024/06/06/HMI/fits/hmi.m_720s_nrt.20240606_000000_TAI.3.magnetogram.fits"
)

threshold_map, cos_correction = smart_prep(hmi_map)
threshold_map_prev, cos_correction_prev = smart_prep(hmi_map_prev)

rotated_map = diff_rotation(hmi_map, hmi_map_prev)

sorted_labels = index_and_grow_mask(hmi_map, rotated_map)

dBdt, dt = dB_dt(hmi_map, hmi_map_prev)

properties = get_properties(hmi_map, dBdt, dt, sorted_labels)

for i in range(len(properties)):
    for prop, value in properties[i].items():
        print(prop, ":", value)
    print()

#####################################################
# We can also use the `~smart.calculate_properties.smart_identify_and_characterize` function to quickly and easily get these properties

smart_properties = smart_indentify_and_characterize(hmi_map, hmi_map_prev)

for i in range(len(smart_properties)):
    for prop, value in smart_properties[i].items():
        print(prop, ":", value)
    print()
