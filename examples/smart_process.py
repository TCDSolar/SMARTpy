"""
=========================
Smart processing example.
=========================


"""

from matplotlib import pyplot as plt

from sunpy.map import Map

from smart.characterization.dB_dt import cosine_weighted_area_map, dB_dt, extract_features, get_properties
from smart.differential_rotation import diff_rotation
from smart.IGM import indexed_grown_mask, plot_indexed_grown_mask
from smart.map_processing import cosine_correction, get_cosine_correction, map_threshold, smooth_los_threshold

#####################################################
#
# More text
# ---------
#
# .. note::
#     More text something about imports
#
# hmi_map = Map("http://jsoc.stanford.edu/data/hmi/fits/2024/06/06/hmi.M_720s.20240606_230000_TAI.fits")
hmi_map = Map(
    "https://solmon.dias.ie/data/2024/06/06/HMI/fits/hmi.m_720s_nrt.20240606_230000_TAI.3.magnetogram.fits"
)
hmi_map.plot()

#####################################################
#

thresholded_map = map_threshold(hmi_map)
thresholded_map.plot()

#####################################################
#

smooth_map, filtered_labels, mask_sizes = smooth_los_threshold(thresholded_map)
cos_correction, *_ = get_cosine_correction(smooth_map)
corrected_data = cosine_correction(smooth_map, cos_correction)
plt.imshow(corrected_data.value)

#####################################################
#

# hmi_map_prev = Map("http://jsoc.stanford.edu/data/hmi/fits/2024/06/05/hmi.M_720s.20240605_230000_TAI.fits")
hmi_map_prev = Map(
    "https://solmon.dias.ie/data/2024/06/06/HMI/fits/hmi.m_720s_nrt.20240606_000000_TAI.3.magnetogram.fits"
)
thresholded_map_prev = map_threshold(hmi_map_prev)
smooth_map_prev, filtered_labels_prev, mask_sizes_prev = smooth_los_threshold(thresholded_map_prev)
cos_correction_prev, *_ = get_cosine_correction(smooth_map_prev)
corrected_data_prev = cosine_correction(smooth_map_prev, cos_correction_prev)

#####################################################
#

rotated_map = diff_rotation(hmi_map, hmi_map_prev)
diff_map = Map((hmi_map.data - rotated_map.data, rotated_map.meta))
fig, ax = plt.subplot_mosaic(
    [["cur", "prev", "diff"]],
    figsize=(9, 6),
    per_subplot_kw={
        "cur": {"projection": hmi_map},
        "prev": {"projection": hmi_map_prev},
        "diff": {"projection": rotated_map},
    },
)
hmi_map.plot(axes=ax["cur"])
hmi_map_prev.plot(axes=ax["prev"])
diff_map.plot(axes=ax["diff"])

#####################################################
#

sorted_labels = indexed_grown_mask(hmi_map, rotated_map)
plot_indexed_grown_mask(hmi_map, sorted_labels)

#####################################################
#

feature_masks = extract_features(sorted_labels)
region1_feature_mask = feature_masks[:1]
region1_area, region1_area_map = cosine_weighted_area_map(hmi_map, region1_feature_mask)
region1_area

#####################################################
#

dBdt, dt = dB_dt(hmi_map, hmi_map_prev)
dBdt.plot()

#####################################################
#

properties = get_properties(hmi_map, dBdt, dt, sorted_labels)
properties
