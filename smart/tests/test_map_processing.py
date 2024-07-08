import numpy as np
import pytest

import astropy.units as u

from sunpy.map import Map, all_coordinates_from_map, coordinate_is_on_solar_disk
from sunpy.map.mapbase import GenericMap


def mag_sample():
    return "https://solmon.dias.ie/data/2024/06/06/HMI/fits/hmi.m_720s_nrt.20240606_230000_TAI.3.magnetogram.fits"


def mag_map_sample():
    from smart.map_processing import map_processing

    return map_processing(
        "https://solmon.dias.ie/data/2024/06/06/HMI/fits/hmi.m_720s_nrt.20240606_230000_TAI.3.magnetogram.fits"
    )


def create_fake_map(value, shape=((4098, 4098))):
    fake_data = np.ones(shape) * value
    return Map(fake_data, mag_map_sample().meta)


@pytest.fixture
def test_map_processing(file):
    from smart.map_processing import map_processing

    processed_map = map_processing(file)

    assert isinstance(processed_map, GenericMap), "Result is not a SunPy Map."

    coordinates = all_coordinates_from_map(processed_map)
    on_solar_disk = coordinate_is_on_solar_disk(coordinates)
    assert np.all(np.isnan(processed_map.data[~on_solar_disk])), "Off-disk NaN values not set correctly."


@pytest.fixture
def test_get_cosine_correction_shape(im_map):
    from smart.map_processing import get_cosine_correction

    cos_cor, d_angular, off_limb = get_cosine_correction(im_map)
    assert cos_cor.shape == im_map.data.shape, "cos_cor shape != im_map.data.shape"
    assert d_angular.shape == im_map.data.shape, "d_angular shape != im_map.data.shape"
    assert off_limb.shape == im_map.data.shape, "off_limb shape != im_map.data.shape"


@pytest.fixture
def test_get_cosine_correction_limits(im_map):
    from smart.map_processing import get_cosine_correction

    cos_cor, d_angular, off_limb = get_cosine_correction(im_map)

    edge = 0.99
    x, y = np.meshgrid(*[np.arange(v.value) for v in im_map.dimensions]) * u.pixel
    hp_coords = im_map.pixel_to_world(x, y)
    xx = hp_coords.Tx.to(u.arcsec)
    yy = hp_coords.Ty.to(u.arcsec)
    d_radial = np.sqrt(xx**2 + yy**2)

    off_disk = d_radial >= (im_map.rsun_obs * edge)
    on_disk = d_radial < (im_map.rsun_obs * edge)

    assert np.all(cos_cor >= 0), "cos_cor lower limits incorrect"
    assert np.all(cos_cor <= 1 / np.cos(np.arcsin(edge))), "cos_cor upper limits incorrect"

    assert np.all(d_angular >= np.arcsin(-1) * u.rad), "d_angular lower limits incorrect"
    assert np.all(d_angular <= np.arcsin(1) * u.rad), "d_angular upper limist incorrect"

    assert np.all(off_limb[off_disk] == 0), "not all off_disk values = 0"
    assert np.all(off_limb[on_disk] == 1), "not all on_disk values = 1"


@pytest.fixture
def test_cosine_correction():
    from smart.map_processing import cosine_correction

    fake_data = np.ones((4098, 4098))
    fake_map = Map(fake_data, mag_map_sample().meta)
    fake_cosmap = fake_data * 0.5

    test_corrected_data, fake_cosmap = cosine_correction(fake_map, fake_cosmap)
    assert np.all(test_corrected_data == 0.5), "corrected_data not being correctly calculated"
    assert test_corrected_data.shape == fake_map.data.shape, "output data shape != original data shape"


"""
@pytest.fixture
def test_STL():
    from smart.map_processing import STL

    fake_map = create_fake_map(3)
    smooth_map, filtered_labels, mask_sizes = STL(
        fake_map, thresh=2, min_size=len(fake_map.data.flatten() + 1)
    )

    num_true = np.count_nonzero(filtered_labels == True)

    assert isinstance(smooth_map, type(fake_map)), "smooth_map no longer of type Map"
    assert np.all(smooth_map.data == 0), "smooth_map.data not behaving as expected"

    assert filtered_labels.shape == fake_map.data.shape, "filtered_labels.shape != map.data.shape"
    assert num_true == 0, "filtered_labels detected despite threshold"

    assert mask_sizes.shape[0] > 0, "no mask_sizes"
    assert np.sum(mask_sizes) == 0, "no regions should have met the min_size requirement"
"""
