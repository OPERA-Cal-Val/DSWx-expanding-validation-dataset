# data science imports
import numpy as np
from scipy.ndimage import find_objects
import scipy.ndimage.measurements as measurements
from skimage.restoration import denoise_tv_bregman
import pandas as pd

# misc imports
import os
import sys
from datetime import datetime
from typing import Tuple
import warnings
from collections import Counter

# gis imports
import geopandas as gpd
from rasterio.warp import Resampling, reproject
from affine import Affine
from rasterio.features import shapes

def _get_superpixel_means_band(label_array: np.array,
                               band: np.array) -> np.array:
    # Assume labels are 0, 1, 2, ..., n
    # scipy wants labels to begin at 1 and transforms to 1, 2, ..., n+1
    labels_ = label_array + 1
    labels_unique = np.unique(labels_)
    labels_unique = labels_unique[np.where(labels_unique>0)]
    means = measurements.mean(band, labels=labels_, index=labels_unique)
    return means.reshape((-1, 1))


def get_superpixel_means_as_features(label_array: np.array,
                                     img: np.array) -> np.array:
    if len(img.shape) == 2:
        measurements = _get_superpixel_means_band(label_array, img)
    elif len(img.shape) == 3:
        measurements = [_get_superpixel_means_band(label_array,
                                                   img[..., k])
                        for k in range(img.shape[2])]
        measurements = np.concatenate(measurements, axis=1)
    else:
        raise ValueError('img must be 2d or 3d array')
    return measurements


def _get_superpixel_stds_band(label_array: np.array,
                              band: np.array) -> np.array:
    # Assume labels are 0, 1, 2, ..., n
    # scipy wants labels to begin at 1 and transforms to 1, 2, ..., n+1
    labels_ = label_array + 1
    labels_unique = np.unique(labels_)
    labels_unique = labels_unique[np.where(labels_unique>0)]
    stddevs = measurements.standard_deviation(band,
                                            labels=labels_,
                                            index=labels_unique)
    return stddevs.reshape((-1, 1))

def get_superpixel_stds_as_features(label_array: np.array,
                                    img: np.array) -> np.array:
    if len(img.shape) == 2:
        measurements = _get_superpixel_stds_band(label_array,
                                                 img)
    elif len(img.shape) == 3:
        measurements = [_get_superpixel_stds_band(label_array,
                                                  img[..., k])
                        for k in range(img.shape[2])]
        measurements = np.concatenate(measurements, axis=1)
    else:
        raise ValueError('img must be 2d or 3d array')
    return measurements

def get_array_from_features(label_array: np.ndarray,
                            features: np.ndarray) -> np.ndarray:
    """
    Using p x q segmentation labels (2d) and feature array with dimension (m x
    n) where m is the number of unique labels and n is the number of features,
    obtain a p x q x m channel array in which each spatial segment is labeled
    according to n-features.
    See `find_objects` found
    [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.find_objects.html)
    for the crucial scipy function used.
    Parameters
    ----------
    label_array : np.array
        p x q integer array of labels corresponding to superpixels
    features : np.array
        m x n array of features - M corresponds to number of distinct items to
        be classified and N number of features for each item.
    Returns
    -------
    out : np.array
        p x q (x n) array where we drop the dimension if n == 1.
    Notes
    ------
    Inverse of get_features_from_array with fixed labels, namely if `f` are
    features and `l` labels, then:
        get_features_from_array(l, get_array_from_features(l, f)) == f
    And similarly, if `f_array` is an array of populated segments, then
        get_array_from_features(l, get_features_from_array(l, f)) == f
    """
    # Assume labels are 0, 1, 2, ..., n
    if len(features.shape) != 2:
        raise ValueError('features must be 2d array')
    elif features.shape[1] == 1:
        out = np.zeros(label_array.shape, dtype=features.dtype)
    else:
        m, n = label_array.shape
        out = np.zeros((m, n, features.shape[1]), dtype=features.dtype)

    labels_p1 = label_array + 1
    indices = find_objects(labels_p1)
    labels_unique = np.unique(labels_p1)
    # ensures that (number of features) == (number of unique superpixel labels)
    assert(len(labels_unique) == features.shape[0])
    for k, label in enumerate(labels_unique):
        indices_temp = indices[label-1]
        # if features is m x 1, then do not need extra dimension when indexing
        label_slice = labels_p1[indices_temp] == label
        if features.shape[1] == 1:
            out[indices_temp][label_slice] = features[k, 0]
        # if features is m x n with n > 1, then requires extra dimension when
        # indexing
        else:
            out[indices_temp + (np.s_[:], )][label_slice] = features[k, ...]
    return out


def denoise(img, weight=0.2, return_db=False):

    idx = np.where((np.isnan(img)) | (np.isinf(img)) | (img == 0))
    
    # prevent nan issues
    img[idx] = 1e-5
    
    # Convert to db and make noise additive
    img = 10 * np.log10(img)

    # # Fill in nodata areas with nearest neighbor
    # # Won't do anything if img is completely NaN/Inf
    # img_nn = interpolate_nn(img)

    # Use TV denoising
    # The weight parameter lambda = .2 worked well
    # Higher values mean less denoising and lower mean
    # image will appear smoother.
    img_tv = denoise_tv_bregman(img, weight)

    if return_db:
        return img_tv
    else:
        return 10**(img_tv / 10.)


def addImageCalc(filePaths, metaData, awsSession):
    """
    Created on Tue Jul 19 21:21:49 2022

    @author: mbonnema
    """

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    s3 = awsSession.resource('s3')
    s3_client = awsSession.client('s3')

    print('Creating geojson table')
    s3_keys = []
    for key in filePaths:
        if not os.path.isfile(filePaths[key]):
            sys.exit('File does not exist or path is incorrect: '+filePaths[key])

        s3_key_pending = 'pending/files/'+now+'_imagecalc/'+filePaths[key].split('/')[-1]
        s3_keys.append(s3_key_pending)
    bucket_name_staging = 'opera-calval-database-dswx-staging'
    metaData['bucket'] = bucket_name_staging
    metaData['s3_keys'] = ','.join(s3_keys)

    metaData['upload_date'] = now

    newRow = pd.DataFrame(metaData, index=[0])
    newRow = gpd.GeoDataFrame(newRow, geometry='geometry')

    print('Uploading geojson table')
    newRow_bytes = bytes(newRow.to_json(drop_id=True).encode('UTF-8'))
    s3object = s3.Object(bucket_name_staging, 'pending/'+now+'_imagecalc.geojson')
    s3object.put(Body=newRow_bytes)

    print('Uploading files')
    for key, s3_key in zip(filePaths, s3_keys):
        response = s3_client.upload_file(filePaths[key],
                                         bucket_name_staging,
                                         s3_key)
    print('staging complete')

    return 'pending/'+now+'_imagecalc.geojson'


def reproject_arr_to_match_profile(src_array: np.ndarray,
                                   src_profile: dict,
                                   ref_profile: dict,
                                   nodata: str = None,
                                   resampling='bilinear') \
                                           -> Tuple[np.ndarray, dict]:
    """
    Reprojects an array to match a reference profile providing the reprojected
    array and the new profile.  Simply a wrapper for rasterio.warp.reproject.
    Parameters
    ----------
    src_array : np.ndarray
        The source array to be reprojected.
    src_profile : dict
        The source profile of the `src_array`
    ref_profile : dict
        The profile that to reproject into.
    nodata : str
        The nodata value to be used in output profile. If None, the nodata from
        src_profile is used in the output profile.  See
        https://github.com/mapbox/rasterio/blob/master/rasterio/dtypes.py#L13-L24.
    resampling : str
        The type of resampling to use. See all the options:
        https://github.com/mapbox/rasterio/blob/08d6634212ab131ca2a2691054108d81caa86a09/rasterio/enums.py#L28-L40
    Returns
    -------
    Tuple[np.ndarray, dict]:
        Reprojected Arr, Reprojected Profile
    Notes
    -----
    src_array needs to be in gdal (i.e. BIP) format that is (# of channels) x
    (vertical dim.) x (horizontal dim).  Also, works with arrays of the form
    (vertical dim.) x (horizontal dim), but output will be: 1 x (vertical dim.)
    x (horizontal dim).
    """
    height, width = ref_profile['height'], ref_profile['width']
    crs = ref_profile['crs']
    transform = ref_profile['transform']

    reproject_profile = ref_profile.copy()

    nodata = nodata or src_profile['nodata']
    src_dtype = src_profile['dtype']
    count = src_profile['count']

    reproject_profile.update({'dtype': src_dtype,
                              'nodata': nodata,
                              'count': count})

    dst_array = np.zeros((count, height, width))

    resampling = Resampling[resampling]

    reproject(src_array,
              dst_array,
              src_transform=src_profile['transform'],
              src_crs=src_profile['crs'],
              dst_transform=transform,
              dst_crs=crs,
              dst_nodata=nodata,
              resampling=resampling)
    return dst_array.astype(src_dtype), reproject_profile


def resample_labels_into_percentage(X: np.ndarray,
                                    profile_src: dict,
                                    profile_dst: dict,
                                    class_label: int,
                                    minimum_nodata_percent_for_exclusion: float = .5) -> np.ndarray:
    """Using a given class label, determines percentage of label (excluding np.nans) in new CRS. The output is

    1. Float32 array with each pixel the percentage of label contained in the pixel using resampling `average`
    2. Mask is filled in with np.nan

    The percent calculation *excludes* nodata values. Can determine how to exclude nodata values with
    `minimum_nodata_percent_for_exclusion`.

    Parameters
    ----------
    X : np.ndarray
        Input Labels
    profile_src : dict
        Source profile metadata from rasterio
    profile_dst : dict
        Destination metadata. Relevant keys are transform, dtype (must be float), crs
        and size (width/height)
    class_label : int
        Which class label to resample
    minimum_nodata_percent_for_exclusion : float, optional
        If 1, only pixels with entire nodata will be turned into nodata. Otherwise,
        `mask >= minimum_nodata_percent_for_exclusion` determines slice, by default `.5`.
        A value of 0 (not accepted) means entire image will be masked.

    Returns
    -------
    np.ndarray
        The percent per pixel of the label under consideration.

    Raises
    ------
    RuntimeError
        1. If profile_dst dtype is not float
        2. If `minimum_nodata_percent_for_exclusion` is > 0 and <= 1
    """
    if ((minimum_nodata_percent_for_exclusion > 1) and
       (minimum_nodata_percent_for_exclusion <= 0)):
        raise RuntimeError('Minimum_nodata_percent_for_exclusion must be between 0 and 1')

    if 'float' not in profile_dst['dtype']:
        raise RuntimeError('dst dtype must be float')
    p = profile_src.copy()
    p['dtype'] = 'float32'
    p['nodata'] = np.nan

    X_float = X.astype(np.float32)
    if not np.isnan(profile_src['nodata']):
        nodata = profile_src['nodata']
        X_float[X_float == nodata] = np.nan

    X_true = ((X_float == class_label) & ~np.isnan(X_float)).astype(float)
    X_false = ((X_float != class_label) & ~np.isnan(X_float)).astype(float)
    mask = np.isnan(X_float).astype(float)

    X_true_r, p_perc = reproject_arr_to_match_profile(X_true,
                                                      p,
                                                      profile_dst,
                                                      resampling='average')
    X_true_r = X_true_r[0, ...]
    X_false_r, _ = reproject_arr_to_match_profile(X_false,
                                                  p,
                                                  profile_dst,
                                                  resampling='average')
    X_false_r = X_false_r[0, ...]

    mask_r, _ = reproject_arr_to_match_profile(mask,
                                               p,
                                               profile_dst,
                                               resampling='average')
    mask_r = mask_r[0, ...]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        X_percent = X_true_r / (X_true_r + X_false_r)

    X_percent[mask_r >= minimum_nodata_percent_for_exclusion] = np.nan
    return X_percent, p_perc


def reclassify_percentage_array_for_dswx(percentange_array: np.ndarray) -> np.ndarray:
    X_new_labels = np.full(percentange_array.shape, 255, dtype=np.uint8)

    mask = np.isnan(percentange_array)

    ind_w = ~mask & (percentange_array == 1)
    X_new_labels[ind_w] = 1

    ind_pw = ~mask & (percentange_array >= .5) & (percentange_array < 1)
    X_new_labels[ind_pw] = 2

    ind_nw = ~mask & (percentange_array < .5)
    X_new_labels[ind_nw] = 0

    X_new_labels[mask] = 255
    return X_new_labels


def get_geopandas_features_from_array(arr: np.ndarray,
                                      transform: Affine,
                                      label_name: str = 'label',
                                      mask: np.ndarray = None,
                                      connectivity: int = 4) -> list:
    """
    Obtains a list of geopandas features in which contigious integers are
    grouped as polygons for use as:
        df =  gpd.GeoDataFrame.from_features(geo_features)
    Parameters
    ----------
    arr : np.ndarray
        The array of integers to group into contiguous polygons. Note some
        labels that are connected through diagonals May be separated depending
        on connectivity.
    transform : Affine
        Rasterio transform related to arr
    label_name : str
        The label name used for each different polygonal feature, default is
        `label`.
    mask : np.ndarray
        Nodata mask in which true values indicate where nodata is located.
    connectivity : int
        4- or 8- connectivity of the polygonal features.  See rasterio:
        https://rasterio.readthedocs.io/en/latest/api/rasterio.features.html#rasterio.features.shapes
        And see: https://en.wikipedia.org/wiki/Pixel_connectivity
    Returns
    -------
    list:
        List of features to use for constructing geopandas dataframe with
        gpd.GeoDataFrame.from_features
    """
    # see rasterio.features.shapes - needs all false values to be no data areas
    if mask is None:
        mask = np.zeros(arr.shape, dtype=bool)
    feature_list = list(shapes(arr,
                               mask=~mask,
                               transform=transform,
                               connectivity=connectivity))
    geo_features = list({'properties': {label_name: (value)},
                         'geometry': geometry}
                        for i, (geometry, value) in enumerate(feature_list))
    return geo_features

def get_segment_sizes(segments: np.ndarray)->np.array:
    """
    For a segmented image containing unique labels, return the size of each segment
    The returned array is sorted by segment number, and only non-negative segment numbers are considered
    This allows for no-data regions to be given a negative value and be ignored
    """
    return np.array([v for (k, v) in sorted(Counter(segments.flatten()).items()) if k>=0]).reshape((-1, 1))
