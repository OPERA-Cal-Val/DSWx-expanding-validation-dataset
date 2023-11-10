# GIS imports 
import rasterio

# data ETL imports
import numpy as np

# misc imports
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom

# ML imports
from skimage.restoration import denoise_tv_bregman

def return_grn_indices(xml_file:str|Path)->list:
    """ 
    Return the indices of the (green, red, and NIR) channels of a 4 or 8 band Planet image
    """
    tree = ET.parse(xml_file)
    numbands = None
    for elem in tree.iter():
        if 'numBands' in elem.tag:
            numbands = int(elem.text)
        
    # we always want (green, red, nir) indices in the image
    if numbands == 4:
        band_idxs = [2, 3, 4] # BGRN image
    else:
        band_idxs = [4, 6, 8] # 8 band MS image
    
    return band_idxs

def return_reflectance_coeffs(xml_file:str|Path, band_idx:int|list):
    """
    Read XML file associated with a Planet image and return the TOA reflectance coefficients
    for specified band indices
    """
    assert isinstance(band_idx, (list, int)), "band_idx must be of type int or list"
    
    if isinstance(band_idx, int):
        band_idx = [band_idx]
    
    # parse XML metadata to obtain TOA reflectance coefficients
    xmldoc = minidom.parse(str(xml_file))
    nodes = xmldoc.getElementsByTagName("ps:bandSpecificMetadata")
    coeffs = {}
    for node in nodes:
        bn = node.getElementsByTagName("ps:bandNumber")[0].firstChild.data
        if bn in [str(x) for x in band_idx]:
            i = int(bn)
            value = node.getElementsByTagName("ps:reflectanceCoefficient")[0].firstChild.data
            coeffs[i] = float(value)
    
    return coeffs

def return_img_bands(img:str|Path, band_idx:int|list, denoising_weight=0)->np.ndarray:
    """ 
    Read a Planet file and return an numpy array containing data from specified bands. The image 
    will be band-wise denoised (using TV denoising) if a denoising weight is specified
    """
    if isinstance(band_idx, int):
        band_idx = list(band_idx)
        
    img_stack = []
        
    with rasterio.open(img) as ds:
        min_val = 1e-4

        for idx in band_idx:
            band_img = ds.read(idx)
            nodata_mask = (band_img == ds.profile['nodata'])
            band_img = band_img.astype(float)
            if denoising_weight:
                band_img = denoise_tv_bregman(band_img, weight=denoising_weight)
                band_img[band_img<0] = min_val
                band_img[nodata_mask]=ds.profile['nodata']

            img_stack.append(band_img)

    return np.stack(img_stack, axis=0)

def calc_ndwi(green, red, fill_value=0):
    with np.errstate(divide='ignore', invalid='ignore'):
        ndwi = (green - red)/(green + red)
    ndwi = np.where(np.isnan(ndwi), fill_value, ndwi)
    return ndwi

def calc_ndvi(red, nir, fill_value=0):
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (nir - red)/(nir + red)
    ndvi = np.where(np.isnan(ndvi), fill_value, ndvi)
    return ndvi