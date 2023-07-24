# In training RFs to expand our validation dataset, we've come across two situations that are not captured by the approach thus far - 
# 1. The spectral characteristics of water within the validation chip is not representative of water surfaces present in the broader Planet image
# 2. A validation chip contains no water classification, while the broader Planet image contains some water. Since the RF trained on this data is not given any OSW examples, it will classify the entire Planet image as "not-water"
# 
# To mitigate this, we will train a random forest on validation data from multiple sites

import rasterio

import pandas as pd
from pathlib import Path
import numpy as np
from collections import defaultdict

# ML imports
from sklearn.ensemble import RandomForestClassifier
from skimage.segmentation import felzenszwalb
from tools import get_superpixel_stds_as_features, get_superpixel_means_as_features, get_array_from_features, reproject_arr_to_match_profile
from sklearn.model_selection import train_test_split
import joblib
from joblib import dump

# local imports
from rf_funcs import calc_ndwi, calc_ndvi, return_grn_indices, return_img_bands, return_reflectance_coeffs

from tqdm import tqdm

# for repeatability
np.random.seed(42)

# Set to True to re-train the model
RETRAIN_MODEL=True
SUBSET_TRAINING=False # Set to True to use only a limited number of chips from each strata
SUBSET_NUMBER=4 # Number of chips to use from each strata

# Read the validation database
data_path = Path('../data/')
val_chips_db = data_path / 'validation_table.csv'
val_df = pd.read_csv(val_chips_db)

site_names = list(val_df['site_name'])
planet_ids = list(val_df['planet_id'])

# Extract planet IDs and associated strata
site_names_stratified = defaultdict(list)
for sn, planet_id in zip(site_names, planet_ids):
    site_names_stratified[sn[:2]].append(planet_id)

print(site_names_stratified.keys())

training_sites = []

# We can either use 4 chips from each strata, or ALL chips from each strata
training_sites = []
for key in site_names_stratified.keys():
    if SUBSET_TRAINING:
        training_sites.extend(np.random.choice(site_names_stratified[key], 4))
    else:
        training_sites.extend(site_names_stratified[key])

print("Training sites: ", training_sites)

# We have the name of the planet ids. For each, do the following - 
# 1. Read the cropped planet image and the corresponding validation labels
# 2. Generate superpixels and calculate mean and std dev.
# 3. Append to list
# 4. Train and save model
# 5. Apply model to broader Planet images

# Re-train the model
if RETRAIN_MODEL:
    X, class_features = None, None
    for idx, site in enumerate(training_sites):
        print(f"Currently processing site # {idx}")

        current_img_path = data_path / site
        cropped_img_path = data_path / 'planet_images_cropped' / site
        
        xml_file = list(current_img_path.glob('*.xml'))[0]
        chip = list(cropped_img_path.glob(f'cropped_{site}*.tif'))[0]
        classification = list(cropped_img_path.glob(f'classification_*.tif'))[0] 

        band_idxs = return_grn_indices(xml_file)
        coeffs = return_reflectance_coeffs(xml_file, band_idxs)
        chip_img = return_img_bands(chip, band_idxs, denoising_weight=None)

        with rasterio.open(chip) as src_ds:
            ref_profile = src_ds.profile

        green = chip_img[0]*coeffs[band_idxs[0]]
        red = chip_img[1]*coeffs[band_idxs[1]]
        nir = chip_img[2]*coeffs[band_idxs[2]]

        with rasterio.open(classification) as src_ds:
            cl = src_ds.read(1)
            cl_profile = src_ds.profile

        # some classification extents are not the same as the corresponding planet chip extent
        # if they are not the same, reproject the validation data to match the profile of the planet data
        if ((ref_profile['transform'] != cl_profile['transform']) | 
            (ref_profile['width'] != cl_profile['width']) | 
            (ref_profile['height'] != cl_profile['height'])):

            cl, _ = reproject_arr_to_match_profile(cl, cl_profile, ref_profile)
            cl = np.squeeze(cl)

        ndwi = calc_ndwi(green, nir)
        ndvi = calc_ndvi(red, nir)

        # segment image using green, nir, and NDWI channels
        img_stack = np.stack([green, nir, ndwi], axis=-1)
        segments = felzenszwalb(img_stack, sigma=0, min_size=10)

        # create training data that includes other channels as well
        img_stack = np.stack([red, nir, green, ndwi, ndvi], axis=-1)     
        std_features = get_superpixel_stds_as_features(segments, img_stack)
        mean_features = get_superpixel_means_as_features(segments, img_stack)

        if X is None:
            X = np.concatenate([mean_features, std_features], axis = 1)
        else:
            X_temp = np.concatenate([mean_features, std_features], axis = 1)
            X = np.concatenate([X, X_temp], axis=0)

        # We have superpixels, we now need to map each of the segments to the associated label
        # A 0 value indicates no label for the segment
        
        class_features_temp = np.zeros((mean_features.shape[0], 1))
        for class_id in [0, 1]:
            # Get all superpixel labels with particular id
            superpixel_labels_for_class = np.unique(segments[class_id == cl])
            # Label those superpixels with approrpriate class
            class_features_temp[superpixel_labels_for_class] = class_id

        if class_features is None:
            class_features = class_features_temp
        else:
            class_features = np.concatenate([class_features, class_features_temp], axis=0)

    print("Beginning model training")
    # Define an RF to be trained. setting n_jobs = -1 uses all available processors
    rf = RandomForestClassifier(n_estimators=300, class_weight='balanced', oob_score=True, random_state=0, n_jobs=-1)

    # train model on all of the available data
    rf.fit(X, class_features.ravel())

    rf_model_folder = data_path / 'trained_model' / 'rf_model'
    rf_model_folder.mkdir(exist_ok=True, parents=True)
    model_path = rf_model_folder/"rf_model_alldata.joblib"

    # save for later use
    dump(rf, model_path)

# Let's make inferences on the broader planet images
def generate_inference_helper(rf, img:str|Path, xml_file:str|Path):
    band_idxs = return_grn_indices(xml_file)
    coeffs = return_reflectance_coeffs(xml_file, band_idxs)
    
    full_img = return_img_bands(img, band_idxs, denoising_weight=None)

    green = full_img[0]*coeffs[band_idxs[0]]
    red = full_img[1]*coeffs[band_idxs[1]]
    nir = full_img[2]*coeffs[band_idxs[2]]

    ndwi = calc_ndwi(green, nir)
    ndvi = calc_ndvi(red, nir)

    img_stack = np.stack([green, nir, ndwi], axis=-1)
    segments = felzenszwalb(img_stack, sigma=0, min_size=10)

    # for inference we include other channels as well
    img_stack = np.stack([red, nir, green, ndwi, ndvi], axis=-1)
    std_features = get_superpixel_stds_as_features(segments, img_stack)
    mean_features = get_superpixel_means_as_features(segments, img_stack)

    X = np.concatenate([mean_features, std_features], axis = 1)
    y = rf.predict(X)

    return get_array_from_features(segments, np.expand_dims(y, axis=1))

def generate_inference(planet_id):
    """ 
    This function takes in a planet_id and generates inferences for the overlapping planet image
    """
    data_path = Path('../data')
    
    current_img_path = data_path / planet_id
    cropped_img_path = data_path / 'planet_images_cropped' / planet_id
    xml_file = list(current_img_path.glob('*.xml'))[0]
    classification = list(cropped_img_path.glob(f'classification_*.tif'))[0]

    img = list(current_img_path.glob(f'{planet_id}*.tif'))[0]

    inference = generate_inference_helper(rf, img, xml_file)

    # use planet image to mask out regions of no data in the model inference
    with rasterio.open(img) as src_ds:
        nodata_mask = np.where(src_ds.read(1) == src_ds.profile['nodata'], 1, 0)
        inference[nodata_mask==1] = 255
        profile_copy = src_ds.profile
        profile_copy.update({'count':1, 'dtype':np.uint8, 'nodata':255})

        # write out model inference
        with rasterio.open(f"{classification.parent}/new_full_img_rf_classification_{classification.name}", 'w', **profile_copy) as dst_ds:
            dst_ds.write(inference.astype(np.uint8).reshape(1, *inference.shape))

    print(f"Completed inference for planet id {planet_id}")

# Make inferences
_ = list(map(generate_inference, tqdm(planet_ids)))

print("Model feature importances: ", rf.feature_importances_)