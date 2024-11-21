# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 10:54:04 2024

@author: Mico
"""

import numpy as np
from skimage import io
from skimage.filters import threshold_otsu
from skimage.segmentation import mark_boundaries
from skimage.transform import resize
from sklearn.decomposition import PCA


def match_images_sizes(img, label_img):
    h, w = img.shape[0:2]
    h_new, w_new = label_img.shape[0:2]

    output_shape = (h_new, w_new)
    if len(img.shape) == 3:
        output_shape = (h_new, w_new, img.shape[2])
    if (h != h_new) or (w != w_new):
        img = resize(img, output_shape, order=0)
    return img


def imshow_contours(ct, ct_multilabel=None):
    for slice_i in range(0, ct.shape[2]):
        img = ct[:, :, slice_i]
        if ct_multilabel is not None:
            label_img = ct_multilabel[:, :, slice_i]
            img = match_images_sizes(img, label_img)
            io.imshow(mark_boundaries(img, label_img))
        else:
            io.imshow(img)
        io.show()


def min_max_scale(data):
    min_v = data.min()
    max_v = data.max()
    range_v = max_v - min_v
    if range_v != 0:
        data = (data - min_v) / range_v
    return data


def pca_colorize(features, output_shape=None, remove_bg=False):
    n_samples = features.shape[0]
    n_components = 3
    if n_samples >= n_components:
        pca = PCA(n_components=n_components)
        pca.fit(features)
        rgb = pca.transform(features)
    else:
        rgb = np.repeat(np.expand_dims(np.ones(n_components), 0), n_samples, axis=0)
    rgb = min_max_scale(rgb)

    if output_shape:
        rgb = rgb.reshape(output_shape + (n_components,))

        if remove_bg:
            thresh = threshold_otsu(rgb[:, :, 0])
            rgb_mask = (rgb[:, :, 0] > thresh)*1
            rgb[:, :, 0] *= rgb_mask
            rgb[:, :, 1] *= rgb_mask
            rgb[:, :, 2] *= rgb_mask
            rgb = min_max_scale(rgb)

    return rgb


def visualize_features(img, features, mask):
    h, w, d = features.shape
    pca_medsam_embeddings = pca_colorize(features.reshape(h*w, d), output_shape=(h, w), remove_bg=False)

    imshow_contours(np.expand_dims(pca_medsam_embeddings, axis=2),
                    np.expand_dims(mask > 0, axis=-1))
    if img is not None:
        imshow_contours(np.expand_dims(img, axis=2),
                        np.expand_dims(mask > 0, axis=-1))

        img_crop = extract_roi(img, mask)
    pca_crop = extract_roi(pca_medsam_embeddings, mask)
    mask_crop = extract_roi(mask, mask)

    imshow_contours(np.expand_dims(pca_crop, axis=2),
                    np.expand_dims(mask_crop > 0, axis=-1))
    if img is not None:
        imshow_contours(np.expand_dims(img_crop, axis=2),
                        np.expand_dims(mask_crop > 0, axis=-1))


def crop_image(img, xmin, ymin, xmax, ymax):
    h, w = img.shape[0:2]
    ymin, ymax = [max(0, min(v, h)) for v in [ymin, ymax]]
    xmin, xmax = [max(0, min(v, w)) for v in [xmin, xmax]]
    cropped = img[ymin:ymax, xmin:xmax]
    return cropped


def extract_coords(mask, margin):
    indices = np.array(np.where(mask))
    ymin = np.min(indices[0, :]) - margin
    xmin = np.min(indices[1, :]) + margin
    ymax = np.max(indices[0, :]) - margin
    xmax = np.max(indices[1, :]) + margin

    h = max((ymax - ymin), margin)
    w = max((xmax - xmin), margin)
    xmax = xmin + w
    ymax = ymin + h
    return xmin, ymin, xmax, ymax


def extract_roi(img, mask, margin=1):
    xmin, ymin, xmax, ymax = extract_coords(mask, margin)
    if img.shape[0:2] != mask.shape[0:2]:
        h = img.shape[0] / mask.shape[0]
        w = img.shape[1] / mask.shape[1]
        xmin, ymin, xmax, ymax = [int(v) for v in [xmin*w, ymin*h, xmax*w, ymax*h]]
        h = max((ymax - ymin), margin)
        w = max((xmax - xmin), margin)
        xmax = xmin + w
        ymax = ymin + h
    return crop_image(img, xmin, ymin, xmax, ymax)


def hu_to_rgb_vectorized(hu_matrix):
    # colores RGB de cada tejido
    air_color = np.array([0, 0, 0])
    lung_color = np.array([194, 105, 82])
    fat_color = np.array([194, 166, 115])
    soft_tissue_color_lower = np.array([102, 0, 0])
    soft_tissue_color_upper = np.array([153, 0, 0])
    bone_color = np.array([255, 255, 255])

    rgb_matrix = np.zeros(hu_matrix.shape + (3,), dtype=int)

    # para interpolar los colores
    def interpolate_color_vectorized(color1, color2, hu_values, min_val, max_val):
        ratios = (hu_values - min_val) / (max_val - min_val)
        return np.array(color1) * (1 - ratios[..., None]) + np.array(color2) * ratios[..., None]

    # Condición 1: HU <= -1000 [aire]
    rgb_matrix[hu_matrix <= -1000] = air_color

    # Condición 2: -1000 < HU < -600 [aire, parenquima]
    mask = (hu_matrix > -1000) & (hu_matrix < -600)
    rgb_matrix[mask] = interpolate_color_vectorized(air_color,
                                                    lung_color,
                                                    hu_matrix[mask], -1000, -600)

    # Condición 3: -600 <= HU <= -400 [parenquima]
    rgb_matrix[(hu_matrix >= -600) & (hu_matrix <= -400)] = lung_color

    # Condición 4: -400 < HU < -100 [parenquima, grasa]
    mask = (hu_matrix > -400) & (hu_matrix < -100)
    rgb_matrix[mask] = interpolate_color_vectorized(lung_color,
                                                    fat_color,
                                                    hu_matrix[mask], -400, -100)

    # Condición 5: -100 <= HU <= -60 [grasa]
    rgb_matrix[(hu_matrix >= -100) & (hu_matrix <= -60)] = fat_color

    # Condición 6: -60 < HU < 40 [grasa, tejido blando]
    mask = (hu_matrix > -60) & (hu_matrix < 40)
    rgb_matrix[mask] = interpolate_color_vectorized(fat_color,
                                                    soft_tissue_color_lower,
                                                    hu_matrix[mask], -60, 40)

    # Condición 7: 40 <= HU <= 80 [tejido blando]
    mask = (hu_matrix >= 40) & (hu_matrix <= 80)
    rgb_matrix[mask] = interpolate_color_vectorized(soft_tissue_color_lower,
                                                    soft_tissue_color_upper,
                                                    hu_matrix[mask], 80, 400)

    # Condición 8: 80 < HU < 400 [tejido blando, huesos]
    mask = (hu_matrix > 80) & (hu_matrix < 400)
    rgb_matrix[mask] = interpolate_color_vectorized(soft_tissue_color_upper,
                                                    bone_color,
                                                    hu_matrix[mask], 80, 400)

    # Condición 9: HU >= 400 [huesos]
    rgb_matrix[hu_matrix >= 400] = bone_color

    return rgb_matrix.astype(np.uint8)
