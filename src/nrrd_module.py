import nrrd
from skimage.transform import resize


DEFAULT_ROI_ARG = "DEFAULT_NONE"

def get_roi_sphere(file_name):
    pet_folder_path = "../../../shared_data/NSCLC_Radiogenomics/images/" + file_name + "/pet/"

    mask_file_name = pet_folder_path + file_name + "_pet_segmentation.nrrd"
    try:
        mask_data, mask_header = nrrd.read(mask_file_name)
    except FileNotFoundError:
        print('No file named: ', mask_file_name)
        mask_file_name = file_name + "_pet_liver.nrrd"
        return None, None

    image_file_name = file_name + "_pet_image.nrrd"
    try:
        image_data, image_header = nrrd.read(pet_folder_path + image_file_name)
    except FileNotFoundError:
        print('No file named: ', image_file_name)
        return None, None

    # data = image_data * mask_data
    # final_data = []
    # for i in range(data.shape[1]):
    #     current_slice = data[:,:,i]
    #     if np.sum(current_slice) > 0:
    #         final_data.append(current_slice)

    return image_data, mask_data


def upscale_cropped_roi_to_liver_cropped_size(roi_img, final_x, final_y):
    target_shape = (final_x, final_y, 256)
    upscaled_array = resize(roi_img, target_shape, order=1, preserve_range=True)

    return upscaled_array