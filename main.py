from source.utils import *

contour_5_403 = 'data/segmentacion_displasias/FCD005_MR_1/AIM_20220301_112949_176_S403/RTSTRUCT/AIM_20220301_112949_176_S403.dcm'
contour_5_902 = 'data/segmentacion_displasias/FCD005_MR_1/AIM_20220301_113047_946_S902/RTSTRUCT/AIM_20220301_113047_946_S902.dcm'

contour_7_801 = 'data/segmentacion_displasias/FCD007_MR_1/AIM_20220301_114502_449_S801/RTSTRUCT/AIM_20220301_114502_449_S801.dcm'
contour_7_301 = 'data/segmentacion_displasias/FCD007_MR_1/AIM_20220301_114433_614_S301/RTSTRUCT/AIM_20220301_114433_614_S301.dcm'

contour_8_502 = 'data/segmentacion_displasias/FCD008_MR_1/AIM_20220301_115436_329_S502/RTSTRUCT/AIM_20220301_115436_329_S502.dcm'

head_5_403 = './data/segmentacion_displasias/FCD005_MR_1/403/DICOM'
head_5_902 = './data/segmentacion_displasias/FCD005_MR_1/902/DICOM'
head_7_801 = './data/segmentacion_displasias/FCD007_MR_1/801/DICOM'
head_7_301 = './data/segmentacion_displasias/FCD007_MR_1/301/DICOM'
head_8_502 = './data/segmentacion_displasias/FCD008_MR_1/502/DICOM'

# Series Number = par'ametro dicom que idenrifica la serie

#plot_every_dcm_singleframe('./segmentacion_displasias/segmentacion_displasias/FCD007_MR_1/801/DICOM/')

contourPixels, slice_numbers, ref_SOP, num_seqs = get_contour_data(contour_7_801)

print(f'Tamano SOP ids: {len(ref_SOP)}\n')

#plot_slice(mri_img[10,:,:],mri_img[15,:,:],mri_img[20,:,:])

for seq in range(num_seqs):
    mri_img, mri_pixel_spacing, mri_image_position_patient, mri_dims = get_image_data(head_7_801, slice_numbers[seq], ref_SOP[seq])

    contour = coor2pix(contourPixels[seq], mri_image_position_patient, mri_pixel_spacing)
    coord = np.array(contour).reshape((len(contour[:])//2, 2))

    plot_dcm_contours(mri_img, slice_numbers[seq], coord)
