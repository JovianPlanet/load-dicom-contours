import math
import os
import numpy as np
import matplotlib.pyplot as plt
from pydicom import dcmread, read_file
from pydicom.data import get_testdata_file
import dicom_contour.contour as ctr
from natsort import natsorted

def plot_slice(t1, t1_sim, flair_sim):
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(hspace=1 ,wspace=1)

    ax1 = fig.add_subplot(1, 3, 1)
    #ax1.title.set_text('imagen')
    ax1.axis("off")
    ax1.imshow(t1, cmap="gray")

    ax2 = fig.add_subplot(1, 3, 2)
    #ax2.title.set_text('mask')
    ax2.axis("off")
    ax2.imshow(t1_sim, cmap="gray")

    ax3 = fig.add_subplot(1, 3, 3)
    #ax3.title.set_text('prediccion')
    ax3.axis("off")
    ax3.imshow(flair_sim, cmap="gray")
    #plt.savefig(save_path, dpi=300)

    plt.show()

def plot_every_dcm_singleframe(path):
    dicoms = next(os.walk(path))[2] # [2]: lists files; [1]: lists subdirectories; [0]: ?
    for dicom in natsorted(dicoms)[72:78]:
        img = dcmread(os.path.join(path, dicom)).pixel_array
        fig = plt.figure(figsize=(16, 16))
        fig.subplots_adjust(hspace=1, wspace=1)

        ax1 = fig.add_subplot(1, 1, 1)
        ax1.title.set_text(dicom)
        ax1.axis("off")
        ax1.imshow(img[:,:], cmap="gray")
        plt.show()

def plot_dcm_contours(dcm, slice_, contour):
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(hspace=1, wspace=1)

    ax1 = fig.add_subplot(1, 1, 1)
    #ax1.title.set_text('imagen')
    ax1.axis("off")
    if len(dcm.shape) == 3:
        ax1.imshow(dcm[slice_-1,:,:], cmap="gray")
    else:
        ax1.imshow(dcm[:,:], cmap="gray")

    #plt.scatter(coord[:,0], coord[:,1], s=2, alpha=0.5, color='red')
    plt.plot(contour[:,0], contour[:,1], color='red') #(coord[:,0]+10, coord[:,1]-5)
    plt.show()

'''
Convierte de coordenadas dicom (reales) a pixeles de numpy
Basada en: https://stackoverflow.com/questions/55154635/how-to-convert-dicom-rt-structure-contour-data-into-image-coordinate
'''
def coor2pix(data, origin, spacing):
    points = []
    for value in data:
        x = value[0]
        y = value[1]
        X = int((float(x) - float(origin[0])) / float(spacing[0]))
        Y = int((float(y) - float(origin[1])) / float(spacing[1]))
        points.append(X)
        points.append(Y)
    points.append(points[0]) # Se agregan al final las coordenadas de primer punto para cerrar la region de interes
    points.append(points[1])
    #print(points)
    return points

'''
Load dicom contour data (Cargar las coordenadas de la ROI delineada por el experto)
'''
def get_contour_data(contour_path):
    ds_roi = dcmread(contour_path)
    # seq[:] == ds_contour
    # seq = ds_roi[0x3006, 0x0039][0][0x3006, 0x0040][0][0x3006, 0x0050]

    #print(f'SOP Instance UID is: {ds_roi.SOPInstanceUID}\n')
    #print(f'Contour sequence is: {ds_roi.ROIContourSequence[0].ContourSequence}\n')

    num_seqs = len(ds_roi.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].ContourImageSequence)
    print(f'Num sequences is: {num_seqs}\n')

    contours = []
    slices = []
    SOP_IDs = []
    for seq in range(num_seqs):

        # print(f'mri referencia: {}')
        SOP_IDs.append(ds_roi.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].ContourImageSequence[seq].ReferencedSOPInstanceUID)
        ds_contour = ds_roi.ROIContourSequence[0].ContourSequence[seq].ContourData
        contourPixels = np.array(ds_contour[:]).reshape((len(ds_contour[:])//3, 3))
        contours.append(contourPixels)
        #print(f'Numero de puntos: {contourPixels.shape[0]}\n')
        #print(seq[:]==ds_contour)

        referenced_slice_number = ds_roi.ROIContourSequence[0].ContourSequence[seq].ContourImageSequence[0].ReferencedFrameNumber
        slices.append(referenced_slice_number)
        #print(f'Referenced slice number is: {referenced_slice_number}\n')

    #slice_number = referenced_slice_number - 1
    #print(f'Referenced SOP IDs = {SOP_IDs}')

    return contours, slices, SOP_IDs, num_seqs

'''
Funcion para extraer datos de la imagen real en el slice referenciado por el dicom con el contorno
'''
def get_image_data(dicom_path, slice_num, ref_SOP_ID):
    dicoms = next(os.walk(dicom_path))[2] # [2]: lists files; [1]: lists subdirectories; [0]: ?
    for dicom in natsorted(dicoms):
        ds_mri = dcmread(os.path.join(dicom_path, dicom))
        SOP_ID = ds_mri.SOPInstanceUID
        #print(f'SOP instance original: {SOP_ID}\n')
        if SOP_ID == ref_SOP_ID:
            break

    mri_img = ds_mri.pixel_array
    mri_dims = len(mri_img.shape)
    #print(f'dimensiones: {mri_dims}\n')
    if mri_dims == 3:
        mri_pixel_spacing = ds_mri[0x5200, 0x9230][slice_num][0x0028, 0x9110][0][0x0028, 0x0030][:]
        #print(f'mri_pixel_spacing = {mri_pixel_spacing}\n')
        mri_image_position_patient = ds_mri[0x5200, 0x9230][slice_num][0x0020, 0x9113][0][0x0020, 0x0032][:]
        #print(f'mri_image_position_patient = {mri_image_position_patient}\n')
    elif mri_dims == 2:
        mri_pixel_spacing = ds_mri.PixelSpacing[:]
        mri_image_position_patient = ds_mri.ImagePositionPatient[:]

    return mri_img, mri_pixel_spacing, mri_image_position_patient, mri_dims
