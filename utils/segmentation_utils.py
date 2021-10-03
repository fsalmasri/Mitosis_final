from skimage.measure import label, regionprops, regionprops_table
import pandas as pd
import numpy as np


def clean_outputs(im, j=7, n=7, duplicates_th=25):
    im_labeled = label(im)
    props_im = regionprops_table(im_labeled, properties=('centroid', 'major_axis_length', 'minor_axis_length'))
    props_im = pd.DataFrame(props_im)
    
    idx_remove = []
    for index, outs in props_im.iterrows():
        if outs['major_axis_length'] < j and outs['minor_axis_length'] < n:
            idx_remove.append(index)
            im_labeled[im_labeled == index+1] = 0
    
    for ix in range(len(props_im)-1):
        for jx in range(ix+1, len(props_im)):
            dist_x, dist_y, pred = validattion_duplicates(props_im.iloc[ix]['centroid-0'], props_im.iloc[ix]['centroid-1'], 
                                                          props_im.iloc[jx]['centroid-0'], props_im.iloc[jx]['centroid-1'],
                                                          threshold = duplicates_th)
            if pred:
                idx = props_im.index[ix] if props_im.iloc[ix]['major_axis_length'] < props_im.iloc[jx]['major_axis_length'] else props_im.index[jx]
                idx_remove.append(idx) 
                im_labeled[im_labeled == idx+1] = 0
     
    props_im.drop(np.unique(idx_remove), inplace=True)
        
    return im, props_im, im_labeled


def validattion_func(prop_out, prop_gt): 
    
    x1, y1 = prop_out['centroid-0'], prop_out['centroid-1']
    x2, y2 = prop_gt['centroid-0'], prop_gt['centroid-1']

    dist_x, dist_y = abs(x1-x2), abs(y1-y2)
    pred = 1 if dist_x <= 20 and dist_y <= 20 else 0
    
    return x1, y1, x2, y2, dist_x, dist_y, pred

def validattion_duplicates(x1, y1, x2, y2, threshold = 25): 

    dist_x, dist_y = abs(x1-x2), abs(y1-y2)
    pred = 1 if dist_x <= threshold and dist_y <= threshold else 0
    
    return dist_x, dist_y, pred


def extract_segmentation(im_bmp, im_fake, im_gt, crop_size, j, n):
    
    im_mask_labeled = label(im_gt)
    props_gt = regionprops_table(im_mask_labeled, properties=('centroid', 'major_axis_length', 'minor_axis_length'))
    props_gt = pd.DataFrame(props_gt)
    
    thresholded_labeled, props_out, im_labeled = clean_outputs(im_fake, j=j, n=n)

    seg_imgs, labels = [], []
    for index, outs in props_out.iterrows():
        flag = 0
        
        x1 = outs['centroid-0']
        y1 = outs['centroid-1']
        
        stx = int(x1)- (crop_size//2)
        if stx < 0: stx = 0
        edx = int(stx) + crop_size
        if edx > im_fake.shape[0]:
            diff = edx - im_fake.shape[0]                
            edx -= diff
            stx -= diff

        sty = int(y1)- (crop_size//2)
        if sty < 0: sty = 0
        edy = int(sty) + crop_size
        if edy > im_fake.shape[1]:
            diff = edy - im_fake.shape[1]                
            edy -= diff
            sty -= diff

        seg_imgs.append(im_bmp[stx:edx, sty:edy])
        labels.append(index)

    
    return np.array(seg_imgs), im_labeled, props_out, props_gt

