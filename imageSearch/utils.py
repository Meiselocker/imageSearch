# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 14:21:34 2020

@author: debac
"""

import sys
import os
import cv2
import numpy as np

def read_image(path, target_size, blurK):
    """ Reads an image, resizes and blurs it. Used in DB.add_entry(). 
    
    Parameters
    ----------
    
    path: STRING.
        Path to the image.
    target_size: Tuple of INT.
        Length and width to which the image is to be resized.
    blurK: INT (odd).
        Kernal size for Gaussian blur.
    
    Returns
    -------
    
    2D gray image of size target_size
    """
    img = cv2.imread(path)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(3,3))
    l = clahe.apply(l)
    lab = cv2.merge([l,a,b])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blurK,blurK), 0)
    return cv2.resize(gray, target_size)

def eucl_distance(pt1, pt2):
    """
    Calculate distanced between pairs of keypoints. Used in extract_features().

    Parameters
    ----------
    pt1 : open-cv KEYPOINT object.

    pt2 : open-cv KEYPOINT object.

    Returns
    -------
    FLOAT array.
        Array of euclidian distances between pairs of keypoints.

    """
    x_diff = pt1[0]-pt2[0]
    y_diff = pt1[1]-pt2[1]
    return (np.sqrt(x_diff**2+y_diff**2))


def extract_features(match_data):
    """
    Extract features from match data, which will be used in a model to assess 
    degree of similarity between image. Used in the method DB.match()

    Parameters
    ----------
    match_data : LIST containing MATCH objects, KEYPOINTS object of image1,
    KEYPOINS object of image2, and homography matrix (3x3).


    Returns
    -------
    1-d FlOAT array
        Array containing match features: number of matches, mean distances between
        the matches, standard deviation of the matches and the Homography matrix
        coefficients.

    """
    matches, kp1, kp2, H = match_data
    n = len(matches)
    if n == 0:
        return np.concatenate([[0,999,999],H.reshape(-1)])
    coords = [(pt1.pt, pt2.pt) for pt1,pt2 in zip(kp1, kp2)]
    eucl_dists = [eucl_distance(pair[0], pair[1]) for pair in coords]
    dists_mean = np.array(eucl_dists).mean()
    dists_std = np.array(eucl_dists).std()
    return np.concatenate([[n, dists_mean, dists_std], H.reshape(-1)])

def load_DB(path):
    """
    Loads database saved as pickle object, saved using DB.save().

    Parameters
    ----------
    path : STRING
        Path to the the saved database pickle object.

    Returns
    -------
    my_DB : DB Class object
        The saved DB object.

    """
    print("loading database...")
    import pickle
    with open(path, 'rb') as src:
        my_DB = pickle.load(src)
    n = len(my_DB.register.keys())
    print("creating keypoints...")
    for i,k in enumerate(my_DB.register.keys()):
        kps = []
        for kp in my_DB.register[k][2]:
            temp = cv2.KeyPoint(x=kp[0][0],y=kp[0][1],_size=kp[1],
                                   _angle=kp[2],_response=kp[3],
                                   _octave=kp[4], _class_id=kp[5])
            kps.append(temp)
        my_DB.register[k][2] = kps
        updt(n,i+1)
    return my_DB    
    


# def remove_many2one(matches, kp1, kp2):
#     """
    
#     Removes matches and corresponding keypoints when target keypoint is matched
#     by multiple source keypoints. Used in extract_features().

#     Parameters
#     ----------
#     matches : open cv list of MATCH objects
#     kp1 : list of KEYPOINTS
#         Source keypoints.
#     kp2 : List of KEYPOINTS
#         Target keypoints.

#     Returns
#     -------
#     matches : open cv list of MATCH objects
#     kp1 : list of KEYPOINTS
#         Source keypoints.
#     kp2 : List of KEYPOINTS
#         Target keypoints.

#     """
#     a = [kp.pt for kp in kp2]
#     duplicates = [item for item, count in collections.Counter(a).items() if count > 1]
#     dupl_idx = [kp.pt in duplicates for kp in kp2]
#     matches = [m for m,i in zip(matches, dupl_idx) if not i]
#     kp1 = [kp for kp,i in zip(kp1, dupl_idx) if not i]
#     kp2 = [kp for kp,i in zip(kp2, dupl_idx) if not i]
#     return matches , kp1,kp2

def updt(total, progress):
    """
    Displays or updates a console progress bar.
    Original source: https://stackoverflow.com/a/15860757/1391441
    
    INPUT
    
    total: number of iteration (int)
    progress : ith iteration (int)
    """
    barLength, status = 20, ""
    progress = progress/total
    if progress >= 1.:
        progress, status = 1, "\r\n"
    block = int(round(barLength * progress))
    text = "\r[{}] {:.0f}% {}".format(
        "#" * block + "-" * (barLength - block), round(progress * 100, 0),
        status)
    sys.stdout.write(text)
    sys.stdout.flush()


def tabular_print(elements):
    """
    Prints list of lists as table
    
    Parameters
    ----------
    
    elements : List of lists with elements of any type
    
    Returns
    ------
    
    None
    """
    pads = [max([len(str(el[i])) for el in elements]) \
            for i in range(len(elements[0]))]
    for el in elements:
        print("")
        for i,item in enumerate(el):
            print("{it:{p_i}}".format(it = str(item), p_i = pads[i]+2), end = " ")
        