# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 14:16:35 2020

@author: debac
"""

#Silence deprecation warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pickle 
from utils import *

class DB:
    """Class for registering images, searching for new input images and training model.
    INPUT
    target_size : Length and width (tuple of 2 integers) to which the image is to be resized
    """
    def __init__(self, target_size):
        self.target_size = target_size
        self.blur_factor = 0.03
        self.register = {}
        self.matches = {}
        self.ratio_threshold = 0.9
        self.hessianThreshold = 100
        self.nOctaves = 10
        self.extended = 1
        self.upright = 1
        self.test_ids = None
        self.load_model("model.pkl")
            
    def add_entry(self, path, identity = 'na', verbose = True, *args):
        """Add a new image to the database in form of keypoints and descriptors.
        Each image is associated with a unique key, an identity and a path.
        
        Parameters
        ----------
        
        path: STRING.
            Path to image.
        identity: STRING, optional.
            Identity of the individual on the image, if known. Default is 'na'.
        verbose: BOOL, optional.
            Whether to print a confirmation that the new entry was added. Default = True.
        
        Returns
        -------
        
        The key associated with the new image (integer)
        """
        paths = [self.register[key][1] for key in self.register]
        if path in paths:
            print("\n{} already exists in database, refer to entry # {}".format(path, paths.index(path)))
            return 
        blurK = int(self.blur_factor*(self.target_size[0] + self.target_size[1])/2)
        blurK = blurK + (1-blurK%2)
        img = read_image(path, self.target_size, blurK)
        surf = cv2.xfeatures2d.SURF_create(upright = self.upright,
                                           hessianThreshold = self.hessianThreshold,
                                           nOctaves = self.nOctaves,
                                           extended = self.extended)
        kp, desc = surf.detectAndCompute(img, None)
        key = len(self.register.keys())
        self.register[key] = [identity, path, kp, desc]
        if verbose:
            print("New entry (# {}) registered. Picture id: {}".format(key, identity))
        return key
    
    def add_entries_from_folder(self, path, id_from_folder = True):
        """
        Adds new images to the register from a folder, where each sub-folder contains
        image(s) from one individual.
        
        Parameters
        ----------
        path: STRING.
            Path to the folder.
        id_from_folder: BOOL, optional.
            Whether id should be inferred from the subfolders' names. Default is True.
        
        Returns
        -------
        None.
        
        """
        paths = []
        idxs = []
        for identity in os.listdir(path):
          fold = os.path.join(path, identity)
          for img_path in os.listdir(fold):
            img_path = os.path.join(fold, img_path)
            paths.append(img_path)
            idxs.append(identity)
        if not id_from_folder:
            idxs = np.repeat(None, idxs.size)
        n_imgs = len(paths)
        #coherent use of backward vers forward slash
        paths = [path.replace('\\', '/') for path in paths]        
        print ("\nRegistering images...")
        runs = len(paths)
        for i, path in enumerate(paths):
            self.add_entry(path, idxs[i], verbose = False)
            updt(runs, i+1)
        print(" ")

    def assign_id(self, key, i):
        """
        Assigns an identity to am image given its key.

        Parameters
        ----------
        key : INT
            Register key of the image.
        i : STRING
            Identity to be assigned.

        Returns
        -------
        None.

        """
        if not isinstance(i,str):
            print ("\ni should be a string. Try again.")
            return 
        self.register[key][0] = i
        print("picture {} was assigned the id {}".format(key, i))
        
    def delete_entry(self, key):
        """
        Deletes an entry and associated data.

        Parameters
        ----------
        key : INT
            Key of entry whose data are to be deleted.

        Returns
        -------
        None.

        """
        del self.register[key]
        ms = []
        for m in self.matches.keys():
            if key in m:
                ms.append(m)
        for m in ms:
            del self.matches[m]
        print("\nEntry # {} has been deleted".format(key))
        
    def delete_id(self, i):
        
        """
        Deletes all entries and matches associated with a given key.

        Parameters
        ----------
        key : INT
            Key of entry whose data are to be deleted.

        Returns
        -------
        None.
        """
        for k in self.register.keys():
            if self.register[k][0] == i:
                del self.register[k]
                break
        ms = []
        for m in self.matches.keys():
            if k in m:
                ms.append(m)
        for m in ms:
            del self.matches[m]
        print("\nAll entries associated with {} have been deleted".format(i)) 
        
    def get_details(self, key):
        """
        Displays details (id and path) for a given register key.

        Parameters
        ----------
        key : INT
            Key of image whose details are to be displayed..

        Returns
        -------
        None.

        """
        print({"id:" : self.register[key][0], "path:" : self.register[key][1]})
        
    def image_for_id(self, ids):
        """
        Returns image keys associated with ids. 
    
        Parameters
        ----------
        ids : STRING or LIST of STRINGS
            A string of list of string containing the ids whose image keys should
            be returns.
    
        Returns
        -------
        ids_dict : DICT
            A dictionary mapping from ids to the associated keys.
    
        """
        if isinstance(ids, str):
            ids = [ids]
        ids_dict = {id:[] for id in ids}
    
        for key,vals in self.register.items():
            identity = vals[0]
            if identity in ids_dict:
                ids_dict[identity].append(key)
        return ids_dict    
     
    def load_model(self, path):
        """Loads image verification model to the DB object.
        
        Parameters
        ----------
        
        Path: STRING.
            Path to a pickled sklearn model.
            
        Returns
        ------
        
        None.
        
        """
        with open(path, 'rb') as src:
            self.model = pickle.load(src)
            
    def match(self, key_tuple, show = False, force_rematch = False, verbose = True):
        """
        
        Matches descriptors of keypoints in two images, and creates open-cv match objects,
        selected keypoints and a homography matrix, from which
        match similarity features are extracted. They are subsequently used to assess
        degree of identity between the images.

        Parameters
        ----------
        key_tuple : tuple of two INT.
            Keys of the image to be matched.
        show : BOOL, optional
            Whether to show the pair of images with the keypoints and drawn matches.
            The default is False
        force_rematch : BOOL, optional
            If match already exsists whether to rematch. The default is False.

        Returns
        -------
        None.

        """
        key_tuple = ((max(key_tuple), min(key_tuple)))
        if show:
            blurK = int(self.blur_factor*(self.target_size[0] + self.target_size[1])/2)
            blurK = blurK + (1-blurK%2)
            im1 = read_image(self.register[key_tuple[0]][1],
                             self.target_size, blurK)
            im2 = read_image(self.register[key_tuple[1]][1],
                             self.target_size, blurK)
            force_rematch = True            
        if self.matches.get(key_tuple) is not None and not force_rematch and verbose:
            print("\nMatch between keys of pair {} already exists".format(key_tuple))
            print("To force rematch, specify: force_rematch = True")
            return
        _, _, kp1, des1 = self.register[key_tuple[0]]
        _, _, kp2, des2 = self.register[key_tuple[1]]
        #BF matching
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k = 2)
        #Lowe's test
        good_matches = []
        for m1, m2 in matches:
            if m1.distance < self.ratio_threshold*m2.distance:
                good_matches.append(m1)
        src_points = [kp1[m.queryIdx].pt for m in good_matches]
        dst_points = [kp2[m.trainIdx].pt for m in good_matches]
        #removing one-to-many
        idx = [dst_points.index(pts) for pts in set(dst_points)]
        good_matches = [good_matches[i] for i in idx]
        #finding homography
        src_pts = np.float32([kp1[m.queryIdx].pt\
                              for m in good_matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt \
                              for m in good_matches]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        #filtering outliers
        def_matches = [m for m, is_good in zip(good_matches, mask) if is_good]
        kp_src = [kp1[m.queryIdx] for m in def_matches]
        kp_dst = [kp2[m.trainIdx] for m in def_matches]
        if show:
            res = cv2.drawMatches(im1,kp1, im2, kp2, def_matches, im1)
            cv2.imshow("images " + str(key_tuple[0]) + " and " + str(key_tuple[1]),
                       res); cv2.waitKey(0)
        x = extract_features([def_matches, kp_src, kp_dst, H])
        self.matches[key_tuple] = x

    def match_with_all(self, i):
        """
        Matches an image of key i with all other images.

        Parameters
        ----------
        i : INT
            Key of the image to be matched to all other images.

        Returns
        -------
        None.

        """
        for k in self.register.keys():
            pair = (max(i,k), min(i,k))
            self.match(pair)
        
    def match_batch(self, ids):
        """
        Matches between all possible pairs of images associated with an
        identity belonging to ids. 
        

        Parameters
        ----------
        ids : LIST of STRINGS.
            Identities whose images should be matched with each other. 

        Returns
        -------
        None.

        """
        n = len(self.register) - 1
        pairs = [(i,j) for i in range(n) for j in range(n) \
                 if i > j \
                     and self.register[i][0] in ids \
                         and self.register[j][0] in ids] 
        n_pairs = len(pairs)
        for i, pair in enumerate(pairs):
            self.match(pair, verbose = False)
            updt(n_pairs ,i)
            
    def save(self, path):
        """
        Saves database at specified path as pickle object. Saved database can
        read again into Python with the function load_DB().
        

        Parameters
        ----------
        path : STRING
            Path where the database is to be saved.

        Returns
        -------
        None.

        """
        temp = []
        for k in self.register.keys():
            for kp in self.register[k][2]:
                temp_tuple = (kp.pt, kp.size, kp.angle, kp.response,
                              kp.octave, kp.class_id)
                temp.append(temp_tuple)
            self.register[k][2] = temp  
        with open(path, 'wb') as dst:
            pickle.dump(self, dst)

    def test_model(self, test_ids = None):
        """
        Tests model's quality by pairing images associated with test_ids 
        and returning the position of the true pairs (i.e. two images of the same
        individual) in the ordered similarity probability ranking. The higher the 
        position, the better the model. If several images of the same individual
        exist, a pair is randomly chosen among them.
        
        Parameters
        ----------    
        test_ids : List of STRING or FLOAT.
            If list of string, what IDs the model should be tested on. If float, 
            proportion of IDS to be used. If None, the model is tested on the 
            IDs assigned in train/test split during training. Omitting test_ids
            is accepted only if train_model() was called before.
        
        Returns
        -------
        2d INT array
            2d array of integers with dimensions(n,3). The first column is 
            the key of tested image, the second is the position of a twin
            image in the similarity ranking, and the third is the/a true
            twin image.
        
        """

        if isinstance(test_ids, list):
            self.test_ids = test_ids
        elif isinstance(test_ids, float):
            all_ids = [self.register[key][0] for key in self.register]
            self.test_ids = np.random.choice(all_ids,
                                             int(len(all_ids)*test_ids),
                                             replace = False)
        if self.test_ids is None:
            print("Specify ID's to test model on.")
            return 
        id2key = self.image_for_id(self.test_ids)

        selected_keys = [np.random.choice(val, 2, replace = False)\
                        for val in id2key.values()]

        keys1 = np.array([key_pair[0] for key_pair in selected_keys])
        keys2 = np.array([key_pair[1] for key_pair in selected_keys])
     
        positions = []
        twins = []
        n = len(selected_keys)
        for i,key1 in enumerate(keys1):
            twin = keys2[i]
            probs = [self.verify((key1, key2)) for key2 in keys2]
            ordered_keys2 = [k for _, k in sorted(zip(probs, keys2),
                                                  reverse = True)]
            pred_position = ordered_keys2.index(twin)
            positions.append(pred_position)
            updt(n, i+1)
        
        return np.stack([keys1, positions, keys2], axis = 1)

    def train_model(self, train_prop = 1.0, clf = None, save_model = None):
        """
        Trains model to return similarity probabilities given the identified 
        images existing in the register.  If a classifier clf is not provided
        a stacking classifier using random forest and support vector machine
        is used. If save_model is given, the model will be saved.

        Parameters
        ----------
        train_prop : FLOAT, optional.
            Proportion of the registered individuals to use for training. The 
            rest is saved as test individuals as DB.test_id attribute.
        clf : scikit-learn model, optional
            The model to be used for training. The default is None, in which 
            case themodel by default is used.
        save_model : STRING, optional
            Path where the model should be saved for subsequent use. Default is
            None, meaning that the model is not saved. 

        Returns
        -------
        model : scikit-learn model
            The model can be used for evaluating similarity probability between pairs
            of images.

        """
        
        pool = list(set([self.register[key][0] for key in self.register]))
        train_ids = np.random.choice(pool, int(len(pool)*train_prop), replace=False)
        self.test_ids = np.setdiff1d(pool, train_ids)
        if clf is None:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import make_pipeline
            from sklearn.ensemble import StackingClassifier
            from sklearn.svm import SVC
            
            svc = make_pipeline(StandardScaler(),
                                SVC(probability = True, class_weight="balanced"))
            rf = RandomForestClassifier(class_weight="balanced")
            clf = StackingClassifier(estimators = [('svc', svc), ('rf', rf)])
        print("\n Matching pairs of images...\n")
        self.match_batch(train_ids)
        image_pairs = self.matches.keys()
        id_pairs = [(self.register[pair[0]][0], self.register[pair[1]][0])\
                        for pair in image_pairs]
        pair_samples = [(i,p) for i,p in zip(id_pairs, image_pairs)\
                        if i[0] in train_ids and i[1] in train_ids]     
        X = np.array([self.matches[p]\
                      for _,p in pair_samples])
        y = np.array([i[0]==i[1] for i,p in pair_samples])
        clf.fit(X,y)
        from sklearn.metrics import confusion_matrix
        print("\nConfusion matrix for training dataset")
        print(confusion_matrix(y, clf.predict(X)))
        if save_model is not None:
            self.model = clf
            with open(save_model, 'wb') as dst:
                pickle.dump(self.model, dst)
        return clf

    def verify(self, pair):
        """
        Gives the similarity probability between a pair of registered images.

        Parameters
        ----------
        pair : Tuple of integers.
            Keys of the images.

        Returns
        -------
        res: (FLOAT, INT, STRING).
            The returned tuple contains a similarity probability, the key 
            and the registered identity.

        """
        pair = (max(pair[0], pair[1]), min(pair[0], pair[1]))
        if self.matches.get(pair) is None:
            self.match(pair)
        X = self.matches[pair]
        res = self.model.predict_proba(X.reshape(1,-1))[:,1]
        return res

    def verify_with_all(self, i, n = None, return_ratio = False):
        """
        Gives the pairwise similarity probabilities between the image registered
        under key i and all other images.

        Parameters
        ----------
        i : INT
            Key of the image.
        n : INT
            How many top results are to be returned. By default None, i.e. all
        return_ratio: BOOL, optional.
            Whether ratio between probabilities ranked i and i+1 should be
            returned. Dafault is false.

        Returns
        -------
        LIST of tuples.
            The list returned contains n-1 tuples 
            (where n is the number of images in the register by default, or a 
            specified integer).
            The tuples
            are those returned by the DB.verify() method. The tuples are
            ordered by decreasing probability.

        """
        probs = []
        ids = []
        paired = list(self.register.keys())
        paired.remove(i)
        n_pairs = len(paired)
        for j in paired:
            pr = self.verify((i,j))[0]
            probs.append(np.round(pr,4))
            ids.append(self.register[j][0])
        res = sorted(zip(probs, ids, paired), key = lambda tr: tr[0],
                     reverse = True)
        if return_ratio:
            ratios = [res[i][0]/res[i+1][0] for i in range(len(res)-1)]
            ratios.append(-1)
            res = [(a,b,c,round(d,4)) for (a,b,c),d in zip(res, ratios)]
        if n is None:
            n = len(res) 
        return res[:(n+1)]

    def verify_with_batch(self, i, keys, return_ratio = False):
        """
        Gives the pairwise similarity probabilities between the image registered
        under key i and a list of images.        

        Parameters
        ----------
        i : INT
            Key of the image to be paired.
        keys : LIST of INT
            Keys of the images to be paired with image i .
        return_ratio: BOOL, optional.
            Whether ratio between probabilities ranked i and i+1 should be
            returned. Dafault is false.
        Returns
        -------
        LIST of tuples.
            The list returned contains n-1 tuples 
            (where n is the number of images in the register). The tuple
            are those returned by the DB.verify() method. The tuples are
            ordered by decreasing probability.

        """
        keys = list(keys)
        probs = []
        ids = []
        paired = [x for x in self.register.keys() if x in keys]
        paired.remove(i)
        for j in paired:
            pr = self.verify((i,j))[0]
            probs.append(np.round(pr,4))
            ids.append(self.register[j][0])
        res = sorted(zip(probs, ids, paired), key = lambda tr: tr[0],
                     reverse = True)
        if return_ratio:
            ratios = [res[i][0]/res[i+1][0] for i in range(len(res)-1)]
            ratios.append(-1)
            res = [(a,b,c, round(d,4)) for (a,b,c),d in zip(res, ratios)]
        return res

    def view_register(self, keys = None):
        """
        Displays data for images in register (key, id, path, and other images
        of same individual). To display a limited
        number of entries, use start and end to select slices of the register.
        For example, DB.view_register(10,20) displays the 10th to the 20th 
        entries.

        Parameters
        ----------
        keys : INT or list of INT, Optional. 
            Keys in the register that are to be viewed. By default all keys.

        Returns
        -------
        None.

        """
        if keys is None:
            keys = list(self.register.keys())
        if not isinstance(keys, list):
            keys = [keys]
        elements = []
        elements.append(["*Key*", "*Assigned_id*", "*Images_for_id*", "*Path*"])
        for key in keys:
            id_ = self.register[key][0]
            path = self.register[key][1]
            other_ims = self.image_for_id(id_)[id_]
            elements.append([key, id_, other_ims, path])
        tabular_print(elements)
            
    def view_twin_keys(self,key):
        identity = self.register[key][0]
        images = self.image_for_id(identity)[identity]
        images.remove(key)
        return images
        
