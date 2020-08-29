# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 14:15:48 2020

@author: debac
"""


from imageSearch.utils import * 
from imageSearch import database
import numpy as np
import time

start = time.time()
test = database.DB((150,200))

test.add_entries_from_folder("data/utrecht")
print("Training model...")
test.train_model(train_prop = 0.5, save_model="utrecht_face_model.pkl")

print("Testing model...")
res = test.test_model()
duration = time.time() - start
print("Training and testing took {:.2f} minutes".format(duration))

prop0 = np.mean(res[:,1]==0)
prop3 = np.mean(res[:,1] <3)
prop5 = np.mean(res[:,1] <5)
print("% of right matches in first places of ranking: {:2.2%}".format(prop0))
print("% of right matches in 3 first places of ranking: {:2.2%}".format(prop3))
print("% of right matches in 5 first places in ranking: {:2.2%}".format(prop5))

print("10 best matches for image #2, id '{}':".format(test.register[2][0]))
print(test.verify_with_all(2,10))
print("Matching between images #0 and #1: ")
test.match((0,1), show = True)


    
    

