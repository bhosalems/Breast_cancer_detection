#!/usr/bin/env python
# coding: utf-8

# In[11]:


import os
import re
import pydicom as pyd
import png
import re
import numpy as np


# In[12]:


files=os.listdir('C:/Users/yeswa/Desktop/Semester-02 Spring 22/CSE_676-Deep Learning/project/datasets/INbreast/AllDICOMs')


# In[15]:


dicom=[]
for file in files:
    if file.endswith('dcm'):
        dicom.append(os.path.join(file))
print(len(dicom))


# In[18]:


filevals=[]
for filename in dicom:
    filevals.append(re.findall( r"^([^.]*).*" , filename)[0])
print(filevals)


# In[27]:


os.mkdir('datasets/INbreast/AllPNGs')


# In[37]:


for file in filevals:
    dcm=pyd.dcmread('datasets/INBreast/AllDICOMs/%s.dcm'%file)
    shape=dcm.pixel_array.shape
    image_2d=dcm.pixel_array.astype(float)
    image_2d_scaled=(np.maximum(image_2d,0)/image_2d.max())*256
    image_2d_scaled=np.uint8(image_2d_scaled)
    
    with open('datasets/INBreast/AllPNGs/%s.png'%file,'wb') as png_file:
        w=png.Writer(shape[1],shape[0],greyscale=True)
        w.write(png_file,image_2d_scaled)


# In[ ]:




