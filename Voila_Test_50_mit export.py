#!/usr/bin/env python
# coding: utf-8

# In[5]:


import fastbook
fastbook.setup_book()
from fastbook import *
from fastai.vision.widgets import *


# In[7]:


path = Path()
learn_inf = load_learner(path/'export.pkl')
btn_upload = widgets.FileUpload()
out_pl = widgets.Output()
lbl_pred = widgets.Label()


# In[8]:


def on_click_classify(change):
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl: display(img.to_thumb(128,128))
    pred,pred_idx,probs = learn_inf.predict(img)
    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'


# In[9]:


btn_upload.observe(on_click_classify, names=['data'])


# In[10]:


display(VBox([widgets.Label('Select a picture!'), 
      btn_upload, out_pl, lbl_pred]))

