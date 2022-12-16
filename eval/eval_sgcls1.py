#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
import sys
import os
codebase = '/data/yejj/programs/gbnet-master-gai-1'
sys.path.append(codebase)


# In[2]:


import torch
torch.__version__


# In[3]:


exp_name = 'exp_145'
os.environ["CUDA_VISIBLE_DEVICES"]="3"


# In[4]:




# In[5]:


from dataloaders.visual_genome import VGDataLoader, VG
import numpy as np
from torch import optim
import torch
import pandas as pd
import time
import os
from tqdm import tqdm
import pickle

from config import ModelConfig, BOX_SCALE, IM_SCALE
from torch.nn import functional as F
from lib.pytorch_misc import optimistic_restore, de_chunkize, clip_grad_norm
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator, calculate_mR_from_evaluator_list, eval_entry
from lib.pytorch_misc import print_para
from torch.optim.lr_scheduler import ReduceLROnPlateau

from lib.my_model_33 import KERN


# In[6]:

eval_epoch = ModelConfig().eval_epoch

conf = ModelConfig(f'''
-m sgcls -p 1000 -clip 5 
-ckpt checkpoints/kern_sgcls/{exp_name}/vgrel-{eval_epoch}.tar 
-test
-b 4
-ngpu 1
-cache caches/{exp_name}/kern_sgcls-{eval_epoch}.pkl \
-save_rel_recall results/{exp_name}/kern_rel_recall_sgcls-{eval_epoch}.pkl
''')


# In[7]:


train, val, test = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                          use_proposals=conf.use_proposals,
                          filter_non_overlap=conf.mode == 'sgdet')


# In[8]:
print('----------------------------------------------------')
print('mode: ',conf.mode,'   ', 'eval_epoch: ',eval_epoch)
print('----------------------------------------------------')



ind_to_predicates = train.ind_to_predicates # ind_to_predicates[0] means no relationship
if conf.test:
    val = test


# In[9]:


train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                               batch_size=conf.batch_size,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus)


# In[10]:


detector = KERN(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates,
                num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
                use_resnet=conf.use_resnet, use_proposals=conf.use_proposals, pooling_dim=conf.pooling_dim,
                ggnn_rel_time_step_num=3, ggnn_rel_hidden_dim=1024, ggnn_rel_output_dim=None,
                graph_path=os.path.join(codebase, 'graphs/005/internal2hop.pkl'),
                emb_path=os.path.join(codebase, 'graphs/001/emb_mtx.pkl'), 
                rel_counts_path=os.path.join(codebase, 'graphs/001/pred_counts.pkl'), 
                use_knowledge=True, use_embedding=True, refine_obj_cls=True,
                class_volume=1000.0, top_k_to_keep=5, normalize_messages=False,
               )


# In[11]:


detector.cuda();


# In[12]:


ckpt = torch.load(conf.ckpt)
optimistic_restore(detector, ckpt['state_dict'])


# In[ ]:





# In[13]:


def val_batch(batch_num, b, evaluator, evaluator_multiple_preds, evaluator_list, evaluator_multiple_preds_list, thrs=(20, 50, 100)):
    det_res = detector[b]
    if conf.num_gpus == 1:
        det_res = [det_res]

    for i, (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i) in enumerate(det_res):
        gt_entry = {
            'gt_classes': val.gt_classes[batch_num + i].copy(),
            'gt_relations': val.relationships[batch_num + i].copy(),
            'gt_boxes': val.gt_boxes[batch_num + i].copy(),
        }
        assert np.all(objs_i[rels_i[:,0]] > 0) and np.all(objs_i[rels_i[:,1]] > 0)
        # assert np.all(rels_i[:,2] > 0)

        pred_entry = {
            'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE,
            'pred_classes': objs_i,
            'pred_rel_inds': rels_i,
            'obj_scores': obj_scores_i,
            'rel_scores': pred_scores_i,
        }
        all_pred_entries.append(pred_entry)

        eval_entry(conf.mode, gt_entry, pred_entry, evaluator, evaluator_multiple_preds, 
                   evaluator_list, evaluator_multiple_preds_list)


# In[14]:


evaluator = BasicSceneGraphEvaluator.all_modes()
evaluator_multiple_preds = BasicSceneGraphEvaluator.all_modes(multiple_preds=True)
evaluator_list = [] # for calculating recall of each relationship except no relationship
evaluator_multiple_preds_list = []
for index, name in enumerate(ind_to_predicates):
    if index == 0:
        continue
    evaluator_list.append((index, name, BasicSceneGraphEvaluator.all_modes()))
    evaluator_multiple_preds_list.append((index, name, BasicSceneGraphEvaluator.all_modes(multiple_preds=True)))


# In[15]:


all_pred_entries = []

if conf.cache is not None and os.path.exists(conf.cache): ########## IMPORTANT ############
    print("Found {}! Loading from it".format(conf.cache))
    with open(conf.cache,'rb') as f:
        all_pred_entries = pickle.load(f)
    for i, pred_entry in enumerate(tqdm(all_pred_entries)):
        gt_entry = {
            'gt_classes': val.gt_classes[i].copy(),
            'gt_relations': val.relationships[i].copy(),
            'gt_boxes': val.gt_boxes[i].copy(),
        }

        eval_entry(conf.mode, gt_entry, pred_entry, evaluator, evaluator_multiple_preds, 
                   evaluator_list, evaluator_multiple_preds_list)

    recall = evaluator[conf.mode].print_stats()
    recall_mp = evaluator_multiple_preds[conf.mode].print_stats()
    
    mean_recall = calculate_mR_from_evaluator_list(evaluator_list, conf.mode, save_file=conf.save_rel_recall)
    mean_recall_mp = calculate_mR_from_evaluator_list(evaluator_multiple_preds_list, conf.mode, multiple_preds=True, save_file=conf.save_rel_recall)

else:
    detector.eval()
    for val_b, batch in enumerate(tqdm(val_loader)):
        val_batch(conf.num_gpus*val_b, batch, evaluator, evaluator_multiple_preds, evaluator_list, evaluator_multiple_preds_list)

    recall = evaluator[conf.mode].print_stats()
    recall_mp = evaluator_multiple_preds[conf.mode].print_stats()
    
    mean_recall = calculate_mR_from_evaluator_list(evaluator_list, conf.mode, save_file=conf.save_rel_recall)
    mean_recall_mp = calculate_mR_from_evaluator_list(evaluator_multiple_preds_list, conf.mode, multiple_preds=True, save_file=conf.save_rel_recall)

    if conf.cache is not None:
        with open(conf.cache,'wb') as f:
            pickle.dump(all_pred_entries, f)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




