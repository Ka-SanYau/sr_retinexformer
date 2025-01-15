import basicsr
from os import path as osp

# import 
# import metrics
# from metrics import *
import data
import model
import archs

if __name__=='__main__':
    # root_path = osp.
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    basicsr.test_pipeline(root_path)