# DFI-Net
# Dual-Branch Feature Interaction Network
If you use data and code, please cite the paper:

M. Xu, M. Liu, Y. Liu, S. Liu and H. Sheng, "Dual-Branch Feature Interaction Network for Coastal Wetland Classification Using Sentinel-1 and 2," IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 17, pp. 14368-14379, 2024, doi: 10.1109/JSTARS.2024.3440640. 
# Requirements
python==3.8, CUDA==11.8, torch==1.13.1+cu116, scikit-learn==1.1.3, numpy==1.23.5

# Datasets
Our self-made Yellow River Delta dataset, then organize these datasets like:
datasets/
    Ori/
      data.mat
      label.mat

Datasets can be downloaded at: https://pan.baidu.com/s/1tJADqb2TjrUurROJutK1WA?pwd=abcd 
# Acknowledgement
Some of our codes references to the following projects, and we are thankful for their great work:
https://github.com/MeiShaohui/Group-Aware-Hierarchical-Transformer
