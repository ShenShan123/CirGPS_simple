# CirGPS_simple
## Few-shot Learning on AMS Circuits and Its Application to Parasitic Capacitance Prediction
Note: 
* This is a simple version of CirGPS.
* Dataset can be found at [Google Drive](https://drive.google.com/drive/folders/1sBQEXEFYQzav43KghIh1pybnusiJLotS?usp=drive_link)

![](imgs/fig-gps.png)
Graph representation learning is a powerful method to extract features from graph-structured data, such as analog/mixed-signal (AMS) circuits. However, training deep learning models for AMS designs is severely limited by the scarcity of integrated circuit design data. 
This is the repository of CirGPS, a few-shot learning method for parasitic effect prediction in AMS circuits.

The proposed method contains five steps: 
1. AMS netlist conversion, 
2. enclosing subgraph extraction, 
3. position encoding, 
4. model pre-training/fine-tuning. 

CirGPS is built on the top of [GraphGPS](https://github.com/rampasek/GraphGPS.git), which is
using [PyG](https://www.pyg.org/) and [GraphGym from PyG2](https://pytorch-geometric.readthedocs.io/en/2.0.0/notes/graphgym.html).
Specifically *PyG v2.2* is required.
To use all features backed up by GraphGPS, please go to another [repository](https://github.com/ShenShan123/CirGPS.git).

## Instructions

### Python environment setup with Conda
In this simple version of cirgps, we employed `LinkNeighborLoader`, please see the [pyg doc](https://pytorch-geometric.readthedocs.io/en/2.5.1/modules/loader.html#torch_geometric.loader.LinkNeighborLoader) for more details.
These codes are tested on our platform with 
- torch==2.1.0+cu118
- torch-cluster==1.6.3
- torch-geometric==2.6.1
- torch-scatter==2.1.2
- torch-sparse==0.6.18
- torch-spline-conv==1.2.2


### Running an experiment with CirGPS
run the `main.py` for training & evaluation

```bash
conda activate cirgps
```
To conduct training, type
```bash
# To use all arguments, see ArgumentParser in main.py
python main.py --dataset ssram+digtime --use_pe 0 --num_hops 1
```
