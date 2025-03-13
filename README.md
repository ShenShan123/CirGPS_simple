# CircuitGPS_simple
## Few-shot Learning on AMS Circuits and Its Application to Parasitic Capacitance Prediction
Note: 
* This is a simple version of CircuitGPS.
* Dataset can be found at [Google Drive](https://drive.google.com/drive/folders/1sBQEXEFYQzav43KghIh1pybnusiJLotS?usp=drive_link)

![](imgs/fig-gps.png)
Graph representation learning is a powerful method to extract features from graph-structured data, such as analog/mixed-signal (AMS) circuits. However, training deep learning models for AMS designs is severely limited by the scarcity of integrated circuit design data. 
This is the repository of CirGPS, a few-shot learning method for parasitic effect prediction in AMS circuits.

The proposed method contains five steps: 
1. AMS netlist conversion, 
2. enclosing subgraph extraction, 
3. position encoding, 
4. model pre-training/fine-tuning. 

CircuitGPS is built on the top of [GraphGPS](https://github.com/rampasek/GraphGPS.git), which is
using [PyG](https://www.pyg.org/) and [GraphGym from PyG2](https://pytorch-geometric.readthedocs.io/en/2.0.0/notes/graphgym.html).
Specifically *PyG v2.2* is required.
To use all features backed up by GraphGPS, please go to another [repository](https://github.com/ShenShan123/CirGPS.git).

## Instructions

### Python environment setup with Conda
These codes are tested on our platform with pytorch=1.13.1, cuda=11.7.

Firstly, you need to install Git Large File Storage [LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage).

Then, you can directly use our conda config file.
``` bash
conda env create -f environment.yaml
```
If you enconter any problem, try pip instead,
```bash
conda create -n cirgps
pip install -r requirements.txt
```
or use the oringal environment of [GraphGPS](https://github.com/rampasek/GraphGPS.git).

### Running an experiment with CirGPS
In the 'configs/sram' floder, the files starting with 'sram-' are for the link-prediction task, those starting with 'reg-' are for the edge-regression task.

```bash
conda activate cirgps
```
To conduct training, type
```bash
# To use all arguments, see ArgumentParser in main.py
python main.py --dataset ssram+digtime --use_pe 0 --num_hops 1
```
