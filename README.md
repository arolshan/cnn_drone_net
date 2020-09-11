# cnn_drone_net

## Setup
### 1. Initialize environment
In command line: 
```shell script
conda env create -f environment.yml
```

### 2. Activate environment
In command line: 
```shell script
conda activate cnn_drone_net
```
__Note:__
If you encounter 'import errors' make sure to run 'conda install <missing package>' or 'pip install <missing package>' for the missing packages. 

## Run
### 1. Single experiment
In command line: 
```shell script
#!/bin/bash
python main.py \ 
    --output_path="./output" \ 
    --lr="0.001" \
    --download_data="True" \
    --batch_size="8" \
    --epochs="3" \ 
    --cuda_device="0" \
    --print_rate="3" \
    --dropout="0.2" \
    --with_cuda="true" \
    --download_data="true" \
    --train_set="GoogleEarth" \
    --model="Resnet50"

# output_path: The output directory in which the logs, graphs, model state and args.json will be output. 
# download_data: Set to True, if first run. This will download the data sets from web and store in './data' directory. 
# lr: Learning rate for training. 
# batch_size: Batch size for training. 
# epochs: Number of epochs for training.
# print_rate: The rate of sampling metrics and tracing. 
# dropout: Dropout probability for training. 
# with_cuda: Set to 'True' if running on GPU.
# cuda_device: Cuda device, default should be '0'. Ignored if 'with_cuda' is set to 'False'.
# model: The CNN architecture for training. Possible values: ['Resnet50', 'VGG16', 'Mobilenet_V2'].
# train_set: The data set on which we fine-tune the model. Possible values: ['GoogleEarth', 'Satellite', 'UAV']
```

### 2. Multiple experiments
Place proper JSON files with desired arguments in a new directory of your choosing, for example: './args'. 
In command line: 
```shell script
python main.py \ 
    --input_args_dir="./args"
```

This will run every experiment based on the args files in the given directory.
Example for a proper args JSON file:

```json
{
    "output_path": "./output/3",
    "batch_size": 8,
    "epochs": 3,
    "print_rate": 10,
    "cuda_device": 0,
    "lr": 0.001,
    "dropout": 0.2,
    "with_cuda": true,
    "download_data": true,
    "train_set": "GoogleEarth",
    "model": "Resnet50"
}
```

### 3. Results
3.1. Results would be placed the directory set in the 'output_path' argument. 
3.2. In addition, plots aggregating results from all experiments (or single) would be placed in the relative './output' directory. 
