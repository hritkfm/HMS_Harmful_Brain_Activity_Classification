# HMS-harmful-brain-activity-classification

This is the solution for the T.H. part of the KTMUD team that won 5th place in the [HMS-harmful-brain-activity-classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification).

## Solution

Please read [this post](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492652#2745305) on Kaggle Discussion.

## Hardware

I used Nvidia DGX V100 station

```
GPU: Tesla V100 32GB * 4
CPU: Intel Xeon E5-2698 v4（20core、2.2GHz）
memory: 256GB
```

## Recuirement

I recommend using docker.

## HOW TO USE

### Build docker

Create a container based on docker/Dockerfile.  
An example of creating a container is shown below.
Adjust command options to suit your environment!

```bash
cd docker
docker build -t kaggle:hms .
docker container run --gpus all -id -d --net host \
  --shm-size=40g --name hms_2024	\
  -v $HOME/.cache/:/home/user/.cache/ \
	-v $HOME/.kaggle/:/home/user/.kaggle/ \
	-v $HOME/.vscode-server/:/home/user/.vscode-server/ \
	-v $HOME/.tmp/:/home/user/.tmp/ \
  -v $HOME/HMS_Harmful_Brain_Activity_Classification/:/home/user/HMS_Harmful_Brain_Activity_Classification/ \
  kaggle:hms /bin/bash
docker attach hms_2024
```

### Data download

Please download competition data from [kaggle](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/data) and put them to `./hms-harmful-brain-activity-classification` and unzip them.

### Prepare_data

```bash
cd src
python prepare_data.py
```

This will create `train_fold_irr_mark_v2.csv` and `train_fold_irr_mark_v5.csv` in the `./hms-harmful-brain-activity-classification`

### Train

#### 1st stage training

Run the first stage training script.
This will train 9 models.

```bash
bash run_train_1st.sh
```

When the 1st stage training is completed, a checkpoints folder(`./checkpoints`) will be created in the root directory of this project.The weights of the model and the config file in the first stage are saved in it.

#### 2nd stage training

Run the 2nd stage training script.  
Please note that it will not work if there are multiple .ckpt files in the folder of each model.

```bash
bash run_train_2nd.sh
```

Similar to the 1st stage training, model weights and config files are saved in the `checkpoints` folder.  
The folder name containing the letters "_finetune_" is the result of the 2nd stage.

### Inference

#### OOF inference

Output the predictions of the out of fold model used in the UEMU part. When you run this command, the `{exp_name}_logit.csv` file will be output to the submissions directory. (Other files are also output, but are not used)
Run the following command.

```bash
python inference_hms_1d_for_oofv1.py
python inference_hms_1d_for_oofv2.py
```

The output `{exp_name}_logit.csv` file is used for training the ensemble models(UEMU&kazumax&TH's part) of UEMU part.  
Change the file name to match the UEMU part according to the experiment name(`exp_name`).

exsample:

```bash
mv wavenet_effnetb4_downsample_logit.csv result_TH_wavenet_effnetb4_downsample_oof.csv
```

#### Test data infernce

see [Inference notebook](https://www.kaggle.com/code/asaliquid1011/hms-team-inference-ktmud)

If you want to inference on test data(`test.csv`) locally, run the following command.

```bash
python inference_hms_1d_for_test.py
```

When executed, a `submission.csv` file is output to the `./submissions` folder, which is an ensemble of 9 models trained in the T.H. part, achieving public LB = 0.250401 / private LB=0.301930.
