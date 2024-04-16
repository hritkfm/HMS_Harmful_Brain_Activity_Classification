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

```
docker build ./docker -t kaggle:hms
```

### Data download

Please download competition data from [kaggle](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/data) and put them to `./hms-harmful-brain-activity-classification` and unzip them.

### Prepare_data

```
cd src
python prepare_data.py
```

This will create `train_fold_irr_mark_v2.csv` and `train_fold_irr_mark_v5.csv` in the `./hms-harmful-brain-activity-classification`

### Train

Run the first stage learning script.
This will train 9 models.

```
bash run_train_1st.sh
```

※ _Note that this code uses wandb._

When the 1st stage learning is completed, a checkpoints folder will be created in the root directory of this project.The weights of the model learned in the first stage and the config file are saved in it.

Before executing 2nd stage learning, it is necessary to specify the model weight path.
Write the checkpoint path for each model in checkpoints list of `run_train_2nd.sh`.  
Example:

```
### wavenet_maxxvitv2
checkpoints=(
    "../checkpoints/run-20240311_162443-wtgvtt7f-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_fold_0/epoch=48-val_metric_kldiv=0.506.ckpt"
    "../checkpoints/run-20240311_162558-52u6k3pz-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_fold_1/epoch=39-val_metric_kldiv=0.483.ckpt"
    "../checkpoints/run-20240311_162601-19rayuro-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_fold_2/epoch=43-val_metric_kldiv=0.531.ckpt"
    "../checkpoints/run-20240311_162604-vhwn8ok2-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_fold_3/epoch=43-val_metric_kldiv=0.501.ckpt"
    )
```

※ _Please write the path of the model corresponding to the comment._  
※ _Write in order from fold0 to fold4._

Run the 2nd stage learning script.

```
bash run_train_2nd.sh
```

### Inference

Since UEMU was responsible for the model ensemble for our team, our main objective here is to create a file to pass to UEMU. Create a file containing logit values in the same format as the sample submission for each model.
submission.csv is also created at the same time, which is an average ensemble of 9 models, public LB = 0.250401/ purivate LB=0.301930.

- Test  
  Run the following command.

```
python inference_hms_1d.py
```

A file called `{model_name}_logit.csv` will be created. Use these in the UEMU part.

- Validation  
  it is necessary to modify and execute the code as shown below in order to output the OOF logit value of the training data.

  - fold ver1

    - Change to `MODE="val"`.
    - Change `CFG` parameters
      - Comment out the last three models written in `checkpoints`. (Because only the first 6 models learned with fold1 are used)
      - Change `datasets` for validation: fold v1
      ```
        # test:
        # datasets: List = field(default_factory=lambda: [HMS1DDataset] * 9)
        # validation: fold v1
        datasets: List = field(default_factory=lambda: [HMS1DDataset] * 6)
        # validation: fold v2
        # datasets: List = field(default_factory=lambda: [HMS1DDataset] * 3)
      ```
      - Change `model_weight` to validation: fold v1  
        Change it to the same as above using the comments as a reference.
      - Change `model_name` to validation: fold v1  
        Change it to the same as above using the comments as a reference.
    - Run the following command.

    ```
    python inference_hms_1d.py
    ```

  - fold ver2
    - Change to `MODE="val"`.
    - Change `CFG` parameters
      - Comment out the first six models written in `checkpoints`. (Because only the first 3 models learned with fold2 are used)
      - Change `datasets` for validation: fold v2
      ```
        # test:
        # datasets: List = field(default_factory=lambda: [HMS1DDataset] * 9)
        # validation: fold v1
        # datasets: List = field(default_factory=lambda: [HMS1DDataset] * 6)
        # validation: fold v2
        datasets: List = field(default_factory=lambda: [HMS1DDataset] * 3)
      ```
      - Change `model_weight` to validation: fold v2  
        Change it to the same as above using the comments as a reference.
      - Change `model_name` to validation: fold v2  
         Change it to the same as above using the comments as a reference.
    - Run the following command.
    ```
    python inference_hms_1d.py
    ```
