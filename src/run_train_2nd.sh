#! /bin/bash
gpu=0
logger_mode="online"

### wavenet_maxxvitv2n
checkpoints=(
    "../checkpoints/run-20240311_162443-wtgvtt7f-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_fold_0/epoch=48-val_metric_kldiv=0.506.ckpt"
    "../checkpoints/run-20240311_162558-52u6k3pz-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_fold_1/epoch=39-val_metric_kldiv=0.483.ckpt"
    "../checkpoints/run-20240311_162601-19rayuro-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_fold_2/epoch=43-val_metric_kldiv=0.531.ckpt"
    "../checkpoints/run-20240311_162604-vhwn8ok2-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_fold_3/epoch=43-val_metric_kldiv=0.501.ckpt"
    )
for i in ${!checkpoints[@]}
do
python train_hms_1D.py -o gpu=$gpu seed=42 dataset.fold=$i logger.mode=$logger_mode \
    logger.runName="1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_finetune" \
    model.args.encoder_type="wavenet" model.args.extracter_type="cnn2d" model.args.extracter_backbone="maxxvitv2_nano_rw_256" \
    model.args.extracter_pretrained=True train.epoch=30 optimizer.scheduler.args.first_cycle_steps=30 \
    model.args.encoder_output_size=256 datamodule.batch_size=32 \
    model.args.wavenet_params.downsample=False \
    dataset.feature_type="standard" model.args.feature_type="standard" \
    model.load_checkpoint=${checkpoints[$i]} \
    dataset.vote_min_thresh=10 \
    checkpoint.monitor=val_metric_kldiv_high_votes \
    optimizer.scheduler.args.warmup_steps=3 \
    optimizer.args.lr=1.e-4
done


### wavenet_maxxvitv2_downsample
checkpoints=(
    "../checkpoints/run-20240315_093258-4vcdh3kr-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_fold_0/epoch=48-val_metric_kldiv=0.494.ckpt"
    "../checkpoints/run-20240315_093300-e1zliaye-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_fold_1/epoch=43-val_metric_kldiv=0.496.ckpt"
    "../checkpoints/run-20240315_093302-eh3bfg7h-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_fold_2/epoch=40-val_metric_kldiv=0.541.ckpt"
    "../checkpoints/run-20240315_093304-3206t2wv-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_fold_3/epoch=35-val_metric_kldiv=0.505.ckpt"
    )
for i in ${!checkpoints[@]}
do
python train_hms_1D.py -o gpu=$gpu seed=42 dataset.fold=$i logger.mode=$logger_mode \
    logger.runName="1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downsample_finetune" \
    model.args.encoder_type="wavenet" model.args.extracter_type="cnn2d" model.args.extracter_backbone="maxxvitv2_nano_rw_256" \
    model.args.extracter_pretrained=True train.epoch=30 optimizer.scheduler.args.first_cycle_steps=30 \
    model.args.encoder_output_size=256 datamodule.batch_size=32 \
    model.args.wavenet_params.downsample=True \
    dataset.feature_type="standard" model.args.feature_type="standard" \
    model.load_checkpoint=${checkpoints[$i]} \
    dataset.vote_min_thresh=10 \
    checkpoint.monitor=val_metric_kldiv_high_votes \
    optimizer.scheduler.args.warmup_steps=3 \
    optimizer.args.lr=1.e-4
done
### wavenet_maxxvitv2_downsample_seed0
checkpoints=(
    "../checkpoints/run-20240331_092207-wd9z8fgc-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_seed0_fold_0/epoch=47-val_metric_kldiv=0.498.ckpt"
    "../checkpoints/run-20240331_145603-qwo5bx04-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_seed0_fold_1/epoch=39-val_metric_kldiv=0.496.ckpt"
    "../checkpoints/run-20240331_202910-hmm3yowr-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_seed0_fold_2/epoch=31-val_metric_kldiv=0.583.ckpt"
    "../checkpoints/run-20240401_020514-w85fm98b-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_seed0_fold_3/epoch=45-val_metric_kldiv=0.496.ckpt"
    )
for i in ${!checkpoints[@]}
do
python train_hms_1D.py -o gpu=$gpu seed=0 dataset.fold=$i logger.mode=$logger_mode \
    logger.runName="1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downsample_seed0_finetune" \
    model.args.encoder_type="wavenet" model.args.extracter_type="cnn2d" model.args.extracter_backbone="maxxvitv2_nano_rw_256" \
    model.args.extracter_pretrained=True train.epoch=30 optimizer.scheduler.args.first_cycle_steps=30 \
    model.args.encoder_output_size=256 datamodule.batch_size=32 \
    model.args.wavenet_params.downsample=True \
    dataset.feature_type="standard" model.args.feature_type="standard" \
    model.load_checkpoint=${checkpoints[$i]} \
    dataset.vote_min_thresh=10 \
    checkpoint.monitor=val_metric_kldiv_high_votes \
    optimizer.scheduler.args.warmup_steps=3 \
    optimizer.args.lr=1.e-4
done
### wavenet_maxxvitv2_downsample_seed123
checkpoints=(
    "../checkpoints/run-20240331_092204-83irzg77-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_seed123_fold_0/epoch=48-val_metric_kldiv=0.499.ckpt"
    "../checkpoints/run-20240331_145639-riq6hh5i-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_seed123_fold_1/epoch=45-val_metric_kldiv=0.497.ckpt"
    "../checkpoints/run-20240331_203051-j6nfot42-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_seed123_fold_2/epoch=40-val_metric_kldiv=0.546.ckpt"
    "../checkpoints/run-20240401_020520-vra67vnh-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_seed123_fold_3/epoch=46-val_metric_kldiv=0.497.ckpt"
    )
for i in ${!checkpoints[@]}
do
python train_hms_1D.py -o gpu=$gpu seed=123 dataset.fold=$i logger.mode=$logger_mode \
    logger.runName="1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downsample_seed123_finetune" \
    model.args.encoder_type="wavenet" model.args.extracter_type="cnn2d" model.args.extracter_backbone="maxxvitv2_nano_rw_256" \
    model.args.extracter_pretrained=True train.epoch=30 optimizer.scheduler.args.first_cycle_steps=30 \
    model.args.encoder_output_size=256 datamodule.batch_size=32 \
    model.args.wavenet_params.downsample=True \
    dataset.feature_type="standard" model.args.feature_type="standard" \
    model.load_checkpoint=${checkpoints[$i]} \
    dataset.vote_min_thresh=10 \
    checkpoint.monitor=val_metric_kldiv_high_votes \
    optimizer.scheduler.args.warmup_steps=3 \
    optimizer.args.lr=1.e-4
done
### wavenet_maxxvits_downsample
checkpoints=(
    "../checkpoints/run-20240331_092730-shfd8ff5-1D_cls_RTpIcS_LS_Wavenet-maxxvits_1e-3_standard_e50_warmup_downwample_fold_0/epoch=29-val_metric_kldiv=0.505.ckpt"
    "../checkpoints/run-20240331_155222-zgqkmvju-1D_cls_RTpIcS_LS_Wavenet-maxxvits_1e-3_standard_e50_warmup_downwample_fold_1/epoch=14-val_metric_kldiv=0.493.ckpt"
    "../checkpoints/run-20240331_221452-l8lmiy5f-1D_cls_RTpIcS_LS_Wavenet-maxxvits_1e-3_standard_e50_warmup_downwample_fold_2/epoch=30-val_metric_kldiv=0.571.ckpt"
    "../checkpoints/run-20240401_044218-99wn003l-1D_cls_RTpIcS_LS_Wavenet-maxxvits_1e-3_standard_e50_warmup_downwample_fold_3/epoch=26-val_metric_kldiv=0.513.ckpt"
    )
for i in ${!checkpoints[@]}
do
python train_hms_1D.py -o gpu=$gpu seed=42 dataset.fold=$i logger.mode=$logger_mode \
    logger.runName="1D_cls_RTpIcS_LS_Wavenet-maxxvits_1e-3_standard_e50_warmup_downsample_finetune" \
    model.args.encoder_type="wavenet" model.args.extracter_type="cnn2d" model.args.extracter_backbone="maxxvit_rmlp_small_rw_256" \
    model.args.extracter_pretrained=True train.epoch=30 optimizer.scheduler.args.first_cycle_steps=30 \
    model.args.encoder_output_size=256 datamodule.batch_size=32 \
    model.args.wavenet_params.downsample=True \
    dataset.feature_type="standard" model.args.feature_type="standard" \
    model.load_checkpoint=${checkpoints[$i]} \
    dataset.vote_min_thresh=10 \
    checkpoint.monitor=val_metric_kldiv_high_votes \
    optimizer.scheduler.args.warmup_steps=3 \
    optimizer.args.lr=1.e-4
done
### wavenet_effnetb4_downsample
checkpoints=(
    "../checkpoints/run-20240331_093954-xmakusup-1D_cls_RTpIcS_LS_Wavenet-effnetb4_1e-3_standard_e50_warmup_downwample_fold_0/epoch=19-val_metric_kldiv=0.534.ckpt"
    "../checkpoints/run-20240331_151353-u3jqgek0-1D_cls_RTpIcS_LS_Wavenet-effnetb4_1e-3_standard_e50_warmup_downwample_fold_1/epoch=20-val_metric_kldiv=0.543.ckpt"
    "../checkpoints/run-20240331_204609-t825jlg5-1D_cls_RTpIcS_LS_Wavenet-effnetb4_1e-3_standard_e50_warmup_downwample_fold_2/epoch=17-val_metric_kldiv=0.589.ckpt"
    "../checkpoints/run-20240401_022450-mpyf84lx-1D_cls_RTpIcS_LS_Wavenet-effnetb4_1e-3_standard_e50_warmup_downwample_fold_3/epoch=14-val_metric_kldiv=0.551.ckpt"
    )
for i in ${!checkpoints[@]}
do
python train_hms_1D.py -o gpu=$gpu seed=42 dataset.fold=$i logger.mode=$logger_mode \
    logger.runName="1D_cls_RTpIcS_LS_Wavenet-effnetb4_1e-3_standard_e50_warmup_downsample_finetune" \
    model.args.encoder_type="wavenet" model.args.extracter_type="cnn2d" model.args.extracter_backbone="tf_efficientnet_b4_ns" \
    model.args.extracter_pretrained=True train.epoch=30 optimizer.scheduler.args.first_cycle_steps=30 \
    model.args.encoder_output_size=320 datamodule.batch_size=32 \
    model.args.wavenet_params.downsample=True \
    dataset.feature_type="standard" model.args.feature_type="standard" \
    model.load_checkpoint=${checkpoints[$i]} \
    dataset.vote_min_thresh=10 \
    checkpoint.monitor=val_metric_kldiv_high_votes \
    optimizer.scheduler.args.warmup_steps=3 \
    optimizer.args.lr=1.e-4
done

### wavenet_maxxvitv2_downsample_foldv2
checkpoints=(
    "../checkpoints/run-20240404_093006-hcrzbbd6-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_foldV2_fold_0/epoch=34-val_metric_kldiv=0.489.ckpt"
    "../checkpoints/run-20240404_145752-5ltl9fmz-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_foldV2_fold_1/epoch=46-val_metric_kldiv=0.516.ckpt"
    "../checkpoints/run-20240404_204001-ca7kjo4a-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_foldV2_fold_2/epoch=39-val_metric_kldiv=0.491.ckpt"
    "../checkpoints/run-20240405_020131-hledwno7-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_foldV2_fold_3/epoch=49-val_metric_kldiv=0.535.ckpt"
    )
for i in ${!checkpoints[@]}
do
python train_hms_1D.py -o gpu=$gpu seed=42 dataset.fold=$i logger.mode=$logger_mode \
    logger.runName="1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downsample_foldV2_finetune" \
    model.args.encoder_type="wavenet" model.args.extracter_type="cnn2d" model.args.extracter_backbone="maxxvitv2_nano_rw_256" \
    model.args.extracter_pretrained=True train.epoch=30 optimizer.scheduler.args.first_cycle_steps=30 \
    model.args.encoder_output_size=256 datamodule.batch_size=32 \
    model.args.wavenet_params.downsample=True \
    dataset.csv_path=../hms-harmful-brain-activity-classification/train_fold_irr_mark_v5.csv \
    dataset.feature_type="standard" model.args.feature_type="standard" \
    model.load_checkpoint=${checkpoints[$i]} \
    dataset.vote_min_thresh=10 \
    checkpoint.monitor=val_metric_kldiv_high_votes \
    optimizer.scheduler.args.warmup_steps=3 \
    optimizer.args.lr=1.e-4
done
### wavenet_maxxvits_downsample_foldv2
checkpoints=(
    "../checkpoints/run-20240404_093006-xk4w5c6h-1D_cls_RTpIcS_LS_Wavenet-maxxvits_1e-3_standard_e50_warmup_downwample_foldV2_fold_0/epoch=21-val_metric_kldiv=0.509.ckpt"
    "../checkpoints/run-20240404_155631-2bzb9zia-1D_cls_RTpIcS_LS_Wavenet-maxxvits_1e-3_standard_e50_warmup_downwample_foldV2_fold_1/epoch=24-val_metric_kldiv=0.514.ckpt"
    "../checkpoints/run-20240404_222049-avllzbz1-1D_cls_RTpIcS_LS_Wavenet-maxxvits_1e-3_standard_e50_warmup_downwample_foldV2_fold_2/epoch=25-val_metric_kldiv=0.505.ckpt"
    "../checkpoints/run-20240405_042525-hi4n1g20-1D_cls_RTpIcS_LS_Wavenet-maxxvits_1e-3_standard_e50_warmup_downwample_foldV2_fold_3/epoch=34-val_metric_kldiv=0.549.ckpt"
    )
for i in ${!checkpoints[@]}
do
python train_hms_1D.py -o gpu=$gpu seed=42 dataset.fold=$i logger.mode=$logger_mode \
    logger.runName="1D_cls_RTpIcS_LS_Wavenet-maxxvits_1e-3_standard_e50_warmup_downsample_foldV2_finetune" \
    model.args.encoder_type="wavenet" model.args.extracter_type="cnn2d" model.args.extracter_backbone="maxxvit_rmlp_small_rw_256" \
    model.args.extracter_pretrained=True train.epoch=30 optimizer.scheduler.args.first_cycle_steps=30 \
    model.args.encoder_output_size=256 datamodule.batch_size=32 \
    model.args.wavenet_params.downsample=True \
    dataset.csv_path=../hms-harmful-brain-activity-classification/train_fold_irr_mark_v5.csv \
    dataset.feature_type="standard" model.args.feature_type="standard" \
    model.load_checkpoint=${checkpoints[$i]} \
    dataset.vote_min_thresh=10 \
    checkpoint.monitor=val_metric_kldiv_high_votes \
    optimizer.scheduler.args.warmup_steps=3 \
    optimizer.args.lr=1.e-4
done
### wavenet_effnetb4_downsample_foldv2
checkpoints=(
    "../checkpoints/run-20240404_150430-6cspsrdh-1D_cls_RTpIcS_LS_Wavenet-effnetb4_1e-3_standard_e15_warmup_downwample_foldV2_fold_0/epoch=10-val_metric_kldiv=0.532.ckpt"
    "../checkpoints/run-20240404_170157-lf2uzpt4-1D_cls_RTpIcS_LS_Wavenet-effnetb4_1e-3_standard_e15_warmup_downwample_foldV2_fold_1/epoch=14-val_metric_kldiv=0.541.ckpt"
    "../checkpoints/run-20240404_190202-04eht0lr-1D_cls_RTpIcS_LS_Wavenet-effnetb4_1e-3_standard_e15_warmup_downwample_foldV2_fold_2/epoch=13-val_metric_kldiv=0.504.ckpt"
    "../checkpoints/run-20240404_205700-4yn4p2dk-1D_cls_RTpIcS_LS_Wavenet-effnetb4_1e-3_standard_e15_warmup_downwample_foldV2_fold_3/epoch=14-val_metric_kldiv=0.533.ckpt"
    )
for i in ${!checkpoints[@]}
do
python train_hms_1D.py -o gpu=$gpu seed=42 dataset.fold=$i logger.mode=$logger_mode \
    logger.runName="1D_cls_RTpIcS_LS_Wavenet-effnetb4_1e-3_standard_e50_warmup_downsample_foldV2_finetune" \
    model.args.encoder_type="wavenet" model.args.extracter_type="cnn2d" model.args.extracter_backbone="tf_efficientnet_b4_ns" \
    model.args.extracter_pretrained=True train.epoch=15 optimizer.scheduler.args.first_cycle_steps=15 \
    model.args.encoder_output_size=320 datamodule.batch_size=32 \
    model.args.wavenet_params.downsample=True \
    dataset.csv_path=../hms-harmful-brain-activity-classification/train_fold_irr_mark_v5.csv \
    train.amp=False \
    dataset.feature_type="standard" model.args.feature_type="standard" \
    model.load_checkpoint=${checkpoints[$i]} \
    dataset.vote_min_thresh=10 \
    checkpoint.monitor=val_metric_kldiv_high_votes \
    optimizer.scheduler.args.warmup_steps=3 \
    optimizer.args.lr=1.e-4
done
