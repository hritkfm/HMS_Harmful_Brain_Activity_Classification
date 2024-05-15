#! /bin/bash
gpu=0
logger_mode="None"

debug=""  # defalult: disabled
while [ "$#" != 0 ]; do
  case $1 in
    -d | --debug      ) debug="-d";;
  esac
  shift
done

batch_size=32
epoch=30
limit_train_batches=1.0
if [ "${debug}" = "-d" ]; then
    echo "debug mode..."
    epoch=4
    limit_train_batches=3
fi

### wavenet_maxxvitv2n
checkpoints=(
    "../checkpoints/1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_fold_0/"
    "../checkpoints/1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_fold_1/"
    "../checkpoints/1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_fold_2/"
    "../checkpoints/1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_fold_3/"
    )
for i in ${!checkpoints[@]}
do
ckpt=`find ${checkpoints[$i]} -name "*.ckpt"`
python train_hms_1D.py -o gpu=$gpu seed=42 dataset.fold=$i logger.mode=$logger_mode \
    logger.runName="1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_finetune" \
    train.limit_train_batches=$limit_train_batches \
    model.args.encoder_type="wavenet" model.args.extracter_type="cnn2d" model.args.extracter_backbone="maxxvitv2_nano_rw_256" \
    model.args.extracter_pretrained=True train.epoch=$epoch optimizer.scheduler.args.first_cycle_steps=$epoch \
    model.args.encoder_output_size=256 datamodule.batch_size=$batch_size \
    model.args.wavenet_params.downsample=False \
    dataset.feature_type="standard" model.args.feature_type="standard" \
    model.load_checkpoint=$ckpt \
    dataset.vote_min_thresh=10 \
    checkpoint.monitor=val_metric_kldiv_high_votes \
    optimizer.scheduler.args.warmup_steps=3 \
    optimizer.args.lr=1.e-4
done


### wavenet_maxxvitv2_downsample
checkpoints=(
    "../checkpoints/1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_fold_0/"
    "../checkpoints/1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_fold_1/"
    "../checkpoints/1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_fold_2/"
    "../checkpoints/1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_fold_3/"
    )
for i in ${!checkpoints[@]}
do
ckpt=`find ${checkpoints[$i]} -name "*.ckpt"`
python train_hms_1D.py -o gpu=$gpu seed=42 dataset.fold=$i logger.mode=$logger_mode \
    logger.runName="1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downsample_finetune" \
    train.limit_train_batches=$limit_train_batches \
    model.args.encoder_type="wavenet" model.args.extracter_type="cnn2d" model.args.extracter_backbone="maxxvitv2_nano_rw_256" \
    model.args.extracter_pretrained=True train.epoch=$epoch optimizer.scheduler.args.first_cycle_steps=$epoch \
    model.args.encoder_output_size=256 datamodule.batch_size=$batch_size \
    model.args.wavenet_params.downsample=True \
    dataset.feature_type="standard" model.args.feature_type="standard" \
    model.load_checkpoint=$ckpt \
    dataset.vote_min_thresh=10 \
    checkpoint.monitor=val_metric_kldiv_high_votes \
    optimizer.scheduler.args.warmup_steps=3 \
    optimizer.args.lr=1.e-4
done
### wavenet_maxxvitv2_downsample_seed0
checkpoints=(
    "../checkpoints/1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_seed0_fold_0/"
    "../checkpoints/1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_seed0_fold_1/"
    "../checkpoints/1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_seed0_fold_2/"
    "../checkpoints/1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_seed0_fold_3/"
    )
for i in ${!checkpoints[@]}
do
ckpt=`find ${checkpoints[$i]} -name "*.ckpt"`
python train_hms_1D.py -o gpu=$gpu seed=0 dataset.fold=$i logger.mode=$logger_mode \
    logger.runName="1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downsample_seed0_finetune" \
    train.limit_train_batches=$limit_train_batches \
    model.args.encoder_type="wavenet" model.args.extracter_type="cnn2d" model.args.extracter_backbone="maxxvitv2_nano_rw_256" \
    model.args.extracter_pretrained=True train.epoch=$epoch optimizer.scheduler.args.first_cycle_steps=$epoch \
    model.args.encoder_output_size=256 datamodule.batch_size=$batch_size \
    model.args.wavenet_params.downsample=True \
    dataset.feature_type="standard" model.args.feature_type="standard" \
    model.load_checkpoint=$ckpt \
    dataset.vote_min_thresh=10 \
    checkpoint.monitor=val_metric_kldiv_high_votes \
    optimizer.scheduler.args.warmup_steps=3 \
    optimizer.args.lr=1.e-4
done
### wavenet_maxxvitv2_downsample_seed123
checkpoints=(
    "../checkpoints/1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_seed123_fold_0/"
    "../checkpoints/1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_seed123_fold_1/"
    "../checkpoints/1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_seed123_fold_2/"
    "../checkpoints/1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_seed123_fold_3/"
    )
for i in ${!checkpoints[@]}
do
ckpt=`find ${checkpoints[$i]} -name "*.ckpt"`
python train_hms_1D.py -o gpu=$gpu seed=123 dataset.fold=$i logger.mode=$logger_mode \
    logger.runName="1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downsample_seed123_finetune" \
    train.limit_train_batches=$limit_train_batches \
    model.args.encoder_type="wavenet" model.args.extracter_type="cnn2d" model.args.extracter_backbone="maxxvitv2_nano_rw_256" \
    model.args.extracter_pretrained=True train.epoch=$epoch optimizer.scheduler.args.first_cycle_steps=$epoch \
    model.args.encoder_output_size=256 datamodule.batch_size=$batch_size \
    model.args.wavenet_params.downsample=True \
    dataset.feature_type="standard" model.args.feature_type="standard" \
    model.load_checkpoint=$ckpt \
    dataset.vote_min_thresh=10 \
    checkpoint.monitor=val_metric_kldiv_high_votes \
    optimizer.scheduler.args.warmup_steps=3 \
    optimizer.args.lr=1.e-4
done
### wavenet_maxxvits_downsample
checkpoints=(
    "../checkpoints/1D_cls_RTpIcS_LS_Wavenet-maxxvits_1e-3_standard_e50_warmup_downwample_fold_0/"
    "../checkpoints/1D_cls_RTpIcS_LS_Wavenet-maxxvits_1e-3_standard_e50_warmup_downwample_fold_1/"
    "../checkpoints/1D_cls_RTpIcS_LS_Wavenet-maxxvits_1e-3_standard_e50_warmup_downwample_fold_2/"
    "../checkpoints/1D_cls_RTpIcS_LS_Wavenet-maxxvits_1e-3_standard_e50_warmup_downwample_fold_3/"
    )
for i in ${!checkpoints[@]}
do
ckpt=`find ${checkpoints[$i]} -name "*.ckpt"`
python train_hms_1D.py -o gpu=$gpu seed=42 dataset.fold=$i logger.mode=$logger_mode \
    logger.runName="1D_cls_RTpIcS_LS_Wavenet-maxxvits_1e-3_standard_e50_warmup_downsample_finetune" \
    train.limit_train_batches=$limit_train_batches \
    model.args.encoder_type="wavenet" model.args.extracter_type="cnn2d" model.args.extracter_backbone="maxxvit_rmlp_small_rw_256" \
    model.args.extracter_pretrained=True train.epoch=$epoch optimizer.scheduler.args.first_cycle_steps=$epoch \
    model.args.encoder_output_size=256 datamodule.batch_size=$batch_size \
    model.args.wavenet_params.downsample=True \
    dataset.feature_type="standard" model.args.feature_type="standard" \
    model.load_checkpoint=$ckpt \
    dataset.vote_min_thresh=10 \
    checkpoint.monitor=val_metric_kldiv_high_votes \
    optimizer.scheduler.args.warmup_steps=3 \
    optimizer.args.lr=1.e-4
done
### wavenet_effnetb4_downsample
checkpoints=(
    "../checkpoints/1D_cls_RTpIcS_LS_Wavenet-effnetb4_1e-3_standard_e50_warmup_downwample_fold_0/"
    "../checkpoints/1D_cls_RTpIcS_LS_Wavenet-effnetb4_1e-3_standard_e50_warmup_downwample_fold_1/"
    "../checkpoints/1D_cls_RTpIcS_LS_Wavenet-effnetb4_1e-3_standard_e50_warmup_downwample_fold_2/"
    "../checkpoints/1D_cls_RTpIcS_LS_Wavenet-effnetb4_1e-3_standard_e50_warmup_downwample_fold_3/"
    )
for i in ${!checkpoints[@]}
do
ckpt=`find ${checkpoints[$i]} -name "*.ckpt"`
python train_hms_1D.py -o gpu=$gpu seed=42 dataset.fold=$i logger.mode=$logger_mode \
    logger.runName="1D_cls_RTpIcS_LS_Wavenet-effnetb4_1e-3_standard_e50_warmup_downsample_finetune" \
    train.limit_train_batches=$limit_train_batches \
    model.args.encoder_type="wavenet" model.args.extracter_type="cnn2d" model.args.extracter_backbone="tf_efficientnet_b4_ns" \
    model.args.extracter_pretrained=True train.epoch=$epoch optimizer.scheduler.args.first_cycle_steps=$epoch \
    model.args.encoder_output_size=320 datamodule.batch_size=$batch_size \
    model.args.wavenet_params.downsample=True \
    dataset.feature_type="standard" model.args.feature_type="standard" \
    model.load_checkpoint=$ckpt \
    dataset.vote_min_thresh=10 \
    checkpoint.monitor=val_metric_kldiv_high_votes \
    optimizer.scheduler.args.warmup_steps=3 \
    optimizer.args.lr=1.e-4
done

### wavenet_maxxvitv2_downsample_foldv2
checkpoints=(
    "../checkpoints/1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_foldV2_fold_0/"
    "../checkpoints/1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_foldV2_fold_1/"
    "../checkpoints/1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_foldV2_fold_2/"
    "../checkpoints/1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_foldV2_fold_3/"
    )
for i in ${!checkpoints[@]}
do
ckpt=`find ${checkpoints[$i]} -name "*.ckpt"`
python train_hms_1D.py -o gpu=$gpu seed=42 dataset.fold=$i logger.mode=$logger_mode \
    logger.runName="1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downsample_foldV2_finetune" \
    train.limit_train_batches=$limit_train_batches \
    model.args.encoder_type="wavenet" model.args.extracter_type="cnn2d" model.args.extracter_backbone="maxxvitv2_nano_rw_256" \
    model.args.extracter_pretrained=True train.epoch=$epoch optimizer.scheduler.args.first_cycle_steps=$epoch \
    model.args.encoder_output_size=256 datamodule.batch_size=$batch_size \
    model.args.wavenet_params.downsample=True \
    dataset.csv_path=../hms-harmful-brain-activity-classification/train_fold_irr_mark_v5.csv \
    dataset.feature_type="standard" model.args.feature_type="standard" \
    model.load_checkpoint=$ckpt \
    dataset.vote_min_thresh=10 \
    checkpoint.monitor=val_metric_kldiv_high_votes \
    optimizer.scheduler.args.warmup_steps=3 \
    optimizer.args.lr=1.e-4
done
### wavenet_maxxvits_downsample_foldv2
checkpoints=(
    "../checkpoints/1D_cls_RTpIcS_LS_Wavenet-maxxvits_1e-3_standard_e50_warmup_downwample_foldV2_fold_0/"
    "../checkpoints/1D_cls_RTpIcS_LS_Wavenet-maxxvits_1e-3_standard_e50_warmup_downwample_foldV2_fold_1/"
    "../checkpoints/1D_cls_RTpIcS_LS_Wavenet-maxxvits_1e-3_standard_e50_warmup_downwample_foldV2_fold_2/"
    "../checkpoints/1D_cls_RTpIcS_LS_Wavenet-maxxvits_1e-3_standard_e50_warmup_downwample_foldV2_fold_3/"
    )
for i in ${!checkpoints[@]}
do
ckpt=`find ${checkpoints[$i]} -name "*.ckpt"`
python train_hms_1D.py -o gpu=$gpu seed=42 dataset.fold=$i logger.mode=$logger_mode \
    logger.runName="1D_cls_RTpIcS_LS_Wavenet-maxxvits_1e-3_standard_e50_warmup_downsample_foldV2_finetune" \
    train.limit_train_batches=$limit_train_batches \
    model.args.encoder_type="wavenet" model.args.extracter_type="cnn2d" model.args.extracter_backbone="maxxvit_rmlp_small_rw_256" \
    model.args.extracter_pretrained=True train.epoch=$epoch optimizer.scheduler.args.first_cycle_steps=$epoch \
    model.args.encoder_output_size=256 datamodule.batch_size=$batch_size \
    model.args.wavenet_params.downsample=True \
    dataset.csv_path=../hms-harmful-brain-activity-classification/train_fold_irr_mark_v5.csv \
    dataset.feature_type="standard" model.args.feature_type="standard" \
    model.load_checkpoint=$ckpt \
    dataset.vote_min_thresh=10 \
    checkpoint.monitor=val_metric_kldiv_high_votes \
    optimizer.scheduler.args.warmup_steps=3 \
    optimizer.args.lr=1.e-4
done
### wavenet_effnetb4_downsample_foldv2
epoch=15
if [ "${debug}" = "-d" ]; then
    echo "debug mode..."
    epoch=4
fi
checkpoints=(
    "../checkpoints/1D_cls_RTpIcS_LS_Wavenet-effnetb4_1e-3_standard_e15_warmup_downwample_foldV2_fold_0/"
    "../checkpoints/1D_cls_RTpIcS_LS_Wavenet-effnetb4_1e-3_standard_e15_warmup_downwample_foldV2_fold_1/"
    "../checkpoints/1D_cls_RTpIcS_LS_Wavenet-effnetb4_1e-3_standard_e15_warmup_downwample_foldV2_fold_2/"
    "../checkpoints/1D_cls_RTpIcS_LS_Wavenet-effnetb4_1e-3_standard_e15_warmup_downwample_foldV2_fold_3/"
    )
for i in ${!checkpoints[@]}
do
ckpt=`find ${checkpoints[$i]} -name "*.ckpt"`
python train_hms_1D.py -o gpu=$gpu seed=42 dataset.fold=$i logger.mode=$logger_mode \
    logger.runName="1D_cls_RTpIcS_LS_Wavenet-effnetb4_1e-3_standard_e50_warmup_downsample_foldV2_finetune" \
    train.limit_train_batches=$limit_train_batches \
    model.args.encoder_type="wavenet" model.args.extracter_type="cnn2d" model.args.extracter_backbone="tf_efficientnet_b4_ns" \
    model.args.extracter_pretrained=True train.epoch=$epoch optimizer.scheduler.args.first_cycle_steps=$epoch \
    model.args.encoder_output_size=320 datamodule.batch_size=$batch_size \
    model.args.wavenet_params.downsample=True \
    dataset.csv_path=../hms-harmful-brain-activity-classification/train_fold_irr_mark_v5.csv \
    train.amp=False \
    dataset.feature_type="standard" model.args.feature_type="standard" \
    model.load_checkpoint=$ckpt \
    dataset.vote_min_thresh=10 \
    checkpoint.monitor=val_metric_kldiv_high_votes \
    optimizer.scheduler.args.warmup_steps=3 \
    optimizer.args.lr=1.e-4
done
