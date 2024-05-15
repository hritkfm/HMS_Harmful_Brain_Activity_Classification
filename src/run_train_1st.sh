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
epoch=50
limit_train_batches=1.0
if [ "${debug}" = "-d" ]; then
    echo "debug mode..."
    epoch=4
    limit_train_batches=3
fi



### wavenet_maxxvitv2n
for i in 0 1 2 3
do
python train_hms_1D.py -o gpu=$gpu seed=42 dataset.fold=$i logger.mode=$logger_mode \
    logger.runName="1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup" \
    train.limit_train_batches=$limit_train_batches \
    model.args.encoder_type="wavenet" model.args.extracter_type="cnn2d" model.args.extracter_backbone="maxxvitv2_nano_rw_256" \
    model.args.extracter_pretrained=True train.epoch=$epoch optimizer.scheduler.args.first_cycle_steps=$epoch \
    model.args.encoder_output_size=256 datamodule.batch_size=$batch_size \
    model.args.wavenet_params.downsample=False \
    dataset.feature_type="standard" model.args.feature_type="standard" \
    optimizer.scheduler.args.warmup_steps=3 \
    optimizer.args.lr=1.e-3
done
### wavenet_maxxvitv2_downsample
for i in 0 1 2 3
do
python train_hms_1D.py -o gpu=$gpu seed=42 dataset.fold=$i logger.mode=$logger_mode \
    logger.runName="1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downsample" \
    train.limit_train_batches=$limit_train_batches \
    model.args.encoder_type="wavenet" model.args.extracter_type="cnn2d" model.args.extracter_backbone="maxxvitv2_nano_rw_256" \
    model.args.extracter_pretrained=True train.epoch=$epoch optimizer.scheduler.args.first_cycle_steps=$epoch \
    model.args.encoder_output_size=256 datamodule.batch_size=$batch_size \
    model.args.wavenet_params.downsample=True \
    dataset.feature_type="standard" model.args.feature_type="standard" \
    optimizer.scheduler.args.warmup_steps=3 \
    optimizer.args.lr=1.e-3
done
### wavenet_maxxvitv2_downsample_seed0
for i in 0 1 2 3
do
python train_hms_1D.py -o gpu=$gpu seed=0 dataset.fold=$i logger.mode=$logger_mode \
    logger.runName="1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downsample_seed0" \
    train.limit_train_batches=$limit_train_batches \
    model.args.encoder_type="wavenet" model.args.extracter_type="cnn2d" model.args.extracter_backbone="maxxvitv2_nano_rw_256" \
    model.args.extracter_pretrained=True train.epoch=$epoch optimizer.scheduler.args.first_cycle_steps=$epoch \
    model.args.encoder_output_size=256 datamodule.batch_size=$batch_size \
    model.args.wavenet_params.downsample=True \
    dataset.feature_type="standard" model.args.feature_type="standard" \
    optimizer.scheduler.args.warmup_steps=3 \
    optimizer.args.lr=1.e-3
done
### wavenet_maxxvitv2_downsample_seed123
for i in 0 1 2 3
do
python train_hms_1D.py -o gpu=$gpu seed=123 dataset.fold=$i logger.mode=$logger_mode \
    logger.runName="1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downsample_seed123" \
    train.limit_train_batches=$limit_train_batches \
    model.args.encoder_type="wavenet" model.args.extracter_type="cnn2d" model.args.extracter_backbone="maxxvitv2_nano_rw_256" \
    model.args.extracter_pretrained=True train.epoch=$epoch optimizer.scheduler.args.first_cycle_steps=$epoch \
    model.args.encoder_output_size=256 datamodule.batch_size=$batch_size \
    model.args.wavenet_params.downsample=True \
    dataset.feature_type="standard" model.args.feature_type="standard" \
    optimizer.scheduler.args.warmup_steps=3 \
    optimizer.args.lr=1.e-3
done
### wavenet_maxxvits_downsample
for i in 0 1 2 3
do
python train_hms_1D.py -o gpu=$gpu seed=42 dataset.fold=$i logger.mode=$logger_mode \
    logger.runName="1D_cls_RTpIcS_LS_Wavenet-maxxvits_1e-3_standard_e50_warmup_downsample" \
    train.limit_train_batches=$limit_train_batches \
    model.args.encoder_type="wavenet" model.args.extracter_type="cnn2d" model.args.extracter_backbone="maxxvit_rmlp_small_rw_256" \
    model.args.extracter_pretrained=True train.epoch=$epoch optimizer.scheduler.args.first_cycle_steps=$epoch \
    model.args.encoder_output_size=256 datamodule.batch_size=$batch_size \
    model.args.wavenet_params.downsample=True \
    dataset.feature_type="standard" model.args.feature_type="standard" \
    optimizer.scheduler.args.warmup_steps=3 \
    optimizer.args.lr=1.e-3
done
### wavenet_effnetb4_downsample
for i in 0 1 2 3
do
python train_hms_1D.py -o gpu=$gpu seed=42 dataset.fold=$i logger.mode=$logger_mode \
    logger.runName="1D_cls_RTpIcS_LS_Wavenet-effnetb4_1e-3_standard_e50_warmup_downsample" \
    train.limit_train_batches=$limit_train_batches \
    model.args.encoder_type="wavenet" model.args.extracter_type="cnn2d" model.args.extracter_backbone="tf_efficientnet_b4_ns" \
    model.args.extracter_pretrained=True train.epoch=$epoch optimizer.scheduler.args.first_cycle_steps=$epoch \
    model.args.encoder_output_size=320 datamodule.batch_size=$batch_size \
    model.args.wavenet_params.downsample=True \
    dataset.feature_type="standard" model.args.feature_type="standard" \
    optimizer.scheduler.args.warmup_steps=3 \
    optimizer.args.lr=1.e-3
done

### wavenet_maxxvitv2_downsample_foldv2
for i in 0 1 2 3
do
python train_hms_1D.py -o gpu=$gpu seed=42 dataset.fold=$i logger.mode=$logger_mode \
    logger.runName="1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downsample_foldV2" \
    train.limit_train_batches=$limit_train_batches \
    model.args.encoder_type="wavenet" model.args.extracter_type="cnn2d" model.args.extracter_backbone="maxxvitv2_nano_rw_256" \
    model.args.extracter_pretrained=True train.epoch=$epoch optimizer.scheduler.args.first_cycle_steps=$epoch \
    model.args.encoder_output_size=256 datamodule.batch_size=$batch_size \
    model.args.wavenet_params.downsample=True \
    dataset.csv_path=../hms-harmful-brain-activity-classification/train_fold_irr_mark_v5.csv \
    dataset.feature_type="standard" model.args.feature_type="standard" \
    optimizer.scheduler.args.warmup_steps=3 \
    optimizer.args.lr=1.e-3
done
### wavenet_maxxvits_downsample_foldv2
for i in 0 1 2 3
do
python train_hms_1D.py -o gpu=$gpu seed=42 dataset.fold=$i logger.mode=$logger_mode \
    logger.runName="1D_cls_RTpIcS_LS_Wavenet-maxxvits_1e-3_standard_e50_warmup_downsample_foldV2" \
    train.limit_train_batches=$limit_train_batches \
    model.args.encoder_type="wavenet" model.args.extracter_type="cnn2d" model.args.extracter_backbone="maxxvit_rmlp_small_rw_256" \
    model.args.extracter_pretrained=True train.epoch=$epoch optimizer.scheduler.args.first_cycle_steps=$epoch \
    model.args.encoder_output_size=256 datamodule.batch_size=$batch_size \
    model.args.wavenet_params.downsample=True \
    dataset.csv_path=../hms-harmful-brain-activity-classification/train_fold_irr_mark_v5.csv \
    dataset.feature_type="standard" model.args.feature_type="standard" \
    optimizer.scheduler.args.warmup_steps=3 \
    optimizer.args.lr=1.e-3
done
### wavenet_effnetb4_downsample_foldv2
epoch=15
if [ "${debug}" = "-d" ]; then
    echo "debug mode..."
    epoch=4
fi
for i in 0 1 2 3
do
python train_hms_1D.py -o gpu=$gpu seed=42 dataset.fold=$i logger.mode=$logger_mode \
    logger.runName="1D_cls_RTpIcS_LS_Wavenet-effnetb4_1e-3_standard_e15_warmup_downsample_foldV2" \
    train.limit_train_batches=$limit_train_batches \
    model.args.encoder_type="wavenet" model.args.extracter_type="cnn2d" model.args.extracter_backbone="tf_efficientnet_b4_ns" \
    model.args.extracter_pretrained=True train.epoch=$epoch optimizer.scheduler.args.first_cycle_steps=$epoch \
    model.args.encoder_output_size=320 datamodule.batch_size=$batch_size \
    model.args.wavenet_params.downsample=True \
    dataset.csv_path=../hms-harmful-brain-activity-classification/train_fold_irr_mark_v5.csv \
    train.amp=False \
    dataset.feature_type="standard" model.args.feature_type="standard" \
    optimizer.scheduler.args.warmup_steps=3 \
    optimizer.args.lr=1.e-3
done
