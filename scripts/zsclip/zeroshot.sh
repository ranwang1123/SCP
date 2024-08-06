DATA=/home/ranwang/Data/CoOp-main/datasets
TRAINER=ZeroshotCLIP
DATASET=$1
CFG=rn50  # rn50, rn101, vit_b32 or vit_b16

python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file ${DATASET} \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/${DATASET} \
--eval-only
