

TRAINER=SCP
DATASET=$1  # "imagenet_a/imagenet_a/imagenet_a"
SEED=$2
DATA=$3

CFG=vit_b16_c2_ep50_batch4_4+4ctx_few_shot2
SHOTS=1

DIR=output/TTA/vitb16/${DATASET}

python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file ${DATASET} \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
--eval-only
