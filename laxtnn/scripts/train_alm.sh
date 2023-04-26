#! /usr/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

#ARCH=laxtnn_decay_99_pre
#ARCH=laxtnn_sns_laplace
#ARCH=laxtnn_sns_laplace_small
#ARCH=laxtnn_sns_tiny
#wandb_proj=spikes-n-sines-causal

ARCH=ski_alm_tiny
wandb_proj=ski-causal

ARCH=tno_inv_time
wandb_proj=tno-inv-time
# change to your data dir
DATA_DIR=${SCRIPT_DIR}/../../data-bin/wikitext-103

n_gpu=1

profile=false
#profile=true

BATCH_SIZE=8
TOKENS_PER_SAMPLE=512
#BATCH_SIZE=2
#TOKENS_PER_SAMPLE=2048


#### These don't change as much
MAX_TOKEN=$((TOKENS_PER_SAMPLE*BATCH_SIZE))
prefix=lm
MAX_UPDATE=50000
WARM_UP=4000
UPDATE_FREQ=$(( 128 / $BATCH_SIZE / $n_gpu ))
PORT=$(( $RANDOM + 2000 ))
echo $PORT
LR=0.0005
CLIP_NORM=1.0
decay=0.2
if [ "$profile" = true ]; then
    task=profiled_language_modeling
else
    task=language_modeling
fi

fairseq-train --task $task \
    $DATA_DIR \
    --user-dir ${SCRIPT_DIR}/.. \
    --wandb-project $wandb_proj \
    --save-dir checkpoints/$prefix/${ARCH} \
    --distributed-world-size $n_gpu  --distributed-port $PORT \
    --arch $ARCH --share-decoder-input-output-embed \
    --dropout 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay $decay --clip-norm $CLIP_NORM \
    --lr $LR --lr-scheduler inverse_sqrt --warmup-updates $WARM_UP --warmup-init-lr 1e-07 \
    --tokens-per-sample $TOKENS_PER_SAMPLE --sample-break-mode none \
    --max-tokens $MAX_TOKEN --update-freq $UPDATE_FREQ \
    --ddp-backend=legacy_ddp \
    --batch-size $BATCH_SIZE \
    --max-update $MAX_UPDATE --log-interval 10 2>&1 | tee $ARCH.log
