#! /usr/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

BATCH_SIZE=8
TOKENS_PER_SAMPLE=512
MAX_TOKEN=$((TOKENS_PER_SAMPLE*BATCH_SIZE))
DATA_DIR=$3
ARCH=$2
prefix=lm
MAX_UPDATE=50000
WARM_UP=4000
UPDATE_FREQ=$(( 128 / $BATCH_SIZE / $1 ))
PORT=$(( $RANDOM + 2000 ))
echo $PORT
LR=0.0005
CLIP_NORM=1.0
decay=0.2
if [ "$5" = true ]; then
    task=profiled_language_modeling
else
    task=language_modeling
fi

fairseq-train --task $task \
    $DATA_DIR \
    --user-dir ${SCRIPT_DIR}/.. \
    --wandb-project $4 \
    --save-dir checkpoints/$prefix/${ARCH} \
    --distributed-world-size $1  --distributed-port $PORT \
    --arch $ARCH --share-decoder-input-output-embed \
    --dropout 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay $decay --clip-norm $CLIP_NORM \
    --lr $LR --lr-scheduler inverse_sqrt --warmup-updates $WARM_UP --warmup-init-lr 1e-07 \
    --tokens-per-sample $TOKENS_PER_SAMPLE --sample-break-mode none \
    --max-tokens $MAX_TOKEN --update-freq $UPDATE_FREQ \
    --ddp-backend=legacy_ddp \
    --batch-size $BATCH_SIZE \
    --max-update $MAX_UPDATE --log-interval 10 2>&1 | tee $ARCH.log
