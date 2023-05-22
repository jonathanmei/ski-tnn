#! /usr/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

#arch=laxtnn_blm_baseline_3lyrs
#arch=ski_blm_inv_time
arch=laxtnn_blm_tno_fd
wandb=ski-tnn-blm
GPUS=1

TOKENS_PER_SAMPLE=512  # Max sequence length
MAX_SENTENCES=4
UPDATE_FREQ=$(( 512 / $MAX_SENTENCES / $GPUS ))

#TOKENS_PER_SAMPLE=2048  # Max sequence length
#MAX_SENTENCES=4
#UPDATE_FREQ=$(( 128 / $MAX_SENTENCES / $GPUS ))

# change to your data dir
DATA_DIR=${SCRIPT_DIR}/../../data-bin/wikitext-103
ARCH=$arch
TOTAL_UPDATES=50000    # Total number of training steps
WARMUP_UPDATES=3000    # Warmup the learning rate over this many updates    
TOKENS_PER_SAMPLE=512  # Max sequence length
MAX_POSITIONS=512     # Num. positional embeddings (usually same as above)

PEAK_LR=0.0005         # Peak learning rate, adjust as needed
CLIP_NORM=1.0
PORT=$(( $RANDOM + 2000 ))
prefix=roberta
UPDATE_FREQ=$(( 512 / $MAX_SENTENCES / $GPUS ))
NAME=$wandb
WANDB_PROJECT=$wandb

fairseq-train $DATA_DIR \
    --user-dir ${SCRIPT_DIR}/.. \
    --task masked_lm --criterion masked_lm \
    --distributed-world-size $GPUS  --distributed-port $PORT \
    --save-dir checkpoints/$prefix/$NAME \
    --arch $ARCH --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm $CLIP_NORM \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.2 \
    --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --wandb-project $WANDB_PROJECT \
    --ddp-backend=legacy_ddp \
    --find-unused-parameters \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 1  2>&1 | tee $ARCH.log
