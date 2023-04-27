#! /usr/bin/bash
#arch=laxtnn_blm_decay_99
#arch=laxtnn_blm_sns_tiny
#wandb="spikes-n-sines-bidirectional"

#arch=ski_blm_tiny
#wandb="ski-bidirectional"

arch=ski_blm_inv_time
wandb="mlp-free-bidirectional"

TOKENS_PER_SAMPLE=512  # Max sequence length
#TOKENS_PER_SAMPLE=2048  # Max sequence length

GPUS=1

# change to your data dir
DATA_DIR=data-bin/wikitext-103
ARCH=$arch
TOTAL_UPDATES=50000    # Total number of training steps
WARMUP_UPDATES=3000    # Warmup the learning rate over this many updates    
MAX_SENTENCES=1
PEAK_LR=0.0005         # Peak learning rate, adjust as needed
CLIP_NORM=1.0
PORT=$(( $RANDOM + 2000 ))
prefix=roberta
UPDATE_FREQ=$(( 512 / $MAX_SENTENCES / $GPUS ))

fairseq-train $DATA_DIR \
    --user-dir laxtnn \
    --task masked_lm \
    --wandb-project $wandb \
    --criterion masked_lm \
    --distributed-world-size $GPUS  --distributed-port $PORT \
    --save-dir checkpoints/$prefix/$ARCH \
    --arch $ARCH --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm $CLIP_NORM \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.2 \
    --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --ddp-backend=legacy_ddp \
    --find-unused-parameters \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 1  2>&1 | tee $ARCH.log
