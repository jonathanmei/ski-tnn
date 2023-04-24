#! /usr/bin/bash

batch_size=2
data_dir=data-bin/wikitext-103
ARCH=$1
ckpt=checkpoints/lm/$ARCH/checkpoint_best.pt
l=$2

fairseq-eval-lm \
    $data_dir \
    --user-dir laxtnn \
    --sample-break-mode none \
    --path $ckpt \
    --max-sentences 1 \
    --model-overrides "{'max_tokens':${l}, 'tokens_per_sample':${l}, 'max_target_positions':${l}}" \
    --max-tokens $l \
    --tokens-per-sample $l \
    --max-target-positions $l \
    --context-window 0 \
    --wandb-project tnn-fd \
    --log-interval 1

    # --model-overrides \"{'max_tokens':$l, 'tokens_per_sample':$l, 'max_target_positions':$l}\" \
