#! /usr/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

ARCH=$1

lengths=( 512 1024 2048 4096 8192 )

for len in "${lengths[@]}"
do
   bash $SCRIPT_DIR/length_extrapolation.sh $ARCH $len
done
