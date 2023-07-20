#!/bin/bash

SRC_DIR="/mnt/mmtech01/usr/liuwenzhuo/code/HMR-LeReS-new-rm-crop/d_make_leres_dataset/plane_mask"
DST_DIR="/mnt/mmtech01/dataset/vision_text_pretrain/gta-im/FPS-5-addmask"

for dir in $(ls $SRC_DIR)
do
    cp -r $SRC_DIR/$dir/* $DST_DIR/$dir/
done
