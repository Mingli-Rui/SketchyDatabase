#!/bin/bash.sh

ulimit -n 10240

cd /Users/minglirui/gpt/SketchyDatabase

export TRAIN_PHOTO_ROOT=/Users/minglirui/gpt/SketchyDatabase/dataset/photo-train/tx_000000000000
export TRAIN_SKETCH_ROOT=/Users/minglirui/gpt/SketchyDatabase/dataset/sketch-triplet-train/tx_000000000000
export TEST_PHOTO_ROOT=/Users/minglirui/gpt/SketchyDatabase/dataset/photo-train/tx_000000000000
export TEST_SKETCH_ROOT=/Users/minglirui/gpt/SketchyDatabase/dataset/sketch-triplet-train/tx_000000000000
export MODEL_SAVE_PATH=/Users/minglirui/gpt/SketchyDatabase/dataset
export MODEL_ROOT_PATH=$MODEL_SAVE_PATH

python3 train.py \
	--photo_root /Users/minglirui/gpt/SketchyDatabase/dataset/photo-train/tx_000000000000 \
	--sketch_root /Users/minglirui/gpt/SketchyDatabase/dataset/sketch-triplet-train/tx_000000000000 \
	--batch_size 16 \
	--device cpu \
	--support_cuda false \
	--epochs 50 \
	--lr 0.00007 \
	--test true \
	--test_f 5 \
	--photo_test $TEST_PHOTO_ROOT \
	--sketch_test $TEST_SKETCH_ROOT \
	--save_model true \
	--save_dir $MODEL_SAVE_PATH \
	--vis false \
	--env caffe2torch_tripletloss \
	--fine_tune false \
	--model_root $MODEL_ROOT_PATH \
	--net resnet34 \
	--cat true
