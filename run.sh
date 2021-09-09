#!/bin/sh
envname=$1
method=$2 # cher | eher
seed=$3
num_env=1
num_epoch=50

time=2
tag=$(date "+%Y%m%d%H%M%S")

export CUDA_VISIBLE_DEVICES=0
python -m baselines.${method}.experiment.train --env_name ${envname} --seed ${seed} --n_epochs ${num_epoch} --num_env ${num_env} > ~/logs/${envname}_${tag}_0.out 2> ~/logs/${envname}_${tag}_0.err &
sleep ${time}


# ps -ef | grep ${envname} | awk '{print $2}'| xargs kill -9

# Point2DLargeEnv-v1, Point2D-FourRoom-v1, SawyerReachXYEnv-v1, FetchReach-v1, Reacher-v2
# PointMassEmptyEnv-v1, PointMassWallEnv-v1, PointMassRoomsEnv-v1
# FetchSlide-v1, FetchPush-v1, FetchPickAndPlace-v1, SawyerDoorFixEnv-v1, HandReach-v0
# HandManipulateBlock-v0, HandManipulateBlockFull-v0, HandManipulateBlockRotateParallel-v0, HandManipulateBlockRotateXYZ-v0, HandManipulateBlockRotateZ-v0
# HandManipulateEgg-v0, HandManipulateEggFull-v0, HandManipulateEggRotate-v0
# HandManipulatePen-v0, HandManipulatePenFull-v0, HandManipulatePenRotate-v0
