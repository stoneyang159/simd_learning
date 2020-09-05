#!/usr/bin/env bash

# image_io


# cd image_io/
# sh build.sh $1 && cd ../

# # infer_engine

cd infer_engine/
sh build.sh $1 && cd ../

# # cap
cd video_cap/
sh build.sh $1 && cd  ../

## core 
cd core/
sh build.sh $1 && cd ../

## install, simply use cp
if [-d "product/lib/"];then
rm -rf product/lib/
fi

if [-d "product/include/"];then
rm -rf product/include/
fi
mkdir product/lib/
mkdir product/include/
mkdir product/include/video_cap/

cp video_cap/product/lib/*   product/lib/
cp infer_engine/product/lib/*   product/lib/
cp core/product/lib/*   product/lib/

cp video_cap/include/*  product/include/video_cap/
cp core/include/pose_fer_manager.h product/include/
cp core/include/common.h  product/include/

## test
#cd tests/
#sh build.sh $1 && cd ../

## app
cd app/
sh build.sh $1 && cd ../
