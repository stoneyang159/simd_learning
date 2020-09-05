#cd image_io/
#sh build.sh $1 && cd ../



rm -r build
mkdir build && cd build
platform=$1
cross="cross"
echo "$platform"
if [ "$platform" = "$cross" ]

then
    cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake\
    -DCMAKE_BUILD_TYPE=MinSizeRel -DBUILD_SHARED_LIBS=OFF ../
else
    cmake -DCMAKE_BUILD_TYPE=MinSizeRel -DBUILD_SHARED_LIBS=OFF ../
fi
make -j4 VERBOSE=1
cd  ../

