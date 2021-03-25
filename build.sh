#!/bin/bash

tf_ver="1.15.2"
dst_dir="tensorflow_pkg/"
mkdir -p ${HOME}/px/${dst_dir}
dbg_opt=""
help(){
    echo "./build.sh [build/install/all] <-g>"
}

build(){
    echo "build tensorflow"
    bazel build ${dbg_opt} --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package --distdir /home/zhengzg/px/install_pkgs/dist
    # bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
}

install(){
    echo "install tensorflow"
    ./bazel-bin/tensorflow/tools/pip_package/build_pip_package ${HOME}/px/${dst_dir}    
    echo y > ${HOME}/px/y
    pip uninstall tensorflow < ${HOME}/px/y
    pip install ${HOME}/px/${dst_dir}/tensorflow-${tf_ver}-cp36-cp36mu-linux_x86_64.whl
}

all(){
    build
    install
}

if [ $# == 0 ];then
    echo "Please specify a action"
    help
    exit 1
fi
    
if [ $# == 2 ];then
    dbg_opt="--copt=-g -c dbg"
fi

if [ $1 == "build" ];then
    build
elif [ $1 == "install" ];then
    install
elif [ $1 == "all" ];then
    all
else
    help
    exit 1
fi    



