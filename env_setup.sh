apt-get update
apt install libglib2.0-0 libglib2.0-dev
apt-get install libglib2.0-0
apt-get install libgl1-mesa-glx
ldconfig -p | grep libGLX
ldconfig

# cudnn
find /usr -name libcudnn_ops.so.9 
cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2  # Check the cuDNN version
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH 
