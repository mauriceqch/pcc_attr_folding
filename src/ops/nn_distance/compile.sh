TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
NSYNC_INC=$TF_INC/external/nsync/public
$CUDA_HOME/bin/nvcc -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu \
-I$TF_INC -I$NSYNC_INC -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2
g++ -std=c++11 -shared -o tf_nndistance.so tf_nndistance.cpp \
tf_nndistance_g.cu.o -fPIC -I $TF_INC -I$NSYNC_INC -L$TF_LIB -L/usr/local/cuda/targets/x86_64-linux/lib -ltensorflow_framework -O2


$CUDA_HOME/bin/nvcc -std=c++11 -c -o tf_nndistance2_g.cu.o tf_nndistance2_g.cu \
-I$TF_INC -I$NSYNC_INC -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2
g++ -std=c++11 -shared -o tf_nndistance2.so tf_nndistance2.cpp \
tf_nndistance2_g.cu.o -fPIC -I $TF_INC -I$NSYNC_INC -L$TF_LIB -L/usr/local/cuda/targets/x86_64-linux/lib -ltensorflow_framework -O2
