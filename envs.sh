export MACA_PATH=/opt/maca

export MACA_CLANG_PATH=${MACA_PATH}/mxgpu_llvm/bin
export MACA_CLANG=${MACA_PATH}/mxgpu_llvm
export DEVINFO_ROOT=${MACA_PATH}
export CUDA_PATH=${MACA_PATH}/tools/cu-bridge 
export CUDA_HOME=${CUDA_PATH}
export PATH=${MACA_PATH}/bin:${MACA_CLANG}/bin:${PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}
#MACA-PyTorch envs
export ISU_FASTMODEL=1 # must be set, otherwise may induce precision error
export USE_TDUMP=OFF # optional, use to control whether generating debug file
export TMEM_LOG=OFF # optional, use to control whether generating debug file
export DEBUG_ITRACE=0 # optional, use to control whether generating debug file

export MACA_SMALL_PAGESIZE_ENABLE=1
export MALLOC_THRESHOLD=99

#export MCCL_MIN_NCHANNELS=12
#export MCCL_MAX_NCHANNELS=16

export MCCL_P2P_LEVEL=SYS
export MCCL_LIMIT_RING_LL_THREADTHRESHOLDS=1

#export MHA_USE_BLAS=ON
#export MHA_BWD_NO_ATOMIC_F64=1
