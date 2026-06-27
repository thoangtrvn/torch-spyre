
export SPYRE_COMMS_KERNEL_DIR=/home/tmhoangt/spyre-multi/kernel_dir_for_spyre_comms/

# Pre-compile add kernels for device-side compute (must run before tests)
# Shapes: per-chunk sizes for ring allreduce + full tensor sizes
# For 4 ranks: chunk = total/4. For 2 ranks: chunk = total/2.
# 4096 elem (default) → 1024 elem/chunk (4-rank), 2048 elem/chunk (2-rank)
# 32768 elem (64KB) → 8192 elem/chunk (4-rank), 16384 elem/chunk (2-rank)
# 131072 elem (256KB) → 32768 elem/chunk (4-rank), 65536 elem/chunk (2-rank)
python3 examples/distributed/generate_add_kernels.py \
    --shape 1024 --shape 2048 --shape 4096 --shape 8192 --shape 16384 --shape 32768 --shape 65536 --shape 131072
#python3 examples/distributed/generate_add_kernels.py \
#    --shape 2048 --shape 4096 --shape 16384 --shape 32768 --shape 65536 --shape 131072 --dtype float32

#TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc-per-node $NUMSPYRES examples/distributed/broadcast.py 2>&1 | tee log_broadcast4.txt

export NUMSPYRES=4

rm -rf /tmp/torchinductor_tmhoangt/inductor-spyre/
##TORCH_DISTRIBUTED_DEBUG=DETAIL COLL_ALLREDUCE_ALGO=ReduceScatterAllGather torchrun --nproc-per-node $NUMSPYRES examples/distributed/allreduce.py 2>&1 | tee log_allreduce.txt
#TORCH_DISTRIBUTED_DEBUG=DETAIL COLL_ALLREDUCE_ALGO=PairwisePow2 torchrun --nproc-per-node $NUMSPYRES examples/distributed/allreduce.py 2>&1 | tee log_allreduce.txt
#cp /tmp/assignAddresses_rank* .

#TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc-per-node $NUMSPYRES examples/distributed/allgather.py 2>&1 | tee log_allgather_all.txt
#cp /tmp/assignAddresses_rank* .

TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc-per-node $NUMSPYRES examples/distributed/test_spyre_comms_primitives.py 2>&1 | tee log_primitives.txt
#cp /tmp/assignAddresses_rank* .

#LD_PRELOAD=/home/tmhoangt/spyre-multi/sentient/deeptools/lib/libutil.so torchrun --nproc-per-node $NUMSPYRES examples/distributed/allreduce.py

# the below should work
#TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc-per-node $NUMSPYRES examples/distributed/allreduce.py 2>&1 | tee log_allreduce_all.txt
#
#TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc-per-node $NUMSPYRES examples/distributed/reduce.py 2>&1 | tee log_reduce_all.txt
#
#TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc-per-node $NUMSPYRES examples/distributed/gather.py 2>&1 | tee log_gather.txt
#
#TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc-per-node $NUMSPYRES examples/distributed/allgather.py 2>&1 | tee log_allgather_all.txt


