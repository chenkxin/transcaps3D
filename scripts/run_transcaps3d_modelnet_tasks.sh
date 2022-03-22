# Compare our spherical capsule model with respect to s2cnn
# baseline:https://github.com/jonas-koehler/s2cnn/blob/master/examples/mnist/run.py
# - loss: CrossEntropy
# - optimizer: Adam
# - learning rate: 5e-3
# - learning rate scheduler: no
cat << EOF
--------------------------------------------------------------------------------
To see logs in tensorboard:

cd logs
tensorboard --log_dir . --host 10.22.75.213 --port 2333
--------------------------------------------------------------------------------


EOF
cd ..
# ModelNet Data
#TASK=modelet10_transcaps3d_b32_nr_nr_recon
#python main.py --expr_name $TASK  \
#    --model_name caps\
#    --dataset_name modelnet10\
#    --epoch 100 \
#    --batch_size 16  \
#    --learning_rate 5e-3 \
#    --optimizer adam \
#    --loss CapsuleRecon \
#    --routing so3_transformer\
#    --type no_rotate\
#    --use_residual_block\
#    --continue_training \
#    "$@"

#TASK=modelet10_transcaps3d_b32_r_r_recon
#python main.py --expr_name $TASK  \
#    --model_name caps\
#    --dataset_name modelnet10\
#    --epoch 200 \
#    --batch_size 16  \
#    --learning_rate 5e-3 \
#    --optimizer adam \
#    --loss CapsuleRecon \
#    --routing so3_transformer\
#    --type rotate\
#    --use_residual_block\
#    --continue_training \
#    "$@"

TASK=modelet40_transcaps3d_b32_r_r_recon
python main.py --expr_name $TASK  \
    --model_name caps\
    --dataset_name modelnet40\
    --epoch 200 \
    --batch_size 16  \
    --learning_rate 5e-3 \
    --optimizer adam \
    --loss CapsuleRecon \
    --routing so3_transformer\
    --type rotate\
    --use_residual_block\
    --continue_training \
    "$@"

TASK=modelet40_transcaps3d_b32_nr_nr_recon
python main.py --expr_name $TASK  \
    --model_name caps\
    --dataset_name modelnet40\
    --epoch 200 \
    --batch_size 16  \
    --learning_rate 5e-3 \
    --optimizer adam \
    --loss CapsuleRecon \
    --routing so3_transformer\
    --type no_rotate\
    --use_residual_block\
    --continue_training \
    "$@"


