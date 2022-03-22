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

# Shrec15 data

# train 80% test 20%

#TASK=shrec15_0.2_transcaps3d_b32_r_r_recon
#python main.py --expr_name $TASK  \
#    --model_name caps\
#    --dataset_name shrec15_0.2\
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
#
#TASK=shrec15_0.2_transcaps3d_b32_nr_nr_recon
#python main.py --expr_name $TASK  \
#    --model_name caps\
#    --dataset_name shrec15_0.2\
#    --epoch 200 \
#    --batch_size 16  \
#    --learning_rate 5e-3 \
#    --optimizer adam \
#    --loss CapsuleRecon \
#    --routing so3_transformer\
#    --type no_rotate\
#    --use_residual_block\
#    --continue_training \
#    "$@"

# train 70% test 30%

#TASK=shrec15_0.3_transcaps3d_b32_r_r_recon
#python main.py --expr_name $TASK  \
#    --model_name caps\
#    --dataset_name shrec15_0.3\
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

TASK=shrec15_0.3_transcaps3d_b32_nr_nr_recon
python main.py --expr_name $TASK  \
    --model_name caps\
    --dataset_name shrec15_0.3\
    --epoch 200 \
    --batch_size 2  \
    --learning_rate 5e-3 \
    --optimizer adam \
    --loss CapsuleRecon \
    --routing so3_transformer\
    --type no_rotate\
    --use_residual_block\
    --continue_training \
    "$@"

# train 60% test 40%

#TASK=shrec15_0.4_transcaps3d_b32_r_r_recon
#python main.py --expr_name $TASK  \
#    --model_name caps\
#    --dataset_name shrec15_0.4\
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

TASK=shrec15_0.4_transcaps3d_b32_nr_nr_recon
python main.py --expr_name $TASK  \
    --model_name caps\
    --dataset_name shrec15_0.4\
    --epoch 200 \
    --batch_size 2  \
    --learning_rate 5e-3 \
    --optimizer adam \
    --loss CapsuleRecon \
    --routing so3_transformer\
    --type no_rotate\
    --use_residual_block\
    --continue_training \
    "$@"

# train 50% test 50%

#TASK=shrec15_0.5_transcaps3d_b32_r_r_recon
#python main.py --expr_name $TASK  \
#    --model_name caps\
#    --dataset_name shrec15_0.5\
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

TASK=shrec15_0.5_transcaps3d_b32_nr_nr_recon
python main.py --expr_name $TASK  \
    --model_name caps\
    --dataset_name shrec15_0.5\
    --epoch 200 \
    --batch_size 2  \
    --learning_rate 5e-3 \
    --optimizer adam \
    --loss CapsuleRecon \
    --routing so3_transformer\
    --type no_rotate\
    --use_residual_block\
    --continue_training \
    "$@"

# train 60% test 40%
#
#TASK=shrec15_0.6_transcaps3d_b32_r_r_recon
#python main.py --expr_name $TASK  \
#    --model_name caps\
#    --dataset_name shrec15_0.6\
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

TASK=shrec15_0.6_transcaps3d_b32_nr_nr_recon
python main.py --expr_name $TASK  \
    --model_name caps\
    --dataset_name shrec15_0.6\
    --epoch 200 \
    --batch_size 2  \
    --learning_rate 5e-3 \
    --optimizer adam \
    --loss CapsuleRecon \
    --routing so3_transformer\
    --type no_rotate\
    --use_residual_block\
    --continue_training \
    "$@"

# train 30% test 70%
#
#TASK=shrec15_0.7_transcaps3d_b32_r_r_recon
#python main.py --expr_name $TASK  \
#    --model_name caps\
#    --dataset_name shrec15_0.7\
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

TASK=shrec15_0.7_transcaps3d_b32_nr_nr_recon
python main.py --expr_name $TASK  \
    --model_name caps\
    --dataset_name shrec15_0.7\
    --epoch 200 \
    --batch_size 2  \
    --learning_rate 5e-3 \
    --optimizer adam \
    --loss CapsuleRecon \
    --routing so3_transformer\
    --type no_rotate\
    --use_residual_block\
    --continue_training \
    "$@"