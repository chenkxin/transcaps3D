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

TASK=smnist_task_short_bs32_r_r
python main.py --expr_name $TASK  \
    --model_name smnist \
    --dataset_name smnist \
    --epoch 50 --batch_size 32  \
    --learning_rate 5e-3 \
    --optimizer adam \
    --use_residual_block \
    --loss cross_entropy \
    --no_rotate_train \
    --continue_training \
    "$@"


# nr/nr smnist dataset
# our spherical capsule model
TASK=smnist_task_short_bs32_nr_nr
python main.py --expr_name $TASK  \
    --model_name smnist \
    --dataset_name smnist \
    --epoch 50 --batch_size 32  \
    --learning_rate 5e-3 \
    --optimizer adam \
    --use_residual_block \
    --loss cross_entropy \
    --no_rotate_train \
    --continue_training \
    "$@"

# nr/r smnist dataset
# our spherical capsule model
TASK=smnist_task_short_bs32_nr_r
python main.py --expr_name $TASK  \
    --model_name smnist \
    --dataset_name smnist \
    --epoch 50 --batch_size 32  \
    --learning_rate 5e-3 \
    --optimizer adam \
    --use_residual_block \
    --loss cross_entropy \
    --no_rotate_train \
    --continue_training \
    "$@"


# baseline
TASK=smnist_task5
python main.py --expr_name $TASK  \
    --model_name smnist_baseline \
    --dataset_name smnist \
    --epoch 50 --batch_size 32  \
    --learning_rate 5e-3 \
    --optimizer adam \
    --use_residual_block \
    --loss cross_entropy \
    --no_rotate_train \
    --continue_training \
    "$@"

# baseline
TASK=smnist_task6
python main.py --expr_name $TASK  \
    --model_name smnist_baseline_deep \
    --dataset_name smnist \
    --epoch 50 --batch_size 32  \
    --learning_rate 5e-3 \
    --optimizer adam \
    --use_residual_block \
    --loss cross_entropy \
    --no_rotate_train \
    --continue_training \
    "$@"
