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
#TASK=modelet10_scnn_b32_r_r
#python main.py --expr_name $TASK  \
#    --model_name baseline \
#    --dataset_name modelnet10\
#    --epoch 100 \
#    --batch_size 32  \
#    --learning_rate 5e-3 \
#    --optimizer adam \
#    --loss cross_entropy \
#    --continue_training \
#    "$@"
#
#TASK=modelet10_scnn_b32_nr_nr
#python main.py --expr_name $TASK  \
#    --model_name baseline \
#    --dataset_name modelnet10\
#    --epoch 100 \
#    --batch_size 32  \
#    --learning_rate 5e-3 \
#    --optimizer adam \
#    --no_rotate_train \
#    --no_rotate_test \
#    --loss cross_entropy \
#    --continue_training \
#    "$@"
#
#TASK=modelet40_scnn_b32_r_r
#python main.py --expr_name $TASK  \
#    --model_name baseline \
#    --dataset_name modelnet40\
#    --epoch 100 \
#    --batch_size 32  \
#    --learning_rate 5e-3 \
#    --optimizer adam \
#    --loss cross_entropy \
#    --continue_training \
#    "$@"



TASK=shrec15_0.2_scnn_b32_r_r
python main.py --expr_name $TASK  \
    --model_name baseline \
    --dataset_name shrec15_0.2\
    --epoch 200 \
    --batch_size 32  \
    --learning_rate 5e-3 \
    --optimizer adam \
    --type rotate\
    --loss cross_entropy \
    --continue_training \
    "$@"

TASK=shrec15_0.2_scnn_b32_nr_nr
python main.py --expr_name $TASK  \
    --model_name baseline \
    --dataset_name shrec15_0.2\
    --epoch 200 \
    --batch_size 32  \
    --learning_rate 5e-3 \
    --type no_rotate\
    --optimizer adam \
    --loss cross_entropy \
    --continue_training \
    "$@"


TASK=shrec15_0.3_scnn_b32_r_r
python main.py --expr_name $TASK  \
    --model_name baseline \
    --dataset_name shrec15_0.3\
    --epoch 200 \
    --batch_size 32  \
    --learning_rate 5e-3 \
    --optimizer adam \
    --type rotate\
    --loss cross_entropy \
    --continue_training \
    "$@"

TASK=shrec15_0.3_scnn_b32_nr_nr
python main.py --expr_name $TASK  \
    --model_name baseline \
    --dataset_name shrec15_0.3\
    --epoch 200 \
    --batch_size 32  \
    --learning_rate 5e-3 \
    --type no_rotate\
    --optimizer adam \
    --loss cross_entropy \
    --continue_training \
    "$@"

TASK=shrec15_0.4_scnn_b32_r_r
python main.py --expr_name $TASK  \
    --model_name baseline \
    --dataset_name shrec15_0.4\
    --epoch 200 \
    --batch_size 32  \
    --learning_rate 5e-3 \
    --optimizer adam \
    --type rotate\
    --loss cross_entropy \
    --continue_training \
    "$@"

TASK=shrec15_0.4_scnn_b32_nr_nr
python main.py --expr_name $TASK  \
    --model_name baseline \
    --dataset_name shrec15_0.4\
    --epoch 200 \
    --batch_size 32  \
    --learning_rate 5e-3 \
    --type no_rotate\
    --optimizer adam \
    --loss cross_entropy \
    --continue_training \
    "$@"

TASK=shrec15_0.5_scnn_b32_r_r
python main.py --expr_name $TASK  \
    --model_name baseline \
    --dataset_name shrec15_0.5\
    --epoch 200 \
    --batch_size 32  \
    --learning_rate 5e-3 \
    --optimizer adam \
    --type rotate\
    --loss cross_entropy \
    --continue_training \
    "$@"

TASK=shrec15_0.5_scnn_b32_nr_nr
python main.py --expr_name $TASK  \
    --model_name baseline \
    --dataset_name shrec15_0.5\
    --epoch 200 \
    --batch_size 32  \
    --learning_rate 5e-3 \
    --type no_rotate\
    --optimizer adam \
    --loss cross_entropy \
    --continue_training \
    "$@"

TASK=shrec15_0.6_scnn_b32_r_r
python main.py --expr_name $TASK  \
    --model_name baseline \
    --dataset_name shrec15_0.6\
    --epoch 200 \
    --batch_size 32  \
    --learning_rate 5e-3 \
    --optimizer adam \
    --type rotate\
    --loss cross_entropy \
    --continue_training \
    "$@"

TASK=shrec15_0.6_scnn_b32_nr_nr
python main.py --expr_name $TASK  \
    --model_name baseline \
    --dataset_name shrec15_0.6 \
    --epoch 200 \
    --batch_size 32  \
    --learning_rate 5e-3 \
    --type no_rotate\
    --optimizer adam \
    --loss cross_entropy \
    --continue_training \
    "$@"

TASK=shrec15_0.7_scnn_b32_r_r
python main.py --expr_name $TASK  \
    --model_name baseline \
    --dataset_name shrec15_0.7\
    --epoch 200 \
    --batch_size 32  \
    --learning_rate 5e-3 \
    --optimizer adam \
    --type rotate\
    --loss cross_entropy \
    --continue_training \
    "$@"

TASK=shrec15_0.7_scnn_b32_nr_nr
python main.py --expr_name $TASK  \
    --model_name baseline \
    --dataset_name shrec15_0.7\
    --epoch 200 \
    --batch_size 32  \
    --learning_rate 5e-3 \
    --type no_rotate\
    --optimizer adam \
    --loss cross_entropy \
    --continue_training \
    "$@"


