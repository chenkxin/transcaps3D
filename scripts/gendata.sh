# generate data and overlap them
prefix="/home/qiangzibro/caps3d/data/"

##r/r
#python gendata.py --output_file ${prefix}s2_mnist_r_r.gz &
#
##nr/nr
#python gendata.py --no_rotate_train --no_rotate_test \
    #	--output_file ${prefix}s2_mnist_nr_nr.gz &

#nr/r
python gendata.py --no_rotate_train \
    --output_file ${prefix}s2_mnist_nr_r.gz \
    --rotate_angles 0.3 0.4 0.5  &

wait
cd ..
python scripts/gen_overlapped_smnist.py