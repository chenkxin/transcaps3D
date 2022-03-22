# bash scripts/statistics.sh
cd data
for name in modelnet10 modelnet40\
	shrec15_0.2 shrec15_0.3 shrec15_0.4 shrec15_0.5 shrec15_0.6 shrec15_0.7 shrec17
do
    if [ -d "$name" ];then
        train_npys=""
        test_npys=""
        echo "--------------------------------------------------"
        for b in 32 16 8
        do
            train_off=`find $name/$name"_train/" -name "*.off" | wc -l`
            train_npy=`find $name/$name"_train/$1" -name "b$b*.npy" | wc -l`
            test_off=` find $name/$name"_test/" -name "*.off"   | wc -l`
            test_npy=` find $name/$name"_test/$1" -name "b$b*.npy"   | wc -l`
            train_npys="${train_npys}b=${b}:${train_npy}   "
            test_npys="${test_npys}b=${b}:${test_npy}   "
        done

        echo $name
        echo train: [off:$train_off/npy $train_npys]
        echo test: [off:$test_off/npy $test_npys]
        echo "--------------------------------------------------"
    fi
done
cd ..
