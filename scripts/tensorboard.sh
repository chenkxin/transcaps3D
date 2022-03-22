cd logs || exit 1
ip=`hostname --ip-address | awk -F ' ' '{print $1}'`
port=2333
echo "go to http://$ip:$port"
nohup tensorboard --logdir . --host $ip --port $port 2>&1 > /dev/null &
