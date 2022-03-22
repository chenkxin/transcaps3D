GPU0(){

	bash scripts/msvc.sh --datasets 0\
		--cmd_args "--type no_rotate --loss CapsuleRecon $@"


	bash scripts/msvc.sh --datasets 0\
		--cmd_args "--type no_rotate --loss MarginLoss $@"


	bash scripts/msvc.sh --datasets 0\
	--cmd_args "--type rotate --loss CapsuleRecon $@"
}

GPU1(){

	bash scripts/msvc.sh --datasets 0\
		--cmd_args "--type rotate --loss MarginLoss $@"


	bash scripts/msvc.sh --datasets 0\
		--cmd_args "--type rotate --pick_randomly --loss CapsuleRecon $@"


	bash scripts/msvc.sh --datasets 0\
		--cmd_args "--type rotate --pick_randomly --loss MarginLoss $@"
}

CUDA_VISIBLE_DEVICES=0 GPU0 "$@" &
CUDA_VISIBLE_DEVICES=1 GPU1 "$@" &

wait
echo "done!"
