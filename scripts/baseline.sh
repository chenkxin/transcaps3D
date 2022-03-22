#!/bin/bash
#
# This is a rather minimal example Argbash potential
# Example taken from http://argbash.readthedocs.io/en/stable/example.html
#
# ARG_OPTIONAL_SINGLE([datasets],[d],[])
# ARG_OPTIONAL_SINGLE([cmd_args],[c],[])
# ARG_HELP([MSVC train/eval script])
# ARGBASH_GO()
# needed because of Argbash --> m4_ignore([
### START OF CODE GENERATED BY Argbash v2.9.0 one line above ###
# Argbash is a bash code generator used to get arguments parsing right.
# Argbash is FREE SOFTWARE, see https://argbash.io for more info
# Generated online by https://argbash.io/generate


die()
{
    local _ret="${2:-1}"
    test "${_PRINT_HELP:-no}" = yes && print_help >&2
    echo "$1" >&2
    exit "${_ret}"
}


begins_with_short_option()
{
    local first_option all_short_options='dch'
    first_option="${1:0:1}"
    test "$all_short_options" = "${all_short_options/$first_option/}" && return 1 || return 0
}

# THE DEFAULTS INITIALIZATION - OPTIONALS
_arg_datasets=
_arg_cmd_args=
_arg_exp=baseline


print_help()
{
    printf '%s\n' "MSVC train/eval script"
    printf 'Usage: %s [-d|--datasets <arg>] [-c|--cmd_args <arg>] [-h|--help]\n' "$0"
    printf '\t%s\n' "-h, --help: Prints help"
}


parse_commandline()
{
    while test $# -gt 0
    do
        _key="$1"
        case "$_key" in
            -d|--datasets)
                test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
                _arg_datasets="$2"
                shift
                ;;
            --datasets=*)
                _arg_datasets="${_key##--datasets=}"
                ;;
            -d*)
                _arg_datasets="${_key##-d}"
                ;;
            -e|--exp)
                test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
                _arg_exp="$2"
                shift
                ;;
            --exp=*)
                _arg_exp="${_key##--exp=}"
                ;;
            -e*)
                _arg_exp="${_key##-d}"
                ;;
            -c|--cmd_args)
                test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
                _arg_cmd_args="$2"
                shift
                ;;
            --cmd_args=*)
                _arg_cmd_args="${_key##--cmd_args=}"
                ;;
            -c*)
                _arg_cmd_args="${_key##-c}"
                ;;
            -h|--help)
                print_help
                exit 0
                ;;
            -h*)
                print_help
                exit 0
                ;;
            *)
                _PRINT_HELP=yes die "FATAL ERROR: Got an unexpected argument '$1'" 1
                ;;
        esac
        shift
    done
}

parse_commandline "$@"

main()
{
    datasets=("modelnet10" "modelnet40" \
    "shrec15_0.2" "shrec15_0.3" "shrec15_0.4" "shrec15_0.5" "shrec15_0.6" "shrec15_0.7"\
        "shrec17" )
    batch_sizes=("3" "1"\
            "4" "4" "4" "4" "4" "4" \
        "8")

    for i in ${_arg_datasets}
    do
        dataset="${datasets[i]}"
        batch_size="${batch_sizes[i]}"

		x="$(echo $_arg_cmd_args | \
		 sed 's/--debug//g' |\
		 sed 's/--evaluate//g' |\
		 sed -E 's/--//g')"

		expr_name="$(echo $x | sed 's/ /_/g')"
        TASK=${_arg_exp}_${dataset}_${expr_name}
        args="
      --expr_name $TASK \
      --model_name baseline \
      --epoch 150 \
      --decay \
      --decay_step_size 25 \
      --decay_gamma  0.1 \
      --learning_rate 0.005 \
      --optimizer adam \
      --dataset_name ${dataset} \
      --batch_size ${batch_size} \
            ${_arg_cmd_args}
        "
		echo $args
    python main.py $args
  done
}
main
