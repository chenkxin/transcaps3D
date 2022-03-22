# -*- coding: utf-8 -*-
import warnings
import gc
from tensorboardX import SummaryWriter

from lib.dataset import makeDataLoader
from lib.loss import makeLoss
from lib.models import makeModel
from lib.optimizier import makeOptimizer
from lib.utils import *

warnings.filterwarnings("ignore")



def main(config):
    epoch = config.epoch
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    expr_name = config.expr_name
    model_name = config.model_name
    is_distributed = config.is_distributed
    dataset_name = config.dataset_name
    use_residual_block = config.use_residual_block
    bw = get_bw(config)
    config.model_dir = model_dir = check_dir(expr_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    load = args.continue_training or config.evaluate
    early_stopper = EarlyStopping(patience=10)
    N, classes, trainloader = makeDataLoader(
        True,
        bw,
        batch_size=batch_size,
        dataset_name=dataset_name,
        type=config.type,
        config=config,
    )

    Ntest, _, testloader = makeDataLoader(
        False,
        bw,
        batch_size=batch_size,
        dataset_name=dataset_name,
        type=config.type,
        config=config,
    )
    LAST_EPOCH, model = makeModel(
        model_name,
        model_dir,
        nclass=len(classes),
        device=device,
        is_distributed=is_distributed,
        use_residual_block=use_residual_block,
        load=load,
        config=config,
    )
    # print(f"Last epoch:{LAST_EPOCH},\nmodel:{model}")


    # loss
    criterion = makeLoss(config.loss, nclass=len(classes))

    # optimizer
    optimizer = makeOptimizer(
        config.optimizer,
        model,
        learning_rate,
        continue_training=load,
        model_dir=model_dir,
    )

    # learning rate
    if config.decay:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.decay_step_size,
            gamma=config.decay_gamma,
            last_epoch=LAST_EPOCH,
        )
    elif config.exponential:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.exp_gamma, last_epoch=LAST_EPOCH
        )
    else:
        scheduler = None

    def train():
        best_acc = 0
        writer = SummaryWriter(os.path.join(model_dir, "tensorboard"))

        for e in range(LAST_EPOCH + 1, epoch):
            # if early_stopper.early_stop:
            #     break
            total_correct = 0
            train_loss = 0
            pbar = tqdm(trainloader)
            for i, (X, y) in enumerate(pbar):
                loss, correct = train_step(
                    model, optimizer, criterion, X, y, device, config
                )

                total_correct += correct
                train_loss += loss
                # print(f"[Epoch {e+1}|{i+1}/{BATCHES} Training],loss:{loss},lr:{get_learning_rate(e+1, learning_rate)}")
                des = "[Epoch {}|Training],loss:{:.2f},lr:{:.5f}".format(
                    e + 1, loss, optimizer.param_groups[0]["lr"]
                )
                pbar.set_description(des)
                if config.debug:
                    break
            if scheduler:
                scheduler.step(e)
            # train acc log
            train_acc = total_correct / N
            # test acc log
            result_eval = evaluate(
                model,
                criterion,
                testloader,
                device,
                config,
                nclass=len(classes),
                plot=False,
            )
            test_loss, test_acc = result_eval.loss, result_eval.acc
            early_stopper(test_loss)
            print("[Epoch {} Train] <ACC>={:2}".format(e + 1, train_acc))
            print(
                "[Epoch {} Test] <ACC>={:2} <LOSS>={:2}".format(
                    e + 1, test_acc, test_loss
                )
            )

            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": e,
            }
            # best test acc log
            if test_acc > best_acc:
                # print(f"Get best score after {e+1} epochs, saving model.")
                best_acc = test_acc
                param_save(config, state, best_acc)

            if e + 1 % 5 == 0:
                torch.save(
                    state, os.path.join("logs", expr_name, f"{model_name}_{e + 1}.ckpt")
                )

            try:
                writer.add_scalars(
                    f"{expr_name}/loss",
                    {"train": train_loss / N, "test": test_loss / Ntest},
                    e + 1,
                )
                writer.add_scalars(
                    f"{expr_name}/acc", {"train": train_acc, "test": test_acc}, e + 1
                )

                writer.add_scalar(f"{expr_name}/best_acc", best_acc, e + 1)
                writer.close()
            except Exception as e:
                pass
            gc.collect()

    def eva():
        global test_acc
        if config.evaluate and LAST_EPOCH < 0:
            raise Exception(f"No model found for {expr_name}")
        test_type = "no_rotate" if config.type == "rotate" else "rotate"
        Ntest, _, testloader = makeDataLoader(
            False,
            bw,
            batch_size=batch_size,
            dataset_name=dataset_name,
            type=test_type,
            config=config,
        )

        LAST_eva_EPOCH, model = makeModel(
            model_name,
            model_dir,
            nclass=len(classes),
            device=device,
            is_distributed=is_distributed,
            use_residual_block=use_residual_block,
            load=True,
            config=config,
        )

        result_eval = evaluate(
            model,
            criterion,
            testloader,
            device,
            config=config,
            plot=False,
        )
        test_loss, test_acc, acc_for_every_cls = (
            result_eval.loss,
            result_eval.acc,
            result_eval.acc_for_every_cls,
        )
        print(
            "--------------------------------------------------------------------------------"
        )
        print(
            "[{}]\n {}: <total acc>={:2}\n <acc_for_every_cls>={}".format(
                expr_name, test_type, test_acc, acc_for_every_cls
            )
        )
        print(
            "--------------------------------------------------------------------------------"
        )
        if not config.evaluate:
            config.eval_acc = test_acc
            save_info(config)

    if config.evaluate:
        eva()
    else:
        train()
        eva()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--expr_name", type=str, default="capsule_v1")
    parser.add_argument("--evaluate", default=False, action="store_true")

    # Model options
    parser.add_argument(
        "--model_name",
        type=str,
        default="caps",
        choices=[
            "caps",
            "baseline",
            "resnet",
            "smnist",
            "smnist_baseline",
            "smnist_baseline_deep",
            "msvc",
            "msvc_caps",
        ],
    )
    parser.add_argument(
        "--routing",
        type=str,
        default="average",
        choices=["average", "degree_score", "so3_transformer"],
    )
    # whether to use residual block, only for capsule version now
    parser.add_argument(
        "--use_residual_block",
        default=False,
        action="store_true",
        help="use residual block with S2/SO3",
    )

    # Dataset options
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="modelnet10",
        choices=[
            "modelnet10",
            "modelnet40",
            "shrec15_0.2",
            "shrec15_0.3",
            "shrec15_0.4",
            "shrec15_0.5",
            "shrec15_0.6",
            "shrec15_0.7",
            "shrec17",
            "smnist",
        ],
    )
    parser.add_argument("--type", default="rotate", choices=["rotate", "no_rotate"])
    parser.add_argument("--pick_randomly", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--no_rotate_train", default=False, action="store_true")
    parser.add_argument("--no_rotate_test", default=False, action="store_true")
    parser.add_argument(
        "--overlap",
        default=False,
        action="store_true",
        help="use overlapped data to test",
    )
    # for multi-scale input
    parser.add_argument("--bandwidths", nargs="+", type=int, default=[32])

    # training options
    parser.add_argument("--learning_rate", type=float, default=0.005)
    parser.add_argument("--exponential", default=False, action="store_true")
    parser.add_argument("--exp_gamma", type=float, default=0.9)
    parser.add_argument("--decay", default=False, action="store_true")
    parser.add_argument("--decay_step_size", type=int, default=25)
    parser.add_argument("--decay_gamma", type=float, default=0.7)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument(
        "--loss",
        type=str,
        default="nll",
        choices=["nll", "cross_entropy", "CapsuleRecon", "MarginLoss"],
    )
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--continue_training", default=False, action="store_true")
    parser.add_argument(
        "--is_distributed",
        default=False,
        action="store_true",
        help="distributed training",
    )

    args = parser.parse_args()
    add_extra_info(args)
    if not args.evaluate:
        param_save(args)
        print(args)
    main(args)
