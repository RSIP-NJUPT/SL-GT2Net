import argparse
import os
import os.path as osp
import sys
import time
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")
import torch

import mmcv
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction

this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, ".."))

from mmseg.apis import multi_gpu_test, single_gpu_test, set_random_seed
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description="mmseg test (and eval) a model")
    parser.add_argument(
        "--config", default=r"configs/slgt2/slgt2_224x224_40k_brandenburg_s2.py", help="test config file path"
    )
    parser.add_argument(
        "--checkpoint",
        default=r"work_dirs/slgt2_224x224_40k_brandenburg_s2/train_20240609_021841/iter_40000.pth",
        help="checkpoint file",
    )
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument("--aug-test", action="store_true", help="Use Flip and Multi scale aug")
    parser.add_argument("--out", help="output result file in pickle format")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--deterministic", action="store_true", help="whether to set deterministic options for CUDNN backend."
    )
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Format the output results without perform evaluation. It is"
        "useful when you want to format the result to a specific format and "
        "submit it to the test server",
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        default=["mIoU", "mF1score"],
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes',
    )
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument("--show-dir", help="directory where painted images will be saved")
    parser.add_argument("--gpu-collect", action="store_true", help="whether to use gpu to collect results.")
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple "
        "workers, available when gpu_collect is not specified",
    )
    parser.add_argument("--options", nargs="+", action=DictAction, help="custom options")
    parser.add_argument("--eval-options", nargs="+", action=DictAction, help="custom options for evaluation")
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none", help="job launcher")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    # args.show_dir = r"./show_dir"
    assert args.out or args.eval or args.format_only or args.show or args.show_dir, (
        "Please specify at least one operation (save/eval/format/show the "
        'results / save the results) with the argument "--out", "--eval"'
        ', "--format-only", "--show" or "--show-dir"'
    )

    if args.eval and args.format_only:
        raise ValueError("--eval and --format_only cannot be both specified")

    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")

    cfg = mmcv.Config.fromfile(args.config)
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join("./work_dirs", osp.splitext(osp.basename(args.config))[0])

    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init the logger before other steps
    logger = None
    if args.eval:
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        timestamp = "test_" + timestamp
        cfg.work_dir = osp.join(cfg.work_dir, f"{timestamp}")
        # create work_dir
        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
        logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)
        if logger is not None:
            logger.info(f"Set random seed to {args.seed}, deterministic: " f"{args.deterministic}")
        else:
            print(f"Set random seed to {args.seed}, deterministic: " f"{args.deterministic}")

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=cfg.data.workers_per_gpu, dist=distributed, shuffle=False
    )

    # build the model and load checkpoint
    model = build_segmentor(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.CLASSES = checkpoint["meta"]["CLASSES"]
    model.PALETTE = checkpoint["meta"]["PALETTE"]

    args.show_dir = osp.join(cfg.work_dir, "show_dir")
    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(args.show_dir))
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir, background=cfg.background)
    else:
        model = MMDistributedDataParallel(
            model.cuda(), device_ids=[torch.cuda.current_device()], broadcast_buffers=False
        )
        outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f"\nwriting results to {args.out}")
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            dataset.evaluate(outputs, args.eval, logger, **kwargs)


if __name__ == "__main__":
    main()
