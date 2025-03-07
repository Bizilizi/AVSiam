import argparse
import os

os.environ["MPLCONFIGDIR"] = "./plt/"
import ast
import pickle
import sys
import time
import torch
from torch.utils.data import WeightedRandomSampler

basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)

### ----------> check if clip
import dataloader_ft as dataloader

# import dataloader_ft_clip as dataloader
### <-------

import models
import numpy as np
import warnings
import json
from sklearn import metrics
from traintest_ft_base import train, validate
from ipdb import set_trace
import random
import utils
from torch.nn.parallel import DistributedDataParallel as DDP
from seq_dataloader import SequentialDistributedSampler
from collections import OrderedDict
import wandb


from torch.utils.data import WeightedRandomSampler
from yb_sampler import DistributedProxySampler


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    return new_state_dict


# finetune cav-mae model

print(
    "I am process %s, running on %s: starting (%s)"
    % (os.getpid(), os.uname()[1], time.asctime())
)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_train", type=str, default="", help="training data json")
parser.add_argument("--data_val", type=str, default="", help="validation data json")
parser.add_argument("--data_eval", type=str, default=None, help="evaluation data json")
parser.add_argument("--label_csv", type=str, default="", help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")
parser.add_argument("--model", type=str, default="ast", help="the model used")
# parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used", choices=["audioset", "esc50", "speechcommands", "fsd50k", "vggsound", "epic", "k400"])
parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used")

parser.add_argument(
    "--dataset_mean", type=float, help="the dataset mean, used for input normalization"
)
parser.add_argument(
    "--dataset_std", type=float, help="the dataset std, used for input normalization"
)
parser.add_argument("--target_length", type=int, help="the input length in frames")
parser.add_argument("--noise", help="if use balance sampling", type=ast.literal_eval)

parser.add_argument(
    "--exp_dir", type=str, default="", help="directory to dump experiments"
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.001,
    type=float,
    metavar="LR",
    help="initial learning rate",
)
parser.add_argument(
    "--optim",
    type=str,
    default="adam",
    help="training optimizer",
    choices=["sgd", "adam"],
)
parser.add_argument(
    "-b", "--batch_size", default=48, type=int, metavar="N", help="mini-batch size"
)
parser.add_argument(
    "-w",
    "--num_workers",
    default=32,
    type=int,
    metavar="NW",
    help="# of workers for dataloading (default: 32)",
)
parser.add_argument(
    "--n_epochs", type=int, default=10, help="number of maximum training epochs"
)
# not used in the formal experiments, only in preliminary experiments
parser.add_argument(
    "--lr_patience",
    type=int,
    default=1,
    help="how many epoch to wait to reduce lr if mAP doesn't improve",
)
parser.add_argument(
    "--lr_adapt", help="if use adaptive learning rate", type=ast.literal_eval
)
parser.add_argument(
    "--metrics",
    type=str,
    default="mAP",
    help="the main evaluation metrics in finetuning",
    choices=["mAP", "acc"],
)
parser.add_argument(
    "--loss",
    type=str,
    default="BCE",
    help="the loss function for finetuning, depend on the task",
    choices=["BCE", "CE"],
)
parser.add_argument(
    "--warmup",
    help="if use warmup learning rate scheduler",
    type=ast.literal_eval,
    default="True",
)
parser.add_argument(
    "--lrscheduler_start", default=2, type=int, help="when to start decay in finetuning"
)
parser.add_argument(
    "--lrscheduler_step",
    default=1,
    type=int,
    help="the number of step to decrease the learning rate in finetuning",
)
parser.add_argument(
    "--lrscheduler_decay",
    default=0.5,
    type=float,
    help="the learning rate decay ratio in finetuning",
)
parser.add_argument("--freqm", help="frequency mask max length", type=int, default=0)
parser.add_argument("--timem", help="time mask max length", type=int, default=0)

parser.add_argument(
    "--wa", help="if do weight averaging in finetuning", type=ast.literal_eval
)
parser.add_argument(
    "--wa_start",
    type=int,
    default=1,
    help="which epoch to start weight averaging in finetuning",
)
parser.add_argument(
    "--wa_end",
    type=int,
    default=10,
    help="which epoch to end weight averaging in finetuning",
)

parser.add_argument(
    "--n-print-steps", type=int, default=100, help="number of steps to print statistics"
)
parser.add_argument("--save_model", help="save the model or not", type=ast.literal_eval)

parser.add_argument(
    "--mixup",
    type=float,
    default=0,
    help="how many (0-1) samples need to be mixup during training",
)
parser.add_argument(
    "--bal", type=str, default=None, help="use balanced sampling or not"
)

parser.add_argument(
    "--label_smooth", type=float, default=0.1, help="label smoothing factor"
)
parser.add_argument("--weight_file", type=str, default=None, help="path to weight file")
parser.add_argument(
    "--pretrain_path", type=str, default="None", help="pretrained model path"
)
parser.add_argument(
    "--ftmode", type=str, default="multimodal", help="how to fine-tune the model"
)
parser.add_argument(
    "--ftmode_test", type=str, default=None, help="how to fine-tune the model"
)


parser.add_argument(
    "--head_lr",
    type=float,
    default=50.0,
    help="learning rate ratio the newly initialized layers / pretrained weights",
)
parser.add_argument(
    "--mm_lr",
    type=float,
    default=None,
    help="learning rate ratio the newly initialized layers / pretrained weights",
)
parser.add_argument(
    "--freeze_base", help="freeze the backbone or not", type=ast.literal_eval
)
parser.add_argument("--skip_frame_agg", help="if do frame agg", type=ast.literal_eval)
parser.add_argument(
    "--sql_path", type=str, default="./artefacts/sql", help="path to sql"
)
parser.add_argument("--video_path_prefix", type=str, default=None, help="path to video")
parser.add_argument("--image_path_prefix", type=str, default=None, help="path to image")
parser.add_argument(
    "--accumulate_grad_batches",
    type=int,
    default=1,
    help="number of batches to accumulate gradients",
)

## yb: new params

parser.add_argument(
    "--dis_w", type=float, default=0, help="weight term for distillation loss"
)
parser.add_argument(
    "--dis_w_2", type=float, default=0, help="weight term for distillation loss"
)


## yb: torch need these
parser.add_argument("--master_addr", type=str)
parser.add_argument("--nproc_per_node", type=int)


# wandb
parser.add_argument("--wandb", type=int, default=0, help="wandb")
parser.add_argument("--model_name", type=str, default=None, help="for log")


# distributed training parameters
parser.add_argument(
    "--world_size", default=1, type=int, help="number of distributed processes"
)
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument(
    "--dist_url", default="env://", help="url used to set up distributed training"
)


args = parser.parse_args()

if args.ftmode_test is None:
    args.ftmode_test = args.ftmode
if args.mm_lr is None:
    args.mm_lr = args.head_lr

args.local_rank = int(os.environ["LOCAL_RANK"])  # for torchrun

init_seeds(87 + args.local_rank, cuda_deterministic=False)  #: add
utils.init_distributed_mode(args)

if args.wandb:
    if args.local_rank == 0:
        wandb.init(config=args, project="uavm", name=args.model_name)

# all exp in this work is based on 224 * 224 image
im_res = 224
audio_conf = {
    "num_mel_bins": 128,
    "target_length": args.target_length,
    "freqm": args.freqm,
    "timem": args.timem,
    "mixup": args.mixup,
    "dataset": args.dataset,
    "mode": "train",
    "mean": args.dataset_mean,
    "std": args.dataset_std,
    "noise": args.noise,
    "label_smooth": args.label_smooth,
    "im_res": im_res,
}
val_audio_conf = {
    "num_mel_bins": 128,
    "target_length": args.target_length,
    "freqm": 0,
    "timem": 0,
    "mixup": 0,
    "dataset": args.dataset,
    "mode": "eval",
    "mean": args.dataset_mean,
    "std": args.dataset_std,
    "noise": False,
    "im_res": im_res,
}


# average the model weights of checkpoints, note it is not ensemble, and does not increase computational overhead
def wa_model(exp_dir, start_epoch, end_epoch):
    sdA = torch.load(
        exp_dir + "/models/audio_model." + str(start_epoch) + ".pth", map_location="cpu"
    )
    model_cnt = 1
    for epoch in range(start_epoch + 1, end_epoch + 1):
        sdB = torch.load(
            exp_dir + "/models/audio_model." + str(epoch) + ".pth", map_location="cpu"
        )
        for key in sdA:
            sdA[key] = sdA[key] + sdB[key]
        model_cnt += 1
    print("wa {:d} models from {:d} to {:d}".format(model_cnt, start_epoch, end_epoch))
    for key in sdA:
        sdA[key] = sdA[key] / float(model_cnt)
    return sdA


if args.bal == "bal":
    ###### -----------> yb: comment out
    print("balanced sampler is being used")
    if args.weight_file == None:
        samples_weight = np.loadtxt(args.data_train[:-5] + "_weight.csv", delimiter=",")
    else:
        samples_weight = np.loadtxt(
            args.data_train[:-5] + "_" + args.weight_file + ".csv", delimiter=","
        )

    sampler = WeightedRandomSampler(
        samples_weight, len(samples_weight), replacement=True
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=dataloader.AudiosetDataset(
            args.data_train,
            label_csv=args.label_csv,
            audio_conf=audio_conf,
            sql_path=args.sql_path,
            video_path_prefix=args.video_path_prefix,
            image_path_prefix=args.image_path_prefix,
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=sampler,
        pin_memory=True,
        persistent_workers=False,
    )
    ##### <----------
else:
    print("balanced sampler is not used")

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataloader.AudiosetDataset(
            args.data_train,
            label_csv=args.label_csv,
            audio_conf=audio_conf,
            sql_path=args.sql_path,
            video_path_prefix=args.video_path_prefix,
            image_path_prefix=args.image_path_prefix,
        ),
        shuffle=True,
    )
    train_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(
            args.data_train, label_csv=args.label_csv, audio_conf=audio_conf
        ),
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

val_sampler = SequentialDistributedSampler(
    dataloader.AudiosetDataset(
        args.data_val,
        label_csv=args.label_csv,
        audio_conf=val_audio_conf,
        sql_path=args.sql_path,
        video_path_prefix=args.video_path_prefix,
        image_path_prefix=args.image_path_prefix,
    ),
    batch_size=args.batch_size,
)
val_loader = torch.utils.data.DataLoader(
    dataloader.AudiosetDataset(
        args.data_val,
        label_csv=args.label_csv,
        audio_conf=val_audio_conf,
        sql_path=args.sql_path,
        video_path_prefix=args.video_path_prefix,
        image_path_prefix=args.image_path_prefix,
    ),
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    sampler=val_sampler,
    pin_memory=False,
    persistent_workers=False,
)


if args.data_eval != None:
    eval_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(
            args.data_eval,
            label_csv=args.label_csv,
            audio_conf=val_audio_conf,
            sql_path=args.sql_path,
            video_path_prefix=args.video_path_prefix,
            image_path_prefix=args.image_path_prefix,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )


############ model initialze ########
if args.model == "cav-mae-ft":
    print(
        "finetune a cav-mae model with 11 modality-specific layers and 1 modality-sharing layers"
    )
    # audio_model = models.CAVMAEFT(label_dim=args.n_class, modality_specific_depth=11, opt=args)
    audio_model = models.CAVMAEFT_BASE(
        label_dim=args.n_class, modality_specific_depth=9999
    )
    # audio_model = models.CAVMAEFT_BASE_CLIP(label_dim=args.n_class, modality_specific_depth=9999)
else:
    raise ValueError("model not supported")

if args.pretrain_path == "None":
    warnings.warn("Note you are finetuning a model without any finetuning.")

####### finetune based on a CAV-MAE pretrained model, which is the default setting unless for ablation study
if args.pretrain_path != "None":
    # TODO: change this to a wget link
    mdl_weight = torch.load(args.pretrain_path, map_location="cpu")
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = torch.nn.DataParallel(audio_model)
    miss, unexpected = audio_model.load_state_dict(mdl_weight, strict=False)
    print("now load cav-mae pretrained weights from ", args.pretrain_path)
    print("Missing: ", miss)
    print("Unexpected: ", unexpected)

    # mdl_weight = torch.load('/mount/opr/yblin/yb_weights/joint.25.pth', map_location='cpu')
    # miss, unexpected = audio_model.load_state_dict(mdl_weight, strict=False)
    # torch.save(audio_model.state_dict(),args.pretrain_path)

    # audio_model.module.__create_fusion__()
    audio_model = audio_model.module.to(torch.device("cpu"))

print("\nCreating experiment directory: %s" % args.exp_dir)
try:
    os.makedirs("%s/models" % args.exp_dir)
except:
    pass
with open("%s/args.pkl" % args.exp_dir, "wb") as f:
    pickle.dump(args, f)
with open(args.exp_dir + "/args.json", "w") as f:
    json.dump(args.__dict__, f, indent=2)


############ train ########
print("Now starting training for {:d} epochs.".format(args.n_epochs))
if args.n_epochs > 0:
    train(audio_model, train_loader, val_loader, val_sampler, args)
# exit()


################# eval ###################
if args.loss == "BCE":
    loss_fn = torch.nn.BCEWithLogitsLoss()
elif args.loss == "CE":
    loss_fn = torch.nn.CrossEntropyLoss()
args.loss_fn = loss_fn


# evaluate with multiple frames
# if not isinstance(audio_model, torch.nn.DataParallel):
#     audio_model = torch.nn.DataParallel(audio_model)
# if args.wa == True:
#     sdA = wa_model(args.exp_dir, start_epoch=args.wa_start, end_epoch=args.wa_end)
#     torch.save(sdA, args.exp_dir + "/models/audio_model_wa.pth")
# else:
#     # if no wa, use the best checkpint
#     sdA = torch.load(args.exp_dir + '/models/best_audio_model.pth', map_location='cpu')

# sdA = torch.load(args.exp_dir + '/models/best_audio_model.pth', map_location='cpu')

# msg = audio_model.load_state_dict(sdA, strict=True)
# print(msg)
# audio_model = audio_model.module.to(torch.device('cpu'))


print("Now starting evaluation.")
device = torch.device("cuda", args.local_rank)
audio_model = audio_model.to(device)
audio_model = DDP(
    audio_model,
    device_ids=[args.local_rank],
    output_device=args.local_rank,
    find_unused_parameters=True,
)
audio_model.eval()

# skil multi-frame evaluation, for audio-only model
if args.skip_frame_agg == True:
    val_audio_conf["frame_use"] = 5
    val_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(
            args.data_val,
            label_csv=args.label_csv,
            audio_conf=val_audio_conf,
            sql_path=args.sql_path,
            video_path_prefix=args.video_path_prefix,
            image_path_prefix=args.image_path_prefix,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    stats, audio_output, target = validate(
        audio_model, val_loader, val_sampler, args, output_pred=True
    )
    # save audio output and target
    audio_output, target = audio_output.cpu().numpy(), target.cpu().numpy()
    np.save(args.exp_dir + f"/{args.ftmode}_output.npy", audio_output)
    np.save(args.exp_dir + f"/{args.ftmode}_target.npy", target)

    if args.metrics == "mAP":
        cur_res = np.mean([stat["AP"] for stat in stats])
        print("mAP is {:.4f}".format(cur_res))
    elif args.metrics == "acc":
        cur_res = stats[0]["acc"]
        print("acc is {:.4f}".format(cur_res))
else:
    res = []
    multiframe_pred = []
    total_frames = 10  # change if your total frame is different
    for frame in range(total_frames):
        val_audio_conf["frame_use"] = frame
        val_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetDataset(
                args.data_val,
                label_csv=args.label_csv,
                audio_conf=val_audio_conf,
                sql_path=args.sql_path,
                video_path_prefix=args.video_path_prefix,
                image_path_prefix=args.image_path_prefix,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        stats, audio_output, target = validate(
            audio_model, val_loader, val_sampler, args, output_pred=True
        )

        print(audio_output.shape)
        if args.metrics == "acc":
            audio_output = torch.nn.functional.softmax(audio_output.float(), dim=-1)
        elif args.metrics == "mAP":
            audio_output = torch.nn.functional.sigmoid(audio_output.float())

        audio_output, target = audio_output.cpu().numpy(), target.cpu().numpy()

        # save audio output and target
        np.save(args.exp_dir + f"/{args.ftmode}_output_{frame}.npy", audio_output)
        np.save(args.exp_dir + f"/{args.ftmode}_target_{frame}.npy", target)

        multiframe_pred.append(audio_output)
        if args.metrics == "mAP":
            cur_res = np.mean([stat["AP"] for stat in stats])
            print("------------> mAP of frame {:d} is {:.4f}".format(frame, cur_res))
        elif args.metrics == "acc":
            cur_res = stats[0]["acc"]
            print("acc of frame {:d} is {:.4f}".format(frame, cur_res))
        res.append(cur_res)

    # ensemble over frames
    multiframe_pred = np.mean(multiframe_pred, axis=0)
    if args.metrics == "acc":
        acc = metrics.accuracy_score(
            np.argmax(target, 1), np.argmax(multiframe_pred, 1)
        )
        print("multi-frame acc is {:f}".format(acc))
        res.append(acc)
    elif args.metrics == "mAP":
        AP = []
        for k in range(args.n_class):
            # Average precision
            avg_precision = metrics.average_precision_score(
                target[:, k], multiframe_pred[:, k], average=None
            )
            AP.append(avg_precision)
        mAP = np.mean(AP)
        print("multi-frame mAP is {:.4f}".format(mAP))
        res.append(mAP)
    np.savetxt(args.exp_dir + "/mul_frame_res.csv", res, delimiter=",")
