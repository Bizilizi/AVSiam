import csv
import json
import os.path
import os
import time
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
import torchvision.transforms as T
from PIL import Image
import PIL


import torchvision

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from ipdb import set_trace
from glob import glob

import sqlite3

# from moviepy import *
# from moviepy.editor import *


def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, "r") as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row["mid"]] = row["index"]
            line_count += 1
    return index_lookup


def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, "r") as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row["index"]] = row["display_name"]
            line_count += 1
    return name_lookup


def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list


def preemphasis(signal, coeff=0.97):
    """perform preemphasis on the input signal.
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


class AudiosetDataset(Dataset):
    def __init__(
        self,
        dataset_json_file,
        audio_conf,
        label_csv=None,
        sql_path="./artefacts/sql",
        video_path_prefix=None,
        image_path_prefix=None,
    ):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """

        self.sql_path = sql_path
        self.video_path_prefix = video_path_prefix
        self.image_path_prefix = image_path_prefix

        self.audio_conf = audio_conf
        self.label_smooth = self.audio_conf.get("label_smooth", 0.0)
        print("Using Label Smoothing: " + str(self.label_smooth))
        self.melbins = self.audio_conf.get("num_mel_bins")
        self.freqm = self.audio_conf.get("freqm", 0)
        self.timem = self.audio_conf.get("timem", 0)
        print(
            "now using following mask: {:d} freq, {:d} time".format(
                self.audio_conf.get("freqm"), self.audio_conf.get("timem")
            )
        )
        self.mixup = self.audio_conf.get("mixup", 0)
        print("now using mix-up with rate {:f}".format(self.mixup))
        self.dataset = self.audio_conf.get("dataset")
        print("now process " + self.dataset)
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get("mean")
        self.norm_std = self.audio_conf.get("std")
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = (
            self.audio_conf.get("skip_norm")
            if self.audio_conf.get("skip_norm")
            else False
        )
        if self.skip_norm:
            print(
                "now skip normalization (use it ONLY when you are computing the normalization stats)."
            )
        else:
            print(
                "use dataset mean {:.3f} and std {:.3f} to normalize the input.".format(
                    self.norm_mean, self.norm_std
                )
            )

        # if add noise for data augmentation
        self.noise = self.audio_conf.get("noise", False)
        if self.noise == True:
            print("now use noise augmentation")
        else:
            print("not use noise augmentation")

        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        print("number of classes is {:d}".format(self.label_num))

        self.target_length = self.audio_conf.get("target_length")

        # train or eval
        self.mode = self.audio_conf.get("mode")
        print("now in {:s} mode.".format(self.mode))

        # set the frame to use in the eval mode, default value for training is -1 which means random frame
        self.frame_use = self.audio_conf.get("frame_use", -1)
        # by default, 10 frames are used
        self.total_frame = self.audio_conf.get("total_frame", 10)
        print(
            "now use frame {:d} from total {:d} frames".format(
                self.frame_use, self.total_frame
            )
        )

        # by default, all models use 224*224, other resolutions are not tested
        self.im_res = self.audio_conf.get("im_res", 224)
        print("now using {:d} * {:d} image input".format(self.im_res, self.im_res))

        # self.preprocess = T.Compose([
        # 	T.Resize(self.im_res, interpolation=PIL.Image.BICUBIC),
        # 	T.CenterCrop(self.im_res),
        # 	T.ToTensor(),
        # 	T.Normalize(
        # 		mean=[0.4850, 0.4560, 0.4060],
        # 		std=[0.2290, 0.2240, 0.2250]
        # 	)])

        self.my_normalize = Compose(
            [
                Resize([224, 224], interpolation=Image.BICUBIC, antialias=True),
                Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )

        ###### ---------> for some timm pretrained models
        # self.my_normalize = Compose([
        # 		Resize([224,224], interpolation=Image.BICUBIC, antialias=True),
        # 		Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # ])
        #### <----------

        # self.dataset = 'audioset_2m'
        # self.dataset = 'audioset_20k'
        if self.dataset == "vggsound":

            if self.audio_conf["mode"] == "train":
                con = sqlite3.connect(
                    "file:" + f"{self.sql_path}/train_vgg.sqlite.db" + "?mode=ro",
                    uri=True,
                )
                self.cur = con.cursor()
                self.num_samples = self.cur.execute(
                    "SELECT COUNT(*) FROM annos"
                ).fetchone()[0]
            else:
                con = sqlite3.connect(
                    "file:" + f"{self.sql_path}/val_vgg.sqlite.db" + "?mode=ro",
                    uri=True,
                )
                # con = sqlite3.connect("file:" + '/home/wiss/zverev/AVSiam/artefacts/sql/val_vgg_retrieval.sqlite.db' + "?mode=ro", uri=True) ### <----- YB: for retrieval
                self.cur = con.cursor()
                self.num_samples = self.cur.execute(
                    "SELECT COUNT(*) FROM annos"
                ).fetchone()[0]

        elif self.dataset == "audioset_20k":
            if self.audio_conf["mode"] == "train":
                con = sqlite3.connect(
                    "file:" + f"{self.sql_path}/train_as_20k.sqlite.db" + "?mode=ro",
                    uri=True,
                )
                self.cur = con.cursor()
                self.num_samples = self.cur.execute(
                    "SELECT COUNT(*) FROM annos"
                ).fetchone()[0]
            else:
                con = sqlite3.connect(
                    "file:" + f"{self.sql_path}/eval_as_v2.sqlite.db" + "?mode=ro",
                    uri=True,
                )
                # con = sqlite3.connect("file:" + '/home/wiss/zverev/AVSiam/artefacts/sql/val_as_retrieval.sqlite.db' + "?mode=ro", uri=True)
                self.cur = con.cursor()
                self.num_samples = self.cur.execute(
                    "SELECT COUNT(*) FROM annos"
                ).fetchone()[0]

                # self.map = np.linspace(0, self.num_samples -1 , num=1725, dtype=int)
                # self.num_samples = len(self.map)

        elif self.dataset == "audioset_2m":
            if self.audio_conf["mode"] == "train":
                con = sqlite3.connect(
                    "file:" + f"{self.sql_path}/train_as_2m_v2.sqlite.db" + "?mode=ro",
                    uri=True,
                )
                self.cur = con.cursor()
                self.num_samples = self.cur.execute(
                    "SELECT COUNT(*) FROM annos"
                ).fetchone()[0]
            else:
                con = sqlite3.connect(
                    "file:" + f"{self.sql_path}/eval_as_v2.sqlite.db" + "?mode=ro",
                    uri=True,
                )
                self.cur = con.cursor()
                self.num_samples = self.cur.execute(
                    "SELECT COUNT(*) FROM annos"
                ).fetchone()[0]

        print("Dataset has {:d} samples".format(self.num_samples))

        self.num_frame = 10

        # self.num_samples = self.data.shape[0]

    # change python list to numpy array to avoid memory leak.
    def pro_data(self, data_json):
        for i in range(len(data_json)):
            data_json[i] = [
                data_json[i]["wav"],
                data_json[i]["labels"],
                data_json[i]["video_id"],
                data_json[i]["video_path"],
            ]
        data_np = np.array(data_json, dtype=str)
        return data_np

    # reformat numpy data to original json format, make it compatible with old code
    def decode_data(self, np_data):
        datum = {}
        # datum['wav']  = np_data ## yb:!!@@!
        datum["wav"] = np_data[0]
        datum["labels"] = np_data[1]
        # datum['video_id'] = np_data[2]
        # datum['video_path'] = np_data[3]
        return datum

    def get_image(self, filename, filename2=None, mix_lambda=1):
        if filename2 == None:
            img = torchvision.io.read_image(filename)
            img = img / 255
            img = self.my_normalize(img)  #

            return img
        else:
            img1 = torchvision.io.read_image(filename)
            img1 = img1 / 255
            img1 = self.my_normalize(img1)  #

            img2 = torchvision.io.read_image(filename2)
            img2 = img2 / 255
            img2 = self.my_normalize(img2)  #

            image = mix_lambda * img1 + (1 - mix_lambda) * img2
            return image

    def _wav2fbank(self, filename, filename2=None, mix_lambda=-1):
        # self.yb_parse_name(filename)
        # no mixup

        if filename2 == None:
            # if self.audio_conf['mode'] == 'train':
            # 	if self.dataset == 'audioset_2m':
            # 		if "vggsound" in filename:
            # 			filename = self.yb_parse_name(filename).replace('mp4','wav').replace('vggsound','vggsound/wav')
            # 		else:
            # 			filename = self.yb_parse_name(filename).replace('mp4','wav').replace('train_2m','train_2m_audio')  #yb
            # 	else:
            # 		filename = self.yb_parse_name(filename).replace('mp4','wav').replace('train_balanced','train_balanced_audio')  #yb
            # else:
            # 	filename = self.yb_parse_name(filename).replace('mp4','wav').replace('eval_segments','eval_segments_audio')  #yb

            waveform, sr = torchaudio.load(filename)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)

            waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform - waveform.mean()

        # mixup
        else:
            # if self.audio_conf['mode'] == 'train':
            # 	if self.dataset == 'audioset_2m':

            # 		if "vggsound" in filename:
            # 			filename = self.yb_parse_name(filename).replace('mp4','wav').replace('vggsound','vggsound/wav')
            # 			filename2 = self.yb_parse_name(filename2).replace('mp4','wav').replace('vggsound','vggsound/wav')
            # 		else:
            # 			filename = self.yb_parse_name(filename).replace('mp4','wav').replace('train_2m','train_2m_audio')  #yb
            # 			filename2 = self.yb_parse_name(filename2).replace('mp4','wav').replace('train_2m','train_2m_audio')  #yb
            # 	else:
            # 		filename = self.yb_parse_name(filename).replace('mp4','wav').replace('train_balanced','train_balanced_audio')  #yb
            # 		filename2 = self.yb_parse_name(filename2).replace('mp4','wav').replace('train_balanced','train_balanced_audio')  #yb
            # else:
            # 	filename = self.yb_parse_name(filename).replace('mp4','wav').replace('eval_segments','eval_segments_audio')  #yb
            # 	filename2 = self.yb_parse_name(filename2).replace('mp4','wav').replace('eval_segments','eval_segments_audio')  #yb

            waveform1, sr = torchaudio.load(filename)
            if sr != 16000:
                waveform1 = torchaudio.functional.resample(waveform1, sr, 16000)
            waveform1 = waveform1.mean(dim=0, keepdim=True)

            waveform2, _ = torchaudio.load(filename2)
            if sr != 16000:
                waveform2 = torchaudio.functional.resample(waveform2, sr, 16000)
            waveform2 = waveform2.mean(dim=0, keepdim=True)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0 : waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0 : waveform1.shape[1]]

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        try:
            fbank = torchaudio.compliance.kaldi.fbank(
                waveform,
                htk_compat=True,
                sample_frequency=sr,
                use_energy=False,
                window_type="hanning",
                num_mel_bins=self.melbins,
                dither=0.0,
                frame_shift=10,
            )
        except:
            fbank = torch.zeros([512, 128]) + 0.01
            print("there is a loading error")

        target_length = self.target_length
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        return fbank

    def get_frame_path(self, video_id, video_path, frame_idx):
        return video_path + "/frame_" + str(frame_idx) + "/" + video_id + ".jpg"

    def randselect_img(self, video_id, video_path):
        if self.mode == "eval":
            # if not specified, use the middle frame
            if self.frame_use == -1:
                frame_idx = int((self.total_frame) / 2)
            else:
                frame_idx = self.frame_use
        else:
            frame_idx = random.randint(0, 9)

        while (
            os.path.exists(
                video_path + "/frame_" + str(frame_idx) + "/" + video_id + ".jpg"
            )
            == False
            and frame_idx >= 1
        ):
            print("frame {:s} {:d} does not exist".format(video_id, frame_idx))
            frame_idx -= 1
        out_path = video_path + "/frame_" + str(frame_idx) + "/" + video_id + ".jpg"
        # print(out_path)
        return out_path

    def __getitem__(self, index):

        start_time = time.time()

        # SQL query timing
        sql_start = time.time()
        query = f"SELECT * FROM annos WHERE id = {index};"
        res = self.cur.execute(query)
        datum = res.fetchone()
        sql_time = time.time() - sql_start

        # Path construction timing
        path_start = time.time()
        if self.dataset == "vggsound":
            video_path = os.path.join(self.video_path_prefix, datum[1] + ".wav")
            mix_sample_idx = random.randint(0, self.num_samples - 1)
            query = f"SELECT * FROM annos WHERE id = {mix_sample_idx};"
            res = self.cur.execute(query)
            mix_datum = res.fetchone()
            video_path_mix = os.path.join(self.video_path_prefix, mix_datum[1] + ".wav")

        elif self.dataset == "audioset_20k":
            if self.audio_conf["mode"] == "train":
                video_path = (
                    "/mnt/opr/yblin/audioset_sun/train_balanced/" + datum[1] + ".waf"
                )
                mix_sample_idx = random.randint(0, self.num_samples - 1)
                query = f"SELECT * FROM annos WHERE id = {mix_sample_idx};"
                res = self.cur.execute(query)
                mix_datum = res.fetchone()
                video_path_mix = os.path.join(
                    self.video_path_prefix, mix_datum[1] + ".waf"
                )
            else:
                video_path = os.path.join(self.video_path_prefix, datum[1] + ".wav")
                mix_sample_idx = random.randint(0, self.num_samples - 1)
                query = f"SELECT * FROM annos WHERE id = {mix_sample_idx};"
                res = self.cur.execute(query)
                mix_datum = res.fetchone()
                video_path_mix = os.path.join(
                    self.video_path_prefix, mix_datum[1] + ".wav"
                )

        elif self.dataset == "audioset_2m":
            if self.audio_conf["mode"] == "train":
                video_path = os.path.join(self.video_path_prefix, datum[1] + ".wav")
                mix_sample_idx = random.randint(0, self.num_samples - 1)
                query = f"SELECT * FROM annos WHERE id = {mix_sample_idx};"
                res = self.cur.execute(query)
                mix_datum = res.fetchone()
                video_path_mix = os.path.join(
                    self.video_path_prefix, mix_datum[1] + ".wav"
                )
            else:
                video_path = os.path.join(self.video_path_prefix, datum[1] + ".wav")
                mix_sample_idx = random.randint(0, self.num_samples - 1)
                query = f"SELECT * FROM annos WHERE id = {mix_sample_idx};"
                res = self.cur.execute(query)
                mix_datum = res.fetchone()
                video_path_mix = os.path.join(
                    self.video_path_prefix, mix_datum[1] + ".wav"
                )
        path_time = time.time() - path_start

        # Audio/video processing timing
        if random.random() < self.mixup:
            # get the mixed fbank
            mix_lambda = np.random.beta(10, 10)

            try:
                fbank = self._wav2fbank(video_path, video_path_mix, mix_lambda)
            except Exception as e:
                print(e)
                fbank = torch.zeros([self.target_length, 128]) + 0.01
                print("there is an error in loading audio 1", datum[1], mix_datum[1])

            try:

                weight = random.random()
                image = self.get_image(
                    self.randselect_img(datum[1], self.image_path_prefix),
                    self.randselect_img(mix_datum[1], self.image_path_prefix),
                    weight,
                ).unsqueeze(0)
                
            except Exception as e:
                print(e)
                image = torch.zeros([1, 3, self.im_res, self.im_res]) + 0.01
                print(
                    "there is an error in loading image 1", video_path, video_path_mix
                )

            label_indices = np.zeros(self.label_num) + (
                self.label_smooth / self.label_num
            )
            for label_str in datum[-1].split(","):
                label_indices[int(self.index_dict[label_str])] += mix_lambda * (
                    1.0 - self.label_smooth
                )
            for label_str in mix_datum[-1].split(","):
                label_indices[int(self.index_dict[label_str])] += (1.0 - mix_lambda) * (
                    1.0 - self.label_smooth
                )
            label_indices = torch.FloatTensor(label_indices)

        else:
            label_indices = np.zeros(self.label_num) + (
                self.label_smooth / self.label_num
            )
            try:
                fbank = self._wav2fbank(video_path, None, 0)
            except:
                fbank = torch.zeros([self.target_length, 128]) + 0.01
                print("there is an error in loading audio 2", datum)
            try:

                if self.mode == "eval":
                    images = []
                    for i in range(self.num_frame):
                        images.append(
                            self.get_image(
                                self.get_frame_path(
                                    datum[1], self.image_path_prefix, i
                                )
                            )
                        )
                    image = torch.stack(images, dim=0)
                else:
                    image = self.get_image(
                        self.randselect_img(datum[1], self.image_path_prefix),
                    ).unsqueeze(0)
                    
            except Exception as e:
                print(e)
                if self.mode == "eval":
                    image = torch.zeros([10, 3, self.im_res, self.im_res]) + 0.01
                else:
                    image = torch.zeros([1, 3, self.im_res, self.im_res]) + 0.01
                print("there is an error in loading image 2", video_path)

            for label_str in datum[-1].split(","):
                label_indices[int(self.index_dict[label_str])] = 1.0 - self.label_smooth
            label_indices = torch.FloatTensor(label_indices)

        # SpecAug timing
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = torch.transpose(fbank, 0, 1)
        fbank = fbank.unsqueeze(0)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)

        if self.skip_norm == False:
            fbank = (fbank - self.norm_mean) / (self.norm_std)
        else:
            pass

        if self.noise == True:
            fbank = (
                fbank
                + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            )
            fbank = torch.roll(
                fbank, np.random.randint(-self.target_length, self.target_length), 0
            )

        return fbank, image, label_indices

    def __len__(self):
        return self.num_samples
