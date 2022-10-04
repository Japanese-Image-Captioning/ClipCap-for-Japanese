
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm
import os
import pickle
import sys
import json
import cv2
import torchvision
from PIL import Image
from typing import Tuple

DEBUG = False


class StairCaptionDataset(Dataset):
    def __init__(self, tokenizer, clip_preprocess, split, prefix_length, transform=None):
        print(f"Preprocess for {split} ... ")
        dataset = None
        if split == "train":
            with open("STAIR-captions/stair_captions_v1.2_train_tokenized.json", 'r') as f:
                dataset = json.load(f)
        else:
            with open("STAIR-captions/stair_captions_v1.2_val_tokenized.json", 'r') as f:
                dataset = json.load(f)

        caption2token = {}
        labels_dict = {}  # labels[image_id] = captions
        for i, cmeta in enumerate(tqdm(dataset["annotations"])):
            image_id = cmeta["image_id"]
            caption = cmeta["tokenized_caption"].split(' ')
            caplen = len(caption)
            tokens = tokenizer.encode(caption, return_tensors="pt").squeeze(0)
            caption = ''.join(caption)
            labels_dict.setdefault(image_id, [])
            labels_dict[image_id].append((tokens, caption, caplen))
            caption2token[caption] = tokens
            if DEBUG and i >= 100:
                break

        labels = []
        if split == "train":
            for i, (image_id, captions) in enumerate(tqdm(labels_dict.items())):
                for tokens, caption, caplen in captions:
                    labels.append((image_id, tokens, caption, caplen))  # image_id, caption

        else:
            img_idxs = list(labels_dict.keys())
            half_size = len(img_idxs) // 2
            img_idxs = img_idxs[:half_size] if split == "val" else img_idxs[half_size:]  # COCOのvalを半分に分割
            for image_id in tqdm(img_idxs):
                for tokens, caption, caplen in labels_dict[image_id]:
                    labels.append((image_id, tokens, caption, caplen))  # image_id, caption

        self.labels = labels
        self.labels_dict = labels_dict
        self.split = split
        self.transform = transform
        self.caption2token = caption2token
        self.prefix_length = prefix_length
        self.clip_preprocess = clip_preprocess

        all_len = torch.tensor([len(self.labels[i][2]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))

    def __getitem__(self, i):
        image_id, tokens, caption, caplen = self.labels[i]
        img = self.get_coco_image(image_id, self.split)
        if img.shape[0] == 1:
            img = torch.cat([img] * 3, dim=0)

        assert img.shape[0] == 3, f"img.shape == {img.shape}"
        if self.transform is not None:
            img = self.transform(img)

        tokens, mask = self.pad_tokens(tokens)
        tokens, mask = tokens.cuda(), mask.cuda()
        if self.split == 'train':
            return img, tokens, mask, image_id
        else:
            all_captions = [caption for _, caption, _ in self.labels_dict[image_id]]
            return img, tokens, mask, image_id, all_captions[:5]

    def __len__(self):
        return len(self.labels)

    def pad_tokens(self, tokens):
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask

    def get_coco_image(self, image_id, split):
        if split == "test":
            split = "val"  # COCOのvalを使う

        L = len("000000490055")
        prefix = "0" * (L - len(str(image_id)))
        path = f"{split}2014/COCO_{split}2014_{prefix}{image_id}.jpg"
        resized_path = f"{split}2014/resized_COCO_{split}2014_{prefix}{image_id}.jpg"

        if os.path.exists(resized_path):
            path = resized_path
        else:
            img = cv2.imread(path)
            img = cv2.resize(img, (256, 256))
            cv2.imwrite(resized_path, img)
            path = resized_path

        img_pil = Image.open(path)
        img_tensor = self.clip_preprocess(img_pil).squeeze(0).to("cuda")
        return img_tensor


class ClipCocoDataset(Dataset):

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens, mask = self.pad_tokens(item)
        prefix = self.prefixes[self.caption2embedding[item]]
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        return tokens, mask, prefix

    def __init__(self, data_path: str, prefix_length: int, gpt2_type: str = "gpt2",
                 normalize_prefix=False):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Data size is %0d" % len(all_data["clip_embedding"]))
        sys.stdout.flush()
        # self.prefixes = all_data["clip_embedding"]
        captions_raw = all_data["captions"]
        self.image_ids = [caption["image_id"] for caption in captions_raw]
        self.captions = [caption['caption'] for caption in captions_raw]
        if os.path.isfile(f"{data_path[:-4]}_tokens.pkl"):
            with open(f"{data_path[:-4]}_tokens.pkl", 'rb') as f:
                self.captions_tokens, self.caption2embedding, self.max_seq_len = pickle.load(f)
        else:
            self.captions_tokens = []
            self.caption2embedding = []
            max_seq_len = 0
            for caption in captions_raw:
                self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption['caption']), dtype=torch.int64))
                self.caption2embedding.append(caption["clip_embedding"])
                max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
            # self.max_seq_len = max_seq_len
            with open(f"{data_path[:-4]}_tokens.pkl", 'wb') as f:
                pickle.dump([self.captions_tokens, self.caption2embedding, max_seq_len], f)
        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))
