import torch.nn as nn
from transformers import AutoModelForCausalLM
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import sys
import argparse
import japanese_clip as ja_clip
import numpy as np
import wandb
from typing import Tuple, Optional, Union
from dataset import *
from model import *
from transformers import T5Tokenizer, AutoModelForCausalLM, GPT2Model

DEBUG = False


class GPT:
    def __init__(self) -> None:
        tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
        tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading
        model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")

        self.tokenizer = tokenizer
        self.model = model.cuda()

    def __call__(self, tokens):
        return self.model(tokens)


class CLIP:
    def __init__(self) -> None:
        model, preprocess = ja_clip.load("rinna/japanese-clip-vit-b-16", cache_dir="/tmp/japanese_clip", device="cuda")
        self.model = model.cuda()
        self.preprocess = preprocess

    def __call__(self, x):
        # x = self.preprocess(x).unsqueeze(0).to("cuda")
        x = self.model.get_image_features(x)
        return x


class JapaneseClipCap(nn.Module):
    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8, mapping_type: str = "mlp"):

        super().__init__()
        self.gpt = GPT()
        self.clip = CLIP()
        self.prefix_length = prefix_length
        self.gpt_embedding_size = self.gpt.model.transformer.wte.weight.shape[1]
        if mapping_type == "mlp":
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
                                     self.gpt_embedding_size * prefix_length))
        else:
            self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
                                                  clip_length, num_layers)

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, img, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        clip_feat = self.clip(img)
        embedding_text = self.gpt.model.transformer.wte(tokens)
        prefix_projections = self.clip_project(clip_feat).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt.model(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def predict(self, img, temperature=1, entry_length=67, beam_size=5, stop_token="</s>", device="cuda"):
        stop_token_index = self.gpt.tokenizer.encode(stop_token)[0]
        clip_feat = self.clip(img)
        batch_generated = self.clip_project(clip_feat).view(-1, self.prefix_length, self.gpt_embedding_size)
        batch_results = []
        B, L, D = batch_generated.shape
        for i in range(batch_generated.shape[0]):
            generated = batch_generated[i, :, :].unsqueeze(0)
            seq_lengths = torch.ones(beam_size, device=device)
            is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
            tokens = None
            scores = None
            for _ in range(entry_length):
                outputs = self.gpt.model(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                logits = logits.softmax(-1).log()
                if scores is None:
                    scores, next_tokens = logits.topk(beam_size, -1)
                    # print("scores:",scores.shape,next_tokens.shape) # (B,bm), (B,bm)
                    generated = generated.expand(beam_size, *generated.shape[1:])
                    # print("generate1",generated.shape) # (bm,10,1024)
                    next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                    if tokens is None:
                        tokens = next_tokens
                    else:
                        tokens = tokens.expand(beam_size, *tokens.shape[1:])
                        tokens = torch.cat((tokens, next_tokens), dim=1)
                else:
                    logits[is_stopped] = -float(np.inf)
                    logits[is_stopped, 0] = 0
                    scores_sum = scores[:, None] + logits
                    seq_lengths[~is_stopped] += 1
                    scores_sum_average = scores_sum / seq_lengths[:, None]
                    scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                        beam_size, -1
                    )
                    # print("scores2:",scores_sum.shape,next_tokens.shape) # (bm,32000), (bm)
                    next_tokens_source = next_tokens // scores_sum.shape[1]
                    seq_lengths = seq_lengths[next_tokens_source]
                    next_tokens = next_tokens % scores_sum.shape[1]
                    next_tokens = next_tokens.unsqueeze(1)
                    tokens = tokens[next_tokens_source]
                    tokens = torch.cat((tokens, next_tokens), dim=1)
                    generated = generated[next_tokens_source]
                    scores = scores_sum_average * seq_lengths
                    is_stopped = is_stopped[next_tokens_source]
                next_token_embed = self.gpt.model.transformer.wte(next_tokens.squeeze()).view(
                    generated.shape[0], 1, -1
                )
                # print("generated3:",generated.shape, next_token_embed.shape) # (bm, len, 1024) (bm, 1, 1024)
                generated = torch.cat((generated, next_token_embed), dim=1)
                is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
                if is_stopped.all():
                    break

            scores = scores / seq_lengths
            output_list = tokens.cpu().numpy()
            output_texts = [
                self.gpt.tokenizer.decode(output[: int(length)])
                for output, length in zip(output_list, seq_lengths)
            ]
            order = scores.argsort(descending=True)
            output_texts = [output_texts[i] for i in order]
            batch_results.append(output_texts[0].replace(stop_token, ''))

        return batch_results

    def predict_wo_beamsearch(self, img, entry_length=67, top_p=0.8, temperature=1.0, stop_token: str = "</s>", device="cuda"):
        stop_token_index = self.gpt.tokenizer.encode(stop_token)[0]
        clip_feat = self.clip(img)
        batch_generated = self.clip_project(clip_feat).view(-1, self.prefix_length, self.gpt_embedding_size)
        generated = batch_generated
        generated_list = []
        filter_value = -float("Inf")

        tokens = None
        with torch.no_grad():
            for _ in range(entry_length):
                outputs = self.gpt.model(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    nnf.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(1)
                print("next_token", next_token.shape)
                next_token_embed = self.gpt.model.transformer.wte(next_token)
                print("next_token_embed", next_token_embed.shape)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                flag = stop_token_index == next_token
                if flag.all():
                    break

                print(tokens.shape)
                output_list = list(tokens.cpu().numpy())
                output_text = [self.gpt.tokenizer.decode(o) for o in output_list]

        return output_text


def train(dataset, dataloader, model, args,
          lr: float = 2e-5, warmup_steps: int = 5000, output_dir: str = ".", output_prefix: str = ""):
    train_dataset = dataset[0]
    train_dataloader = dataloader[0]

    device = torch.device('cuda:0')
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )
    # save_config(args)
    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        model.train()
        for idx, (img, tokens, mask, image_id) in enumerate(tqdm(train_dataloader)):
            model.zero_grad()
            tokens, mask = tokens.to(device), mask.to(device)
            outputs = model(tokens, img, mask)
            logits = outputs.logits[:, train_dataset.prefix_length - 1: -1]
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if DEBUG and idx >= 100:
                break

            if (idx + 1) % 100 == 0:
                print(f"[{idx+1}] loss: {loss.item()}")
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}_latest.pt"),
                )
        if epoch % args.save_every == 0 or epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )

        # eval
        res = {"loss": loss.item()}
        if args.wandb:
            wandb.log(res)
        if args.eval:
            eval(dataset, dataloader, model, args)
    return model


def eval(dataset, dataloader, model, args, split="val"):
    f = 1 if split == "val" else 2
    _dataloader = dataloader[f]
    model.eval()
    results = []
    with torch.no_grad():
        for idx, (img, tokens, mask, image_id, all_captions) in enumerate(tqdm(_dataloader)):
            # for i, _img in enumerate(img):
            output = model.predict(img)
            # output = model.predict_wo_beamsearch(img)
            for i, _ in enumerate(image_id):
                img_id = int(image_id[i].cpu().numpy())
                print(f"[{img_id}] {output[i]}")
                results.append({"image_id": img_id, "caption": output[i]})

            if DEBUG and idx >= 10:
                break

    with open(f"{args.mapping_type}_eval.json", "w") as f:
        json_res = json.dumps(results)
        f.write(json_res)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/coco/oscar_split_train.pkl')
    parser.add_argument('--out_dir', default='./checkpoints')
    parser.add_argument('--prefix', default='coco_prefix', help='prefix for saved filenames')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--bs', type=int, default=40)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--mapping_type', type=str, default='mlp', help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    if args.wandb:
        wandb.init(project="XXX", name="ClipCap_STAIR")

    prefix_length = args.prefix_length
    prefix_dim = 640 if args.is_rn else 512
    model = JapaneseClipCap(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                            num_layers=args.num_layers, mapping_type=args.mapping_type)

    batch_size = args.bs
    train_dataset, train_dataloader = None, None
    train_dataset = StairCaptionDataset(tokenizer=model.gpt.tokenizer, clip_preprocess=model.clip.preprocess, split="train", prefix_length=args.prefix_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset, val_dataloader = None, None
    val_dataset = StairCaptionDataset(tokenizer=model.gpt.tokenizer, clip_preprocess=model.clip.preprocess, split="val", prefix_length=args.prefix_length)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    test_dataset, test_dataloader = None, None
    test_dataset = StairCaptionDataset(tokenizer=model.gpt.tokenizer, clip_preprocess=model.clip.preprocess, split="test", prefix_length=args.prefix_length)
    test_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    dataset = (train_dataset, val_dataset, test_dataset)
    dataloader = (train_dataloader, val_dataloader, test_dataloader)
    if not args.eval:
        model_path = os.path.join(args.out_dir, f"{args.prefix}_best.pt")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
        model = model.cuda()
        train(dataset, dataloader, model, args, output_dir=args.out_dir, output_prefix=args.prefix)
    else:
        model_path = os.path.join(args.out_dir, f"{args.prefix}_latest.pt")
        model.load_state_dict(torch.load(model_path))
        model = model.cuda()
        eval(dataset, dataloader, model, args)


if __name__ == '__main__':
    main()
