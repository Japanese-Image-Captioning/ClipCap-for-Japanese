# ClipCap for Japanese

<img width="1180" alt="Screen Shot 2022-10-05 at 0 46 10" src="https://user-images.githubusercontent.com/51681991/193865301-e9565edc-a065-415e-abe8-1329df57dbb5.png">

- [ClipCap: CLIP Prefix for Image Captioning](https://arxiv.org/abs/2111.09734)
- This repository uses [rinna/japanese-gpt2-medium](https://huggingface.co/rinna/japanese-gpt2-medium) and [rinna/japanese-clip-vit-b-16](https://huggingface.co/rinna/japanese-clip-vit-b-16)
  - Thanks for rinna Co., Ltd. ;)

- This code uses [STAIR Captions](http://captions.stair.center/) for training Japanese image captioning model.
  - So you should download STAIR Captions and [COCO datasets](https://cocodataset.org/#download)
 
## Instructions

1. Dowload [datasets](http://captions.stair.center/) & unzip & place it in `./` (`./STAIR-captions`).
2. Download [COCO datasets](https://cocodataset.org/#download) & unzip & place it in `./` (`./train2014`, `./val2014`).

### Train 

```
python train.py
```

### Generate Caption

```
  python train.py --eval
```

### Examples

![16](https://user-images.githubusercontent.com/51681991/193881768-88d354c6-378a-471f-bd5d-a8c619d9ccc2.jpg)

output: 庭の芝生の上に青い服の男性がいる

<br>

![25](https://user-images.githubusercontent.com/51681991/193882047-f704573d-8f82-4a37-a59e-1855a631ee41.jpg)

output: 男性がスケートボードをしている

<br>


![27](https://user-images.githubusercontent.com/51681991/193882119-02ff1e8f-f6d8-47c5-9152-5b06aa5fcdb8.jpg)

output: 窓の外にテントが立ている

<br>

![28](https://user-images.githubusercontent.com/51681991/193882217-f86cc6c0-392f-4924-80a8-e5cb50b9eef5.jpg)

output: パソコンのキーボードの上に黒い猫がいる

<br>

![37](https://user-images.githubusercontent.com/51681991/193882424-362245e5-0f1f-4b86-bfdf-e22e88008eac.jpg)

output: 黒い猫が黒い猫のトイレに頭を付けている

<br>


![0](https://user-images.githubusercontent.com/51681991/193881191-6313a3d6-0bf4-4c01-9d11-cf9bf64acf97.jpg)

output: キッチンの中にたくさんの商品が並べられている 

<br>

![1](https://user-images.githubusercontent.com/51681991/193881208-3a868a9a-79d2-4641-bd75-2ca9b8b615cc.jpg)

output: 猫がテーブルの上にある器を見ている

<br>


![42](https://user-images.githubusercontent.com/51681991/193882591-6250cd31-034a-467a-9082-aa5fed4433a2.jpg)

output: 野球の応援をしている男性と、その後ろで観戦している男性

<br>



## Others
### Licence

This work is licensed under the MIT License. To view a copy of this license, see LICENSE.
