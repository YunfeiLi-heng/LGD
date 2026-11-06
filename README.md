<div align="center">

<h1>Location-Guided Discriminative Learning for Face Forgery Detection</h1>

<div>
    <a href='https://yunfeili-heng.github.io//' target='_blank'>Yunfei Li</a>,
    <a href='https://homepage.hit.edu.cn/guopuzhu//' arget='_blank'>Guopu Zhu<sup>*</sup></a>,
  <a href='https://yuezunli.github.io//' target='_blank'>Yuezun Li</a>,
    Yicong Zhou, Hongli Zhang, Xingpeng Zhang
</div>
<div>
    Harbin Institute of Technology;
</div>


</div>

## 📢 Updates
- [11/2026] The test code is released.

## 🪧 Introduction
This is the official implementation of *Location-Guided Discriminative Learning for Face Forgery Detection*. In this paper ,we describe a Location-Guided Discriminative Learning Framework (LGD), which
integrates pretrained vision-language models (VLMs) with forgery location estimation for face forgery detection. Extensive experiments on multiple benchmark datasets demonstrate that LGD achieves superior performance compared with state-of-the-art methods under cross-dataset evaluation settings.

The complete official PyTorch implementation will be made available as soon as the paper is accepted.

<div align="center">
<img src='./overview.png' width='100%'>
</div>




## 🛠️ Installation


### Environment
Ensure your environment meets the following requirements:

- 🐍 Python 3.9
- 🔥 PyTorch 1.11
- 🚀 CUDA 11.3
  
Install dependencies:

```bash
git clone https://github.com/YunfeiLi-heng/LGD.git
cd LGD_Forgery
conda create -n LGD python=3.9
conda activate LGD
sh install.sh
```

## 🏋️ Pretrained Models
We also provide the pretrained models.

| Model                                                                                                                                                      | Description
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------| :----------
| [pretrained-r34](https://stuouceducn-my.sharepoint.com/:u:/g/personal/liyunfei_x_stu_ouc_edu_cn/EbDYY5UOmbBAjQOy8mOH0EQBaNtDJ8idUdhjvfhbzT55sg?e=fZTLg5) | Trained on `facial_components` and `facial_attributes` with `resnet34` backbone.


