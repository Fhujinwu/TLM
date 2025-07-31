<h1 align="center">
     <br>Test-Time Learning for Large Language Models
<p align="center">
    <a href="https://arxiv.org/pdf/2505.20633">
        <img alt="Static Badge" src="https://img.shields.io/badge/Paper-Arxiv-red">
    </a>
    <a href="https://huggingface.co/datasets/Jinwu01/AdaptEval/">
        <img alt="Static Badge" src="https://img.shields.io/badge/HFDataset-TLM-yellow">
    </a>
</p>

## 🔥News
- *2025-07-31*： Update AdaptEval benchmark.
- *2025-05-27*: We have released our paper on Arxiv.
- *2025-05-01*: TLM is accepted by ICML2025.

## 🚀Quick Start 
```bash
## clone our repo
git clone https://github.com/Fhujinwu/TLM.git
cd TLM
## install TLM environment
conda create --name tlm --yes python=3.8
conda activate tlm
pip install -e ".[torch,metrics]" --no-build-isolation
```
## 🗂 Data Selection

Download：https://huggingface.co/datasets/Jinwu01/AdaptEval

## 🔨 Training
TODO

## ⚖️ Evaluation
TODO

## 💬 Citation
Thanks for the open-source code of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

If you find our work interesting and meaningful, welcome to give a 🌟 to our repo and cite our paper.

```text
@inproceedings{hutest,
  title={Test-Time Learning for Large Language Models},
  author={Hu, Jinwu and Zhang, Zitian and Chen, Guohao and Wen, Xutao and Shuai, Chao and Luo, Wei and Xiao, Bin and Li, Yuanqing and Tan, Mingkui},
  booktitle={Forty-second International Conference on Machine Learning}
}
```

## Star History

![Star History Chart](https://api.star-history.com/svg?repos=Fhujinwu/TLM&type=Date)
