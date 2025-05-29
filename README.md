# 2025CCF-网易雷火联合基金课题数据库

回收课题相关数据，请注意文件中的readme和各目录要求：

code：存放项目源代码，按功能或模块组织，需安装依赖确保运行。

demo：提供样例数据和示例结果，含演示脚本与配置，可以速览项目功能。

model：存放训练好的模型文件，使用时要确保正确加载。

readme：即当前目录，项目文档说明，如运行指南、使用手册，使用前建议详读。

### 1. 环境配置

首先，创建虚拟环境

```bash
# 使用conda创建环境
conda create -n Retarget python=3.10
conda activate Retarget
```

通过以下命令检查您当前的CUDA版本：

```bash
nvcc -V # example versions: 11.1, 11.8, 12.1, etc.
```

然后，安装相应版本的torch。以下是不同CUDA版本的示例。如果您的CUDA版本未列出，请访问[PyTorch官方网站](https://pytorch.org/get-started/previous-versions)获取安装命令：

```bash
# for CUDA 11.1
pip install torch==1.10.1+cu111 torchvision==0.11.2 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
# for CUDA 11.8
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
# for CUDA 12.1
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
```

最后，安装其余依赖项：

```bash
pip install -r requirements.txt
```



### 2. 下载预训练模型

（1）下载LivePortrait预训练模型

从[Google Drive](https://drive.google.com/drive/folders/1UtKgzKjFAOmZkhNK-OYT0caJ_w2XAnib)中下载预训练模型并放于**Code/LivePortrait/pretrained_weights**文件夹中

（2）下载口型预测预训练模型

从[Google Drive](https://drive.google.com/file/d/1PYIfppWAIVFuO2dWQgIuLrvSMkAYYTE5/view)下载模型并放于**Model**文件夹中



### 3. 运行

（1） 进行换脸

```python
cd code/LivePortrait
python inference.py
```

（2） 对换脸的结果预测MetaHuman Rig

```python
cd ..
python test.py
```

（3） 对部分样本修正口型

```python
python pred_mouth.py
```

（4） 综合换脸的预测结果和修正口型后的结果，并在maya/UE5中渲染结果，保存在“data/render_final”中

（5） 拼接原始视频和渲染的预测结果

```
python concat_videos.py
```

