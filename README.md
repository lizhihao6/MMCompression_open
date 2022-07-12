<div align="center">
  <img src="resources/mmcomp-logo.png" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab website</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab platform</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

<br />

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mmcompression)](https://pypi.org/project/mmcompression/)
[![PyPI](https://img.shields.io/pypi/v/mmcompression)](https://pypi.org/project/mmcompression)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmcompression.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmcompression/workflows/build/badge.svg)](https://github.com/open-mmlab/mmcompression/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmcompression/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmcompression)
[![license](https://img.shields.io/github/license/open-mmlab/mmcompression.svg)](https://github.com/open-mmlab/mmcompression/blob/master/LICENSE)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmcompression.svg)](https://github.com/open-mmlab/mmcompression/issues)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmcompression.svg)](https://github.com/open-mmlab/mmcompression/issues)

[📘Documentation](https://mmcompression.readthedocs.io/en/latest/) |
[🛠️Installation](https://mmcompression.readthedocs.io/en/latest/get_started.html) |
[👀Model Zoo](https://mmcompression.readthedocs.io/en/latest/model_zoo.html) |
[🆕Update News](https://mmcompression.readthedocs.io/en/latest/changelog.html) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmcompression/issues/new/choose)

</div>

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

## Introduction

MMSegmentation is an open source semantic segmentation toolbox based on PyTorch.
It is a part of the [OpenMMLab](https://openmmlab.com/) project.

The master branch works with **PyTorch 1.5+**.

![demo image](resources/seg_demo.gif)

<details open>
<summary>Major features</summary>

- **Unified Benchmark**

  We provide a unified benchmark toolbox for various semantic compression methods.

- **Modular Design**

  We decompose the learning based compression framework into different components and one can easily construct a customized AI compression framework by combining different modules.

- **Support of multiple methods out of box**

  The toolbox directly supports popular and contemporary AI compression frameworks, *e.g.* NLAIC, TinyLIC, etc.

- **High efficiency**

  The training speed is faster than or comparable to other codebases.

</details>

## What's New

v0.0.1 was released in 7/11/2022:

- First Commit

Please refer to [changelog.md](docs/en/changelog.md) for details and release history.

## Installation

Please refer to [get_started.md](docs/en/get_started.md#installation) for installation and [dataset_prepare.md](docs/en/dataset_prepare.md#prepare-datasets) for dataset preparation.

## Get Started

Please see [train.md](docs/en/train.md) and [inference.md](docs/en/inference.md) for the basic usage of MMSegmentation.
There are also tutorials for:

- [customizing dataset](docs/en/tutorials/customize_datasets.md)
- [designing data pipeline](docs/en/tutorials/data_pipeline.md)
- [customizing modules](docs/en/tutorials/customize_models.md)
- [customizing runtime](docs/en/tutorials/customize_runtime.md)
- [training tricks](docs/en/tutorials/training_tricks.md)
- [useful tools](docs/en/useful_tools.md)

A Colab tutorial is also provided. You may preview the notebook [here](demo/MMSegmentation_Tutorial.ipynb) or directly [run](https://colab.research.google.com/github/open-mmlab/mmcompression/blob/master/demo/MMCompression_Tutorial.ipynb) on Colab.

## Benchmark and model zoo

Results and models are available in the [model zoo](docs/en/model_zoo.md).

Supported VAEs:

- [x] Non-Local (TIP'2021)
- [x] Swin Transformer (Arxiv'2022)

Supported context model:

- [x] Masked3DCNN (TIP'2021)
- [x] MCM (Arxiv'2022)

Supported learining based image compression methods:
- [x] [NLAIC (TIP'2021)](configs/nlaic)
- [x] [TinyLIC (Arxiv'2022)](configs/tinylic)

Supported conventional image compression methods:

- [x] JPEG
- [x] HM (BPG)
- [x] VVC


Supported datasets:

- [x] [Flicker2W](https://github.com/open-mmlab/mmcompression/blob/master/docs/en/dataset_prepare.md#flicker2w)
- [x] [Kodak24](https://github.com/open-mmlab/mmcompression/blob/master/docs/en/dataset_prepare.md#kodak24)
- [x] [IEEE1857](https://github.com/open-mmlab/mmcompression/blob/master/docs/en/dataset_prepare.md#ieee1857)

## FAQ

Please refer to [FAQ](docs/en/faq.md) for frequently asked questions.

## Contributing

We appreciate all contributions to improve MMCompression. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMCompression is an open source project that welcome any contribution and feedback.
We wish that the toolbox and benchmark could serve the growing research
community by providing a flexible as well as standardized toolkit to reimplement existing methods
and develop their own new AI compression methods.

## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@misc{mmcomp2022,
    title={{MMCompression}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
    author={MMSegmentation Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmcompression}},
    year={2022}
}
```

## License

MMCompression is released under the Apache 2.0 license, while some specific features in this library are with other licenses. Please refer to [LICENSES.md](LICENSES.md) for the careful check, if you are using our code for commercial matters.

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition, and understanding toolbox.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning toolbox and benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab model compression toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab fewshot learning toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab Model Deployment Framework.
- [MMCompression](https://github.com/open-mmlab/mmcompression): OpenMMLab AI-based compression toolbox and benchmark.
