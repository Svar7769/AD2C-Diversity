# ADiCo — Adaptive Diversity Control

This repository introduces **ADiCo (Adaptive Diversity Control)**, a framework for enhancing **multi-agent reinforcement learning (MARL)** by dynamically managing behavioral diversity. It extends the work of **[DiCo: Controlling Behavioral Diversity in Multi-Agent Reinforcement Learning (Bettini et al., ICML 2024)](https://openreview.net/forum?id=qQjUgItPq4)** by introducing a novel adaptive control mechanism based on Extremum Seeking Control (ESC). This allows AD2C to intelligently balance exploration and exploitation to solve complex, heterogeneous MARL tasks like multi-agent navigation.

<p align="center">
<img src="https://github.com/Svar7769/AD2C/blob/main/src/ESC_blockDiagram%20-%20AD2C_TITLE_Simplified_v2_page-0001.jpg" alt="ES Controller FlowChart">
</p>

## Install

These instructions were tested on Ubuntu with an NVIDIA GPU. You need Conda, Git, and a C++ compiler installed.

Create an environment:

```bash
conda create -n adico python=3.12 -y
conda activate adico
python -m pip install --upgrade pip
```

Download the project and its dependencies:

```bash
mkdir ADiCo
cd ADiCo

git clone --branch ultimate https://github.com/Svar7769/AD2C-Diversity.git adico
git clone --branch adico https://github.com/Svar7769/BenchMARL.git BenchMARL
git clone --branch adico-compatible https://github.com/Svar7769/torchRL.git torchRL
git clone --branch adico-compatible https://github.com/Svar7769/tensordict.git tensordict
git clone --branch adico https://github.com/Svar7769/VectorizedMultiAgentSimulator.git VectorizedMultiAgentSimulator
```

Install TensorDict first, then the remaining packages:

```bash
python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -e ./tensordict \
  "torch==2.10.0+cu128"

python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -e ./torchRL \
  -e ./VectorizedMultiAgentSimulator \
  -e ./BenchMARL \
  -e ./adico \
  "torch==2.10.0+cu128" \
  "torchvision==0.25.0+cu128"

cd adico
python -m pip check
```

## Run

Run the following commands from the `adico` directory with the environment activated. They save results locally without requiring a Weights & Biases account.

Balance:

```bash
python -m het_control.run_tasks.run_balance \
  experiment.create_json=true \
  experiment.render=false \
  'experiment.loggers=[]'
```

Navigation:

```bash
python -m het_control.run_tasks.run_navigation \
  experiment.create_json=true \
  experiment.render=false \
  'experiment.loggers=[]'
```

Both runners use ESC by default. Keep `experiment.create_json=true` when running without loggers so evaluation and ESC updates take place.

Training settings are in `het_control/conf/`. You can also add settings to a command, such as `seed=0`, `model.desired_snd=0.5`, or `task.n_agents=3`.

Checkpoints are saved under `model_checkpoint/balance_ippo/` and `model_checkpoint/navigation_ippo/`.

## 🙌 Acknowledgements

This repository builds upon:

* [DiCo](https://openreview.net/forum?id=qQjUgItPq4) - Original diversity control framework
* [BenchMARL](https://github.com/matteobettini/BenchMARL) - Multi-agent benchmarking library
* [TorchRL](https://github.com/pytorch/rl) - Reinforcement learning framework
* [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator) - Vectorized multi-agent simulator

Special thanks to the ProrokLab team for their foundational work on behavioral diversity in MARL.

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

## 📄 License

This project is licensed under the same terms as the original DiCo repository. Please refer to the LICENSE file for details.
