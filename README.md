# ADiCo — Adaptive Diversity Control

ADiCo builds on [DiCo](https://openreview.net/forum?id=qQjUgItPq4) and uses Extremum Seeking Control (ESC) to adjust behavioral diversity during training. The project has been updated for Python 3.12 and PyTorch 2.10. Dependencies are now installed automatically, and checkpoints also save the ESC controller state.

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

## License

See the `LICENSE` file for details.
