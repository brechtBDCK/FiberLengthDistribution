# AI line detector bake-off

Five entry points for the methods discussed:

- `linea_detect.py` — LINEA
- `hawpv3_detect.py` — HAWPv3
- `deeplsd_detect.py` — DeepLSD
- `sold2_detect.py` — SOLD2 via Kornia
- `letr_detect.py` — LETR (official notebook launcher; old repo has no clean maintained CLI)

## Common install

Create a fresh environment first:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install torch torchvision opencv-python numpy
```

Then run one detector, e.g.

```bash
python sold2_detect.py my_image.png --output sold2.png
python deeplsd_detect.py my_image.png --output deeplsd.png
python hawpv3_detect.py my_image.png --output hawp.png --threshold 0.05
python linea_detect.py my_image.png --model l
python letr_detect.py my_image.png
```

## Recommendation

Start with SOLD2 and DeepLSD because they are the easiest to get running. Then try
LINEA and HAWPv3 in separate environments because their repos have more specific
dependency assumptions. LETR is much older and the official distribution is notebook-based.

## Outputs

DeepLSD and SOLD2 save:
- a PNG overlay
- a JSON file with line endpoints

HAWPv3 and LINEA use the authors' official visualization scripts.

## Important dependency note

The *checkpoint files* are relatively small (generally tens to a few hundred MB),
but PyTorch/CUDA environments can consume several GB. HAWP and LETR in particular
may need their own environment because their code targets older PyTorch releases.

## SynthMT SAM3Text + HPO

`sam3text_detect.py` reproduces SynthMT's published best SAM3Text settings:
text prompt `thin line`, its HPO thresholds, and preprocessing. Set `INPUT_PATH`
at its top, accept access for [`facebook/sam3`](https://huggingface.co/facebook/sam3),
authenticate with Hugging Face if needed, then run:

```bash
uv run python sam3text_detect.py
```

It writes `overlay.png`, `instances.tiff` (16-bit instance IDs), and one PNG
per instance into `<image folder>/<image stem>_sam3text_hpo/`. No CLI arguments.

## Why the scripts clone official repos

These research models are not all packaged as stable Hugging Face Transformers
models. Cloning the official code and downloading the authors' official checkpoint
is usually more reproducible than copying model architecture code into a standalone script.
