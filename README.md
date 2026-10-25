# PyFly — Learning Python with a Drosophila MaleCNS Reservoir

PyFly is an experimental Python code-generation project built around neural activity from a simulated **Drosophila MaleCNS connectome**.

The project uses the fly connectome as a recurrent neural reservoir, extracts activity from the simulated network, and trains a GPU PyTorch adapter to predict Python-program tokens. The trained adapter can then generate short Python programs in a true closed loop, where each predicted token is fed back through the MaleCNS simulation before the next prediction.

> PyFly does **not** claim that a biological fly brain was literally taught Python. The successful Phase-3 system keeps the MaleCNS recurrent connectome frozen and trains an external neural adapter on top of its dynamics.

## Final Phase-3 Results

The frozen model was evaluated once on a fresh 24-prompt compositional holdout across five neural seeds.

### Teacher-forced evaluation

| Metric | Result |
|---|---:|
| Top-1 accuracy | **86.71%** |
| Top-2 accuracy | **97.11%** |
| Top-3 accuracy | **99.61%** |
| Macro F1 | **83.25%** |
| Semantic accuracy | **78.96%** |

### True closed-loop generation

| Metric | Result |
|---|---:|
| Mean token accuracy | **76.57%** |
| Exact programs | **32.50%** |
| Valid programs | **80.00%** |
| Syntax-valid programs | **80.00%** |
| EOS completion | **100.00%** |

The final evaluation used **120 closed-loop episodes**: 24 logical prompts × 5 MaleCNS neural seeds.

## Architecture

```text
Python prompt
      ↓
PyFly neural stimulation
      ↓
Real MaleCNS connectome simulation
      ↓
41,111 neural features
      ↓
GPU-trained PyTorch adapter
      ↓
Python token prediction
      ↓
predicted token fed back into MaleCNS
      ↓
next token
      ↓
complete Python program
```

## PyFly Studio V2.2

The project includes a PySide6 desktop interface optimized for a 16-inch 2560×1440 laptop.

Features include:

- animated PyFly / neural-core visualization
- prompt and intent selection
- custom supported prompt-token sequences
- true closed-loop generation
- live token confidences
- generated Python preview
- copy and save-to-`.py` controls
- generation history
- dark and light themes
- frozen benchmark summary
- GPU / model system information

Run it with:

```powershell
python scripts\pyfly_studio_v2_2_1440p.py
```

Or on Windows, use:

```text
launch_pyfly_studio_v2_2_1440p.bat
```

## Requirements

Core dependencies include:

- Python
- PyTorch with CUDA support
- NumPy
- scikit-learn
- joblib
- PySide6
- the `flybrain` / MaleCNS simulation dependency used by the project

The project was developed on Windows 11 with an NVIDIA GPU.

## Project Structure

```text
PyFly/
├── scripts/
│   ├── train_phase3_pyfly_adapter.py
│   ├── generate_phase3_closed_loop.py
│   ├── evaluate_phase3_final_holdout.py
│   └── pyfly_studio_v2_2_1440p.py
├── data/
│   ├── mappings/
│   ├── processed/
│   └── raw/
├── artifacts/
│   ├── models/
│   └── metrics/
├── launch_pyfly_studio_v2_2_1440p.bat
└── README.md
```

Large reproducible neural-feature arrays and plasticity experiment checkpoints are intentionally excluded from Git.

## Running the GUI

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then launch PyFly Studio:

```powershell
python scripts\pyfly_studio_v2_2_1440p.py
```

If PySide6 is missing:

```powershell
pip install PySide6
```

## Scientific Notes

Earlier experiments attempted to modify recurrent MaleCNS synapses directly using several plasticity rules. Those experiments did not reliably improve unseen development performance.

The successful Phase-3 architecture therefore treats MaleCNS as a frozen recurrent computational reservoir and trains a separate neural adapter with GPU backpropagation.

The official final holdout result should not be retroactively used to tune the reported frozen model.

## Repository

This project is published at:

`hicham-4r/pyfly_learn_python`
