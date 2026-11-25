# MM_VQA_CNNMLP

**CNN+MLP on Sort-of-CLEVR - convergence study (PAI-SG Multimodal Wave 2)**

Part of the *Physical AI* reference series of the **Statera Working Group**. One model of the visual question answering convergence comparison in [MM_Wave2_SortOfCLEVR](https://github.com/Statera-Workshop/MM_Wave2_SortOfCLEVR).

## Method

Baseline of the Relation Network paper: CNN (4 x conv 24, stride 2, BN) -> 5 x 5 x 24, flattened and concatenated with the question -> MLP 256-256-10.

**Common setup** (identical for the five Wave 2 repositories; full comparison in [MM_Wave2_SortOfCLEVR](https://github.com/Statera-Workshop/MM_Wave2_SortOfCLEVR)).
Sort-of-CLEVR generated with `code/gen_sortofclevr.py` (no download): 75 x 75 RGB images with 6 objects (one per
colour, square or circle), 10 questions per image as 11-d vectors (colour, relational or not, subtype), 10 answers;
5 non-relational questions (shape, left?, top?) and 5 relational ones (shape of the closest / furthest object, count of
the same shape). Train 10,000 images (100,000 questions), test 2,000 (20,000). All models from scratch: Adam 1e-4
constant, batch 64, budgets 20 and 60 epochs, seeds 777 / 778, fp32, deterministic cuDNN. Test accuracy is evaluated
twice per epoch below 30 epochs and every epoch otherwise, each time in plain eval mode and after re-estimating the
BatchNorm statistics on 50 training batches (the series' evaluation practice); the two agree within 0.004 here.
Reported value = last evaluation.

## Results (Sort-of-CLEVR test, mean of seeds; range in brackets)

| Model | Epochs | Seeds | Final accuracy, plain (range) | Final, BN re-estimated (range) | Relational | Non-relational | Peak (epoch) | E90 (epochs) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CNN+MLP | 20 | 2 | 0.654 (0.007) | 0.654 (0.006) | 0.704 | 0.604 | 0.677 (10.2) | 2.8 |
| CNN+MLP | 60 | 2 | 0.628 (0.000) | 0.629 (0.001) | 0.678 | 0.579 | 0.676 (8.5) | 1.5 |

**Finding.** Peaks at about epoch 9 and then overfits; without a relational module it stays far below the others.

## Weights

Release **v1.0** (4 files, SHA-256 in `SHA256SUMS.txt`): `sortofclevr_cnnmlp_e20_s777.pth`, `sortofclevr_cnnmlp_e20_s778.pth`, `sortofclevr_cnnmlp_e60_s777.pth`, `sortofclevr_cnnmlp_e60_s778.pth`.

```python
import torch, sys; sys.path.insert(0, 'code'); import mm_c as M
m = M.build('cnnmlp')
m.load_state_dict(torch.load('sortofclevr_cnnmlp_e60_s777.pth', weights_only=True), strict=True)
```

## Run

```bash
cd code && bash smoke_c.sh        # generates the data, gates, 1-epoch smoke
python mm_c.py --model cnnmlp --epochs 60 --seed 777
python analyze_c.py
```
