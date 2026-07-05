# Morphology-Aware Semantic Graph Learning for Prostate Cancer Grading

PyTorch implementation of a weakly supervised WSI prostate cancer grading framework with:

- ResNet50 patch feature extraction
- IRM-based top-M patch refinement
- Medical concept anchor alignment using cosine similarity
- Learnable Gabor morphology-aware graph construction
- Semantic evidence propagation
- Ordinal focal loss and graph smoothness regularization

## Repository structure

```text
morph_semantic_graph/
├── configs/config.yaml
├── datasets/wsi_bag_dataset.py
├── models/
│   ├── gabor_graph.py
│   ├── losses.py
│   ├── model.py
│   └── semantic_alignment.py
├── scripts/
│   ├── train.py
│   └── inference.py
├── utils/metrics.py
└── requirements.txt
```

## Expected feature file format

Each slide should be stored as a `.pt` file containing:

```python
{
  "features": Tensor[N, d],        # ResNet50 patch features h_i
  "fm_features": Tensor[N, d_fm],  # UNI/Foundation model features u_i for IRM scoring
  "coords": Tensor[N, 2],          # patch coordinates
  "patches": Tensor[N, 3, H, W],   # optional raw patches for learnable Gabor graph
  "label": int                     # Grade Group class index
}
```

A CSV file should contain:

```csv
slide_id,path,label
slide_001,/path/to/slide_001.pt,0
```

## Training

```bash
python scripts/train.py --config configs/config.yaml
```
