import argparse
import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from datasets.wsi_bag_dataset import WSIBagDataset
from models.model import MorphologyAwareSemanticGraph
from models.losses import OrdinalFocalLoss, graph_smoothness_loss
from utils.metrics import compute_metrics


def load_concepts(num_concepts, text_dim, device):
    """Replace this with CONCH concept embeddings loaded from disk."""
    concept_embeds = torch.randn(num_concepts, text_dim)
    concept_embeds = torch.nn.functional.normalize(concept_embeds, dim=-1)
    return concept_embeds.to(device)


def main(cfg):
    device = torch.device(cfg["train"].get("device", "cuda") if torch.cuda.is_available() else "cpu")
    train_ds = WSIBagDataset(cfg["data"]["train_csv"])
    val_ds = WSIBagDataset(cfg["data"]["val_csv"])
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = MorphologyAwareSemanticGraph(
        visual_dim=cfg["model"]["visual_dim"],
        fm_dim=cfg["model"]["fm_dim"],
        text_dim=cfg["model"]["text_dim"],
        num_classes=cfg["data"]["num_classes"],
        num_concepts=cfg["data"]["num_concepts"],
        top_m=cfg["data"]["top_m"],
        hidden_dim=cfg["model"]["hidden_dim"],
        graph_k=cfg["model"]["graph_k"],
        propagation_steps=cfg["model"]["propagation_steps"],
        gabor_orientations=cfg["model"]["gabor_orientations"],
        gabor_frequencies=cfg["model"]["gabor_frequencies"],
        tau_p=cfg["model"]["tau_p"],
        tau_g=cfg["model"]["tau_g"],
    ).to(device)

    concept_embeds = load_concepts(cfg["data"]["num_concepts"], cfg["model"]["text_dim"], device)
    criterion = OrdinalFocalLoss(gamma=cfg["train"]["focal_gamma"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    best_ck = -1.0
    os.makedirs(os.path.dirname(cfg["train"]["save_path"]), exist_ok=True)

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            features = batch["features"].squeeze(0).to(device)
            fm_features = batch["fm_features"].squeeze(0).to(device)
            coords = batch["coords"].squeeze(0).to(device)
            patches = batch["patches"].squeeze(0).to(device) if batch["patches"] is not None else None
            label = batch["label"].to(device)

            out = model(features, fm_features, coords, patches, concept_embeds)
            loss_cls = criterion(out["logits"].unsqueeze(0), label)
            loss_smooth = graph_smoothness_loss(out["evidence_prop"], out["adjacency"])
            loss = loss_cls + cfg["train"]["lambda_smooth"] * loss_smooth

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        y_true, y_prob = [], []
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].squeeze(0).to(device)
                fm_features = batch["fm_features"].squeeze(0).to(device)
                coords = batch["coords"].squeeze(0).to(device)
                patches = batch["patches"].squeeze(0).to(device) if batch["patches"] is not None else None
                label = batch["label"].item()
                out = model(features, fm_features, coords, patches, concept_embeds)
                prob = torch.softmax(out["logits"], dim=-1).cpu().numpy()
                y_true.append(label)
                y_prob.append(prob)
        metrics = compute_metrics(y_true, y_prob)
        print(f"Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, val={metrics}")

        if metrics["ck"] > best_ck:
            best_ck = metrics["ck"]
            torch.save({"model": model.state_dict(), "config": cfg}, cfg["train"]["save_path"])
            print(f"Saved best checkpoint: CK={best_ck:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
