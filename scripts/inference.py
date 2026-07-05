import argparse
import os
import sys
import yaml
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.model import MorphologyAwareSemanticGraph


def main(args):
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MorphologyAwareSemanticGraph(
        visual_dim=cfg["model"]["visual_dim"],
        fm_dim=cfg["model"]["fm_dim"],
        text_dim=cfg["model"]["text_dim"],
        num_classes=cfg["data"]["num_classes"],
        num_concepts=cfg["data"]["num_concepts"],
        top_m=cfg["data"]["top_m"],
        graph_k=cfg["model"]["graph_k"],
        propagation_steps=cfg["model"]["propagation_steps"],
        gabor_orientations=cfg["model"]["gabor_orientations"],
        gabor_frequencies=cfg["model"]["gabor_frequencies"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    item = torch.load(args.slide_pt, map_location="cpu")
    concept_embeds = torch.load(args.concepts, map_location="cpu").float().to(device)
    with torch.no_grad():
        out = model(
            item["features"].float().to(device),
            item["fm_features"].float().to(device),
            item["coords"].float().to(device),
            item["patches"].float().to(device),
            concept_embeds,
        )
        prob = torch.softmax(out["logits"], dim=-1)
    print("Probabilities:", prob.cpu().numpy())
    print("Prediction:", int(prob.argmax().item()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--slide_pt", required=True)
    parser.add_argument("--concepts", required=True, help="Path to CONCH concept embedding tensor [K,d]")
    main(parser.parse_args())
