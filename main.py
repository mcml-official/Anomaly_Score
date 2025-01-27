import torch

from anomaly_score import AnomalyScore
from anomaly_score_i import AnomalyScore_i

def main():
    device = "cuda"

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
    model.eval()
    model.to(device)

    images_real = torch.randn(128, 3, 224, 224).to(device)
    images_generated = torch.randn(128, 3, 224, 224).to(device)

    anomaly_score = AnomalyScore(model)
    result = anomaly_score(images_real, images_generated)

    print("Evaluate a generative model")
    print(result)

    anomaly_score_i = AnomalyScore_i(model)
    result_i = anomaly_score_i(images_generated)
    print("Evaluate an individual image")
    print(result_i)

if __name__ == "__main__":
    main()
