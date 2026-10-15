"""Gradio UI for a trained DINOv2-Small insect linear probe.

Install: pip install gradio torch torchvision transformers pillow
Run from DETR directory: python dinov2_insect_ui.py
Open http://127.0.0.1:7860
"""
import argparse
from pathlib import Path

import gradio as gr
import torch
from torch import nn
from torchvision import transforms
from transformers import AutoModel


TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


class InsectPredictor:
    def __init__(self, checkpoint_path: Path):
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"체크포인트가 없습니다: {checkpoint_path.resolve()}")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        required = {"head_state_dict", "model_id", "class_names", "feature_dim", "image_size"}
        missing = required.difference(checkpoint)
        if missing:
            raise KeyError(f"체크포인트 필수 키가 없습니다: {sorted(missing)}")
        if int(checkpoint["image_size"]) != 224:
            raise ValueError("이 UI는 224px로 학습된 DINOv2 체크포인트를 지원합니다.")
        self.class_names = list(checkpoint["class_names"])
        feature_dim = int(checkpoint["feature_dim"])
        model_id = str(checkpoint["model_id"])
        print(f"Device: {self.device} | Model: {model_id}")
        self.backbone = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.backbone.requires_grad_(False)
        self.head = nn.Linear(feature_dim, len(self.class_names)).to(self.device)
        self.head.load_state_dict(checkpoint["head_state_dict"], strict=True)
        self.head.eval()
        print(f"Loaded {len(self.class_names)} classes from {checkpoint_path}")

    @torch.inference_mode()
    def predict(self, image):
        if image is None:
            return {}, "이미지 파일을 업로드해 주세요."
        pixel_values = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(self.device)
        features = self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0, :]
        logits = self.head(features)
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
        ordered = sorted(enumerate(probs), key=lambda item: item[1], reverse=True)
        top3 = ordered[:3]
        lines = ["### 예측 Top 3", "| 순위 | 클래스 | Softmax 점수 |", "|---:|---|---:|"]
        for rank, (idx, probability) in enumerate(top3, 1):
            lines.append(f"| {rank} | `{self.class_names[idx]}` | **{probability * 100:.2f}%** |")
        lines.append("\n*10종 중 하나를 반드시 선택합니다. 점수는 보정된 실제 정답 확률이 아니며, 학습하지 않은 종류의 곤충에서는 신뢰하기 어려울 수 있습니다.*")
        scores = {name: float(prob) for name, prob in zip(self.class_names, probs)}
        return scores, "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="DINOv2 곤충 10종 분류 Web UI")
    parser.add_argument("--checkpoint", type=Path, default=Path("dinov2_results/best_linear_probe.pth"))
    parser.add_argument("--host", default="127.0.0.1", help="다른 기기 접속 시 0.0.0.0 사용 (보안 유의)")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    predictor = InsectPredictor(args.checkpoint)

    with gr.Blocks(title="DINOv2 곤충 분류기") as demo:
        gr.Markdown("# 🪲 DINOv2-Small 곤충 분류기\n곤충 사진을 올리면 학습된 10개 클래스의 예측 점수를 보여줍니다.")
        with gr.Row():
            image_input = gr.Image(type="pil", label="곤충 사진 업로드", sources=["upload", "clipboard"])
            with gr.Column():
                predict_btn = gr.Button("🔎 곤충 분류하기", variant="primary")
                top3_md = gr.Markdown("이미지를 선택하고 **곤충 분류하기** 버튼을 눌러주세요.")
                scores = gr.Label(label="클래스별 점수", num_top_classes=10)
        gr.Markdown("학습 데이터: 자연사박물관 딱정벌레류 10종. 야외에서 촬영한 다른 종류의 곤충에는 정확도가 보장되지 않습니다.")
        predict_btn.click(fn=predictor.predict, inputs=image_input, outputs=[scores, top3_md])
    demo.queue().launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()
