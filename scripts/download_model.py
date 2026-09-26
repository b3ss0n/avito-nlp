"""Однократная загрузка модели эмбеддингов в папку models/.

Запуск: uv run python scripts/download_model.py

Модель: intfloat/multilingual-e5-small (MIT, ~470 МБ, 384-мерные векторы).
Скачиваем веса один раз, дальше пайплайн грузит модель из models/ и работает
полностью офлайн (HF_HUB_OFFLINE=1) - никаких обращений к внешним API.
Сами веса в репозиторий не кладём: GitHub не принимает файлы больше 100 МБ.
"""

from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_DIR = Path("models") / "multilingual-e5-small"


def main() -> None:
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=MODEL_DIR,
        # нужны только веса в safetensors, конфиги и токенизатор;
        # onnx/openvino-версии и дубли весов в других форматах не качаем
        allow_patterns=["*.json", "*.safetensors", "*.txt", "sentencepiece.bpe.model",
                        "1_Pooling/*"],
    )
    print(f"модель сохранена в {MODEL_DIR}")


if __name__ == "__main__":
    main()
