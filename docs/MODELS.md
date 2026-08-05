# Models

Fine-tuning de detección Ultralytics **YOLO26** preentrenados en COCO, colapsados a **1 clase: `bird`**.

## Variantes

| Checkpoint   | Rol                          | Batch inicial |
| ------------ | ---------------------------- | ------------- |
| `yolo26n.pt` | Nano — rápido / edge         | 32            |
| `yolo26m.pt` | Medium — equilibrio          | 16            |
| `yolo26x.pt` | Extra-large — máxima calidad | 8             |

Docs: [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/)

## Hiperparámetros por defecto

Definidos en `pipelines/shared/domain/config.py`:

| Parámetro   | Valor    | Notas                                      |
| ----------- | -------- | ------------------------------------------ |
| `epochs`    | `100`    |                                              |
| `patience`  | `10`     | Early stopping = 10 % de las épocas        |
| `optimizer` | `AdamW`  |                                              |
| `imgsz`     | `640`    |                                              |
| `seed`      | `42`     |                                              |
| `data`      | `yolo/data.yaml` | Dataset unificado tras `download`   |

## Entrenamiento

- Device: GPU con más VRAM libre (`nvidia-smi`).
- Si hay OOM CUDA: el batch se divide por 2 y se reintenta.
- Salida: `runs/detect/<model_stem>/weights/best.pt`

```bash
uv run train
```
