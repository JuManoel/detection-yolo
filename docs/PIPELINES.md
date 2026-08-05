# Pipelines

Tres pipelines con **arquitectura por capas**. Dependencias solo hacia abajo:

```
CLI → Application → Domain
                 ↘ Infrastructure → Domain
```

| Capa               | Rol                                              |
| ------------------ | ------------------------------------------------ |
| **CLI**            | Args, logging de entrada, exit codes             |
| **Application**    | Orquestación del caso de uso                     |
| **Domain**         | Reglas puras (bbox, configs, specs) — sin I/O ML |
| **Infrastructure** | Kaggle, COCO HTTP, disco, GPU, Ultralytics       |

`pipelines/shared/` concentra domain/infra comunes. Cada pipeline tiene sus propias capas.

## Comandos

```bash
uv sync
uv run download   # Kaggle + COCO bird → data/datasets/yolo
uv run train      # yolo26n / yolo26m / yolo26x
uv run metrics    # val + mAP / precision / recall → runs/metrics/
```

También: `python -m pipelines.download` (igual para `train` / `metrics`).

## Flujo

```
Kaggle (4 datasets) ─┐
                     ├─→ raw/ → convertidores (clase 0) → merge 80/20 → yolo/
COCO bird only ──────┘                                              │
                                                                    ▼
                                              train (YOLO26 n/m/x) → runs/detect/
                                                                    │
                                                                    ▼
                                                              metrics → runs/metrics/
```

## Entradas / salidas

| Pipeline   | Entrada                         | Salida                                      |
| ---------- | ------------------------------- | ------------------------------------------- |
| `download` | `data/datasets.txt`, Kaggle API | `data/datasets/raw/`, `data/datasets/yolo/` |
| `train`    | `data/datasets/yolo/data.yaml`  | `runs/detect/<model>/weights/best.pt`       |
| `metrics`  | `best.pt` + yaml de val         | `runs/metrics/<model>.json`, `summary.md`   |

## Estructura de carpetas

```
pipelines/
  shared/{domain,infrastructure}/
  download/{cli,application,domain,infrastructure}/
  train/{cli,application,domain,infrastructure}/
  metrics/{cli,application,domain,infrastructure}/
```
