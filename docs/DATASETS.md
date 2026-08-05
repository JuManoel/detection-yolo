# Datasets

Fuentes usadas para entrenar un detector de **una sola clase** (`bird`).
Los slugs de Kaggle viven en [`data/datasets.txt`](../data/datasets.txt).

Tras `uv run download`, el layout queda:

```
data/datasets/
  raw/     # descargas originales (Kaggle + COCO bird)
  yolo/    # dataset unificado Ultralytics (80/20)
    images/{train,val}/
    labels/{train,val}/
    data.yaml
```

## Requisitos

- Token de Kaggle en `.env` (copia de `.env.example`):

  ```bash
  cp .env.example .env
  # pega el token de https://www.kaggle.com/settings/api
  KAGGLE_API_TOKEN=KGAT_...
  ```

  El pipeline `download` carga `.env` vía `python-dotenv` **antes** de importar `kaggle`.
- Espacio en disco: varios GB (CUB ~1 GB, NABirds ~GB, FBD-SV ~GB, COCO bird subset).

## Fuentes

### CUB-200-2011 — `wenewone/cub2002011`

- **Qué es:** Caltech-UCSD Birds-200-2011. 11 788 imágenes, 200 especies.
- **Anotaciones:** 1 bounding box por imagen en `bounding_boxes.txt` (`image_id x y w h` en píxeles), más `images.txt` y `train_test_split.txt`.
- **Uso aquí:** se ignora la especie; todas las cajas pasan a clase `0` (bird). El split oficial no se usa; se re-mezcla en el 80/20 global.
- **Paper / sitio:** [Caltech CUB-200-2011](http://www.vision.caltech.edu/datasets/cub_200_2011/)

### NABirds — `duyminhle/nabirds`

- **Qué es:** ~48 000 fotos de aves de Norteamérica (~400 especies / ~700 categorías visuales).
- **Anotaciones:** `bounding_boxes.txt` (`uuid x y w h`), `images.txt`, `image_class_labels.txt`.
- **Uso aquí:** igual que CUB — solo detección genérica de bird (clase `0`).
- **Origen:** Cornell Lab of Ornithology / [NABirds](https://dl.allaboutbirds.org/nabirds)

### FBD-SV-2024 — `swjtuziwei/fbd-sv-2024`

- **Qué es:** Flying Bird Object Detection in Surveillance Video. 483 clips, ~28 k frames, anotaciones de aves en vuelo (objetos pequeños / difíciles).
- **Anotaciones:** VOC XML (`xmin, ymin, xmax, ymax`), clase `bird`. Estructura típica `images/` + `labels/` (train/val).
- **Uso aquí:** VOC → YOLO normalizado, clase `0`.
- **Código auxiliar oficial:** [Ziwei89/FBD-SV-2024_github](https://github.com/Ziwei89/FBD-SV-2024_github)
- **Paper:** Nature Scientific Data — FBD-SV-2024

### Bird Dataset — `samuelayman/bird-dataset`

- **Qué es:** dataset auxiliar de aves en Kaggle; el layout exacto varía.
- **Uso aquí:** el convertidor `auto` inspecciona la carpeta tras la descarga.
  - Si hay etiquetas YOLO, VOC o COCO → se convierten a clase `0`.
  - Si solo hay clasificación por carpetas **sin** bounding boxes → se **omite** (no se inventan cajas).

### COCO (solo clase bird)

- **Qué es:** subset de COCO 2017 con categoría bird (COCO category id `16`, índice YOLO COCO `14`).
- **Uso aquí:** se descargan annotations y **solo** las imágenes que contienen bird; las cajas se remapean a clase `0`. No se descarga COCO completo (~20 GB).
- **Docs:** [COCO](https://cocodataset.org/) / [Ultralytics COCO](https://docs.ultralytics.com/datasets/detect/coco/)

## Unificación

1. Cada fuente produce muestras `(imagen, cajas YOLO clase 0)`.
2. Se mezclan con seed fijo y se parte **80 % train / 20 % val**.
3. Se escribe `data/datasets/yolo/data.yaml`:

```yaml
path: .../data/datasets/yolo
train: images/train
val: images/val
nc: 1
names:
  0: bird
```
