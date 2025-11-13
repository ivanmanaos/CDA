# REPORTE_PASO_1: Auditoría rápida de instalación e inferencia local (CDA)

Fecha: 2025-11-11

## 1) Dependencias reales

Archivo `requirements.txt`:

```
torch>=1.8.0
torchvision>=0.9.0
numpy
scipy
pandas
Pillow
matplotlib
tqdm
```

- Versiones fijadas: solo mínimos para torch/torchvision; el resto sin pin.
- Librerías de API ausentes: fastapi, uvicorn, python-multipart, pydantic, aiofiles [no incluidas].

Comando único consolidado:

```powershell
pip install torch>=1.8.0 torchvision>=0.9.0 numpy scipy pandas Pillow matplotlib tqdm
```

Sugerencia: mantener el uso de `requirements.txt` para reproducibilidad.

## 2) CLI y flags de scripts

No hay argparse definido en los scripts. Variables/hiperparámetros se pasan por llamadas a funciones. Resumen y sugerencia mínima:

- `src/initial_train.py`
  - Función: `train_initial_model(train_loader, valid_loader, device, num_epochs=1000, lr=3e-4)`
  - Guarda: `best_initial_model.pth`, `final_initial_model.pth`
  - [No argparse] Sugerencia: añadir parser con `--data_dir`, `--epochs`, `--batch_size`, `--lr`, `--image_size`, `--device`.

- `src/iterative_train.py`
  - Función esperada: `train_with_pseudo_labels(train_loader, unlabeled_loader, valid_loader, checkpoint, device, num_epochs=100, lr=1e-5)`
  - Importa símbolos no definidos en archivo (falta `get_model`, `AdamW`, `CosineAnnealingLR`, `evaluate_model`, y `pred_record`, ver Bugs).
  - [No argparse] Sugerencia: `--data_dir_labeled`, `--data_dir_unlabeled`, `--epochs`, `--batch_size`, `--lr`, `--checkpoint`, `--device`.

- `src/pseudo_labeling.py`
  - Función: `generate_pseudo_labels(model, unlabeled_loader, device, mc_passes=20, threshold=0.005)`
  - [No argparse]. Si se desea usar como script: `--mc_passes`, `--threshold` y rutas de entrada/salida.

- `src/evaluate.py`
  - Funciones: `evaluate_model(model, data_loader, device, criterion)`, `inference_and_save(model, test_loader, device, output_path)`, `plot_predictions(model, data_loader, device)`
  - [No argparse] Sugerencia: `--data_dir`, `--weights`, `--output_csv`, `--image_size`, `--device`.

Ejemplos de uso (propuestos) se incluyen en la sección 7.

## 3) Modelo (`src/model.py`)

- Instanciación: `get_model(device)` crea `efficientnet_b7` con `weights='DEFAULT'` y sustituye la capa final por `AugmentHead(dim=in_features)` que devuelve 12 valores.
- Salida: vector de 12 que representa 6 puntos (A..F) en formato `[xA,yA,xB,yB,...,xF,yF]` normalizados en [0,1].
- Tamaño de entrada: No está fijado aquí; depende del `transform` del dataset. En `utils.get_transform(image_size)` se usa `Resize((image_size, image_size))`. Valor por defecto no provisto; sugerimos 224.
- Normalización: sí, en `utils.get_transform(image_size)` con mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] (estándar ImageNet).
- Cálculo VHS: `calc_vhs(x)` usa distancias euclidianas entre pares (A-B, C-D, E-F) y `vhs = 6 * (AB + CD) / EF`.

[dato no provisto] Número u orden de clases de clasificación explícitas (se deriva implícitamente de umbrales en evaluate: <8.2, 8.2–<10, ≥10).

## 4) Dataset (`src/dataset.py` y `src/utils.py`)

- Labeled: `DogHeartDataset(root, transforms)`
  - Estructura esperada:
    - `<root>/Images/*.png|*.jpg`
    - `<root>/Labels/*.mat` con claves `six_points` (shape [6,2]) y `VHS` (escalar)
  - Transformaciones: aplicadas si se pasa `transforms` (usar `utils.get_transform(image_size)`).
  - Normalización y resize vienen de `utils.get_transform`.
  - Devuelve: `(idx, img_tensor, six_points_normalized, VHS)`.

- Test/Unlabeled: `DogHeartTestDataset(root, transforms)`
  - Estructura: `<root>/Images/*.png|*.jpg`
  - Devuelve: `(img_tensor, filename)`.

- HighConfidenceDataset: envuelve listas `images` y `pseudo_labels` y devuelve `(idx_dummy, img, pseudo_label, pseudo_vhs)`.

- Extensiones válidas: `.png`, `.jpg`.
- [dato no provisto] No existe soporte directo `--data_dir`; hay que instanciar datasets manualmente en scripts.

## 5) Salidas y métricas

- `initial_train.py` guarda:
  - `best_initial_model.pth` cuando `val_acc` mejora.
  - `final_initial_model.pth` al terminar.
- `evaluate.py`:
  - `inference_and_save` guarda CSV con columnas `ImageName, VHS` en la ruta `output_path` especificada por el llamador.
- No se guardan métricas/curvas adicionales por defecto.

Sugerencia mínima:
- Añadir utilidades en `utils.py`:
  - `save_checkpoint(path, model)` y opcionalmente `save_metrics_csv(path, rows)`.

## 6) Bugs/ajustes evidentes

1) `iterative_train.py`
   - Faltan imports y variables:
     - No se importa `get_model`, `AdamW`, `CosineAnnealingLR`, `evaluate_model`, `pred_record` no definido.
   - Fix mínimo propuesto (añadir al inicio y definir `pred_record`):

```python
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import get_model, calc_vhs
from evaluate import evaluate_model
pred_record = torch.zeros([len(train_loader.dataset), 10, 12])  # definir en contexto correcto
```

   - Nota: `len(train_loader.dataset)` sólo es válido cuando `train_loader` está definido. Mover esta línea dentro de la función tras crear `train_loader`.

2) `dataset.HighConfidenceDataset`
   - Usa `calc_vhs` sin importarlo. Agregar:

```python
from model import calc_vhs
```

3) Uso de EfficientNet con `weights='DEFAULT'`
   - Requiere descarga de pesos preentrenados; en entornos sin internet fallará. Alternativa:

```python
models.efficientnet_b7(weights=None)  # o exponer flag --pretrained
```

4) Normalización de puntos
   - En `initial_train.py` se llama `criterion(outputs.squeeze(), points.squeeze())` pero el modelo produce shape `[B,12]`. Asegurar que `points` sea `[B,12]` también; hoy `six_points` es `[6,2]`. Solución ya presente: dataset normaliza a [0,1] y se puede `reshape(-1)`. Verificar que el DataLoader collation no altere shapes.

5) `evaluate.py` métricas de clasificación
   - Clasificación derivada de umbrales (8.2, 10). Correcto, sólo notar que no hay clases explícitas.

6) Creación de carpetas de salida
   - Guardados de modelos en raíz del repo. Sugerimos usar `models/` y `outputs/` y asegurarse de `os.makedirs(..., exist_ok=True)` antes de guardar.

Snippets aplicables:

```python
import os
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), os.path.join('models', 'best_initial_model.pth'))
```

Y para inference:

```python
os.makedirs('outputs', exist_ok=True)
df.to_csv(os.path.join('outputs', 'preds.csv'), index=False)
```

## 7) Comandos de validación (smoke tests)

Asumiendo dataset de ejemplo en `data/sample_real_images` y `data/sample_synthetic_images` para inferencia/plot (no hay pipeline de DataLoader listo en los scripts, así que proveemos un script corto de prueba en consola):

- Entrenamiento mínimal (1 epoch, CPU) con dataset ficticio pequeño (requeriría añadir un runner; hoy no existe argparse). Propuesta rápida para verificar forward/backward:

```powershell
# Activar venv y ejecutar un tiny-check de forward/backward sin datos reales
python - << 'PY'
import torch
from model import get_model, calc_vhs
model = get_model('cpu')
model.train()
images = torch.randn(2,3,224,224)
points = torch.rand(2,12)
vhs = calc_vhs(points)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
crit = torch.nn.L1Loss()
opt.zero_grad()
out = model(images)
loss = 10*crit(out, points) + 0.1*crit(calc_vhs(out), vhs)
loss.backward()
opt.step()
print('smoke_train_ok', float(loss.detach()))
PY
```

- Inferencia sobre una imagen local (ruta configurable) usando las transformaciones de `utils.get_transform`:

```powershell
python - << 'PY'
import sys
from PIL import Image
import torch
from model import get_model, calc_vhs
from utils import get_transform
img_path = r"data\sample_real_images\original_img_1.png"
image_size = 224
transform = get_transform(image_size)
img = Image.open(img_path).convert('RGB')
img_t = transform(img).unsqueeze(0)
model = get_model('cpu').eval()
with torch.no_grad():
    out = model(img_t)
    vhs = calc_vhs(out)
print('inference_ok; VHS=', float(vhs.squeeze()))
PY
```

## 8) Instalación recomendada y estructura

### Instalación recomendada

Windows (PowerShell):

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS (bash/zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Comandos de entrenamiento e inferencia (propuestos)

- Entrenamiento inicial (runner propuesto, sin argparse existente): ver bloque de "smoke test" para forward/backward. Para entrenamiento real, deberá implementarse un script main que construya DataLoaders con `DogHeartDataset` y llame `train_initial_model`.

- Inferencia y guardado CSV (usando evaluate.inference_and_save con un DataLoader de `DogHeartTestDataset`): [dato no provisto en scripts actuales]; se requiere pequeño runner.

### Checklist de carpetas necesarias

- `models/` para pesos (`best_initial_model.pth`, `final_initial_model.pth`).
- `outputs/` para CSV/figuras.
- `data/YourDataset/{Images,Labels}` para entrenamiento; `data/TestSet/Images` para inferencia.

## Fixes sugeridos (diffs mínimos)

- `src/dataset.py` (import que falta):

```diff
+ from model import calc_vhs
```

- `src/iterative_train.py` (imports faltantes y posición de `pred_record`):

```diff
+ from torch.optim import AdamW
+ from torch.optim.lr_scheduler import CosineAnnealingLR
+ from model import get_model, calc_vhs
+ from evaluate import evaluate_model
@@
- def train_with_pseudo_labels(train_loader, unlabeled_loader, valid_loader, checkpoint, device, num_epochs=100, lr=1e-5):
+ def train_with_pseudo_labels(train_loader, unlabeled_loader, valid_loader, checkpoint, device, num_epochs=100, lr=1e-5):
+     pred_record = torch.zeros([len(train_loader.dataset), 10, 12])
```

- `src/model.py` (opción para entornos sin internet):

```diff
- model = models.efficientnet_b7(weights='DEFAULT')
+ model = models.efficientnet_b7(weights='DEFAULT')  # cambiar a None si no hay Internet
```

---

## Comandos finales sugeridos para tu terminal (ejecutables ahora)

```powershell
# 1) Activar venv (si no está activa) e importar librerías clave
. .\.venv\Scripts\Activate.ps1; python -c "import torch, torchvision, numpy as np; import PIL, pandas; print('OK', torch.__version__)"

# 2) Smoke test de forward/backward (1 mini-epoch sintético)
python - << 'PY'
import torch
from model import get_model, calc_vhs
model = get_model('cpu')
model.train()
images = torch.randn(2,3,224,224)
points = torch.rand(2,12)
vhs = calc_vhs(points)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
crit = torch.nn.L1Loss()
opt.zero_grad()
out = model(images)
loss = 10*crit(out, points) + 0.1*crit(calc_vhs(out), vhs)
loss.backward()
opt.step()
print('smoke_train_ok', float(loss.detach()))
PY

# 3) Inferencia sobre imagen de ejemplo
python - << 'PY'
from PIL import Image
import torch
from model import get_model, calc_vhs
from utils import get_transform
img_path = r"data\sample_real_images\original_img_1.png"
image_size = 224
transform = get_transform(image_size)
img = Image.open(img_path).convert('RGB')
img_t = transform(img).unsqueeze(0)
model = get_model('cpu').eval()
with torch.no_grad():
    out = model(img_t)
    vhs = calc_vhs(out)
print('inference_ok; VHS=', float(vhs.squeeze()))
PY
```
