# rtabmap-mcp

Servidor **MCP** (Model Context Protocol) que deja a un asistente como Claude **medir una habitación o un mueble a partir de un escaneo LiDAR** del iPhone o el iPad hecho con la app [RTAB-Map](https://introlab.github.io/rtabmap/). Lee la base de datos `.db` del escaneo y la convierte en una nube de puntos con el suelo en z = 0 y las paredes alineadas con los ejes. Después la analiza:
- la **planta** vista desde arriba;
- una **tabla de alturas** de las superficies;
- **histogramas** por eje para encontrar paredes y caras de muebles.

- Va en un único archivo, [`server.py`](server.py). Se ejecuta con [`uv`](https://docs.astral.sh/uv/), que instala las dependencias solo (`mcp`, `numpy`, `pillow`).
- No hace falta instalar RTAB-Map en el ordenador: lee el `.db` directamente y en solo lectura.
- Usa las poses que RTAB-Map optimiza al cerrar los bucles del recorrido, como su propia exportación.
- **Todo se queda en tu ordenador**: no se conecta a ningún sitio. Si no le das otra ruta, guarda los archivos en una carpeta temporal privada.
- Licencia **MIT**.

## Herramientas

| Herramienta | Qué hace |
|---|---|
| `rtabmap_info` | Resume el escaneo: versión, fotogramas, resolución, intrínsecos, rango de profundidad, fotogramas del grafo optimizado, cierres de bucle, confianza del LiDAR y recorrido de la cámara. |
| `rtabmap_get_frame` | Guarda la foto de un fotograma tal cual (JPEG sin recomprimir) y una vista previa de su profundidad. |
| `rtabmap_build_cloud` | Une los fotogramas en una nube de puntos (`.npz` y `.ply`) voxelizada, con el suelo en z = 0 y las paredes a lo largo de x e y. |
| `rtabmap_plan_image` | Dibuja la nube vista desde arriba (PNG): un color por franja de altura y las coordenadas escritas cada 0,5 m. |
| `rtabmap_height_grid` | Da una tabla de alturas por celdas: la del asiento, la del respaldo, dónde termina un mueble. |
| `rtabmap_axis_histogram` | Cuenta los puntos de una caja a lo largo de un eje: cada pared o cara es un pico en su coordenada. |
| `rtabmap_transform_cloud` | Recorta un trozo de la nube y lo gira, lo refleja y lo mueve a las coordenadas de otra escena (por ejemplo, un `.blend`). |

Todas devuelven texto (JSON compacto o una tabla) con las rutas exactas de lo que guardan. Cada parámetro lleva una descripción con sus unidades: el asistente la ve al elegir la herramienta.

## Cómo se usa (en 3 pasos)

Un **MCP** es un conjunto de herramientas que se conecta a un asistente (Claude Code, Claude Desktop u otro cliente compatible). Una vez instalado **no lo ejecutas tú**: le pides al asistente lo que quieres, en lenguaje normal, y él decide qué herramienta llamar y con qué parámetros.

**1. Instala `uv`** (una sola vez). Es lo que arranca el servidor y le instala sus dependencias:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh        # macOS / Linux
# Windows (PowerShell): powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**2. Descarga el servidor y conéctalo a tu asistente:**
```bash
git clone https://github.com/Eliaxs1900/rtabmap-mcp.git
claude mcp add --scope user rtabmap -- uv run --script "$PWD/rtabmap-mcp/server.py"
```
Con `claude mcp list` deberías ver `rtabmap: … ✔ Connected`. Para Claude Desktop y otros clientes, mira [Instalación](#instalación).

**3. Copia el `.db` del escaneo al ordenador y pídeselo al asistente**, indicando la ruta:

> «Con `~/Escaneos/salon.db`, dime cuánto mide la habitación de pared a pared y a qué altura está el asiento del sofá.»

El asistente:
1. llama a `rtabmap_info` para ver qué trae el escaneo;
2. construye la nube con `rtabmap_build_cloud`;
3. mira la planta con `rtabmap_plan_image` para situarse;
4. mide con `rtabmap_axis_histogram` (paredes y caras) y `rtabmap_height_grid` (alturas).

Otras peticiones que entiende:

| Le dices | Lo que hace |
|---|---|
| «Enséñame la foto del fotograma 40» | `rtabmap_get_frame` con `frame_id: 40` |
| «Haz la nube hasta 5 m y con vóxeles de 2 cm» | `rtabmap_build_cloud` con `max_depth: 5`, `voxel_size: 0.02` |
| «Quita los puntos poco fiables del LiDAR» | `rtabmap_build_cloud` con `min_confidence: "medium"` |
| «Planta de toda la habitación por franjas de altura» | `rtabmap_plan_image` con `bands: [[0.05, 0.6], [0.6, 1.3], [1.3, 2.05]]` |
| «¿A qué altura está el techo?» | `rtabmap_axis_histogram` con `axis: "z"` en una caja que llegue al techo |
| «Saca el sofá y ponlo mirando al otro lado en mi escena» | `rtabmap_transform_cloud` con `crop_min`, `crop_max`, `rotate_z_deg: 180` y `translate` |

> **Sin asistente:** también puedes llamar a las funciones desde Python. Mira [Probar sin cliente MCP](#probar-sin-cliente-mcp).

## Instalación

Requisitos: macOS, Linux o Windows con [`uv`](https://docs.astral.sh/uv/getting-started/installation/). La primera ejecución descarga `mcp`, `numpy` y `pillow` en un entorno aislado.

```bash
git clone https://github.com/Eliaxs1900/rtabmap-mcp.git
```

**Claude Code**
```bash
claude mcp add --scope user rtabmap -- uv run --script /RUTA/A/rtabmap-mcp/server.py
```

**Claude Desktop y otros clientes MCP** (en Claude Desktop, el archivo es `claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "rtabmap": { "command": "uv", "args": ["run", "--script", "/RUTA/A/rtabmap-mcp/server.py"] }
  }
}
```

## Coordenadas y unidades

- Todo va en **metros**.
- Una nube nivelada (`level_floor: true`, por defecto) tiene **z hacia arriba y el suelo en z = 0**. Con `align_walls: true` (por defecto) las paredes principales quedan a lo largo de x e y.
- El origen x/y es el **centro del suelo escaneado**. Cambia un poco con los parámetros, así que las coordenadas solo se comparan dentro de una misma nube. Las distancias y las alturas sí se pueden comparar entre nubes.
- En la planta, +x va hacia la derecha y +y hacia arriba.
- `poses: "optimized"` (por defecto) usa las poses del grafo de RTAB-Map, con los cierres de bucle, y salta los fotogramas que RTAB-Map dejó fuera del mapa. `poses: "odometry"` usa la odometría cruda de cada fotograma, como las versiones anteriores al 13 de noviembre de 2026, y reproduce sus nubes exactamente.

## Dónde guarda los archivos

Las herramientas que escriben archivos aceptan una ruta. Si no se la das, usan una carpeta temporal privada por escaneo: `$TMPDIR/rtabmap_mcp/<escaneo>/`, que en macOS solo ve tu usuario.
- La nube se llama según las opciones que cambies, por ejemplo `cloud_vox-0.02_max-5.npz`.
- Las plantas y los recortes se guardan junto a su nube.
- Los fotogramas llevan delante el nombre del escaneo, así que dos escaneos no se pisan.

macOS vacía esa carpeta con el tiempo: para conservar algo, pide una ruta propia.

## Privacidad

Un escaneo de tu casa es un dato personal. Este servidor solo lee el `.db`, en solo lectura, y escribe en tu disco; no se conecta a ningún sitio. El [`.gitignore`](.gitignore) deja fuera `.db`, `.npz`, `.ply`, `.png` y `.jpg` para que nunca acaben en un repositorio por error. Por eso este README no trae imágenes de ejemplo.

## Probar sin cliente MCP

`server.py` también se puede importar como un módulo normal:
```bash
uv run --with "mcp>=1.10,<2" --with numpy --with pillow python -c "
import importlib.util as u; s=u.spec_from_file_location('s','server.py'); m=u.module_from_spec(s); s.loader.exec_module(m)
print(m.rtabmap_info('mi_escaneo.db'))
print(m.rtabmap_build_cloud('mi_escaneo.db', voxel_size=0.02))"
```

## Formato del `.db` (lo aprendido)

- **Profundidad:** cada fotograma la guarda en PNG. Si es float32 (metros), OpenCV la guarda como un PNG de 8 bits y 4 canales con los bytes del float en orden BGRA: hay que intercambiar R y B antes de leerla como float32. Un PNG de 16 bits son milímetros.
- **Calibración** (cámara mono, RTAB-Map 0.18 o posterior): una cabecera de int32 con la versión, el tipo, el ancho, el alto y los tamaños de K, D, R, P y `local_transform`. Detrás van K, D, R y P en float64 y `local_transform` en float32. La profundidad tiene menos resolución que la foto, así que K se escala por ancho de la profundidad / ancho de la foto.
- **`Node.pose`:** una matriz 3 × 4 float32 por filas (mundo ← base) con la odometría cruda.
- **`Admin.opt_ids` / `opt_poses`:** las poses del grafo optimizado, con los cierres de bucle, tal como quedaron al cerrar el mapa. Van comprimidas con `compressData2()` de RTAB-Map: un flujo zlib seguido de tres int32 (filas, columnas y tipo de OpenCV). Los nodos con peso −9 quedaron fuera del grafo, por ejemplo porque la cámara no se movió.
- **`Data.depth_confidence`:** la confianza de ARKit por píxel de profundidad, guardada como 0, 50 o 100 (baja, media o alta).

## Rendimiento

En un Mac con Apple Silicon, un escaneo de 130 a 160 fotogramas (LiDAR de 256 × 192) se convierte en nube en 2 a 4 s con vóxeles de 1 cm. El voxelizado se queda con el primer punto de cada vóxel. Lo hace por trozos para que la memoria no crezca, y el resultado es el mismo que haciéndolo de una vez.

## Límites

- Solo RGB-D con una cámara, como el LiDAR del iPhone o el iPad. No lee cámaras estéreo ni varias cámaras.
- Necesita una calibración de RTAB-Map 0.18 o posterior. No lee la profundidad comprimida en RVL.
- Probado con escaneos de la app de RTAB-Map para iOS (versión 0.24).

## Licencia

[MIT](LICENSE).
