# Proyecto final Carrito autónomo con DQN

Este proyecto implementa una pista 2D, un automóvil con dinámica simplificada y
un agente de aprendizaje por refuerzo profundo. La red recibe nueve variables y
produce un valor Q para cada una de nueve acciones discretas derivadas de los
controles acelerar, frenar, girar a la izquierda y girar a la derecha.

## Instalación en Windows

Desde PowerShell, en la raíz del proyecto, utiliza Python 3.11 o 3.12 y
`uv` en `PATH`. Si ya tienes un entorno en `.venv`, usa otro nombre para
crear uno limpio. [Instalación de uv](https://docs.astral.sh/uv/getting-started/installation/).

```powershell
$PythonBase = (Get-Command python.exe).Source
& $PythonBase --version
uv --no-cache venv --python $PythonBase .venv
$ProyectoPython = (Resolve-Path .\.venv\Scripts\python.exe).Path
uv --no-cache pip install --python $ProyectoPython --index-url https://download.pytorch.org/whl/cpu "torch==2.15.0"
uv --no-cache pip install --python $ProyectoPython -r requirements.txt
uv --no-cache pip check --python $ProyectoPython
& $ProyectoPython -X utf8 -B check_environment.py
```

- **Ejecución.** `requirements.txt` cubre la simulación, Pygame, los agentes, el entrenamiento y la evaluación.
- **Notebooks.** `requirements-notebooks.txt` añade JupyterLab, ipykernel y nbconvert al entorno del proyecto.
- **Pruebas automatizadas.** La suite usa `unittest` de Python y las dependencias de ejecución. Para automatizar notebooks se instala el complemento de Jupyter descrito abajo.

## Ejecutar los notebooks

Desde el mismo entorno virtual, instala el complemento y registra su kernel.
Si el registro de entrenamiento se encuentra fuera de `results/training_log.csv`,
indica su ubicación antes de iniciar Jupyter.

```powershell
uv --no-cache pip install --python $ProyectoPython -r requirements-notebooks.txt
& $ProyectoPython -m ipykernel install --sys-prefix --name carrito-dqn --display-name "Python (Carrito DQN)"
$env:CARRITO_TRAINING_LOG = Read-Host "Ruta al training_log.csv existente"
& $ProyectoPython -m jupyterlab
```

Abre el notebook desde JupyterLab, selecciona **Python (Carrito DQN)** y ejecuta
todas las celdas. Los dos notebooks resuelven sus rutas relativas desde la raíz
del repositorio cuando el kernel comienza ahí o en `notebooks/`.

También puedes ejecutar los notebooks sin interfaz y guardar las copias
ejecutadas y las exportaciones en un directorio temporal. Ejecuta estos comandos
desde la raíz, en la misma terminal y con `CARRITO_TRAINING_LOG` configurada.

```powershell
$SalidaNotebook = Join-Path ([System.IO.Path]::GetTempPath()) ("carrito-notebooks-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $SalidaNotebook | Out-Null
$env:CARRITO_NOTEBOOK_OUTPUT = Join-Path $SalidaNotebook "experimental"
& $ProyectoPython -m nbconvert --to notebook --execute notebooks/analisis_experimental.ipynb --output-dir $SalidaNotebook
$env:CARRITO_NOTEBOOK_OUTPUT = Join-Path $SalidaNotebook "entrenamiento"
& $ProyectoPython -m nbconvert --to notebook --execute notebooks/analisis_entrenamiento.ipynb --output-dir $SalidaNotebook
Remove-Item Env:CARRITO_NOTEBOOK_OUTPUT
$SalidaNotebook
```

Sin `CARRITO_NOTEBOOK_OUTPUT` se crea un directorio único por ejecución bajo
`results_notebooks/`. Si se indica un destino explícito, debe ser nuevo.
La última celda muestra la ubicación de las tablas CSV y figuras PNG.
`CARRITO_EVALUATION_CSV` permite elegir otro CSV de ocho columnas para un
análisis independiente.

- **Experimental.** `csv_results/evaluation_results.csv` contiene las 120 observaciones DQN del lote de 240. Las otras 120 corresponden a los agentes de referencia y se conservan en `csv_results/comparison_results.csv`. El archivo `csv_results/comparison_results_integrado.csv` reúne el lote. Los agregados del notebook comparan semillas y distinguen pasos de todos los episodios y pasos de los episodios exitosos.
- **Entrenamiento.** El `training_log.csv` real del Base64 conocido contiene 145 episodios de un entrenamiento. Usa su ruta existente. Las curvas y las tablas corresponden solo a ese registro.
- **Evaluaciones reproducidas.** Los CSV `results/evaluacion_best_*` pertenecen al checkpoint Base64 conocido. El lote experimental se verificó con los CSV publicados. Sus seis checkpoints originales siguen sin estar disponibles.

Consulta la [evidencia de ejecución de los notebooks](docs/evaluacion/evidencias/validacion_notebooks_2026-11-24.md)
para ver los entornos ensayados, las salidas y la comparación matemática.

## Secuencia de uso

Activa el entorno virtual o reemplaza `python` por `& $ProyectoPython` en los
ejemplos de PowerShell de esta sección y de la evaluación.

```powershell
python manual_drive.py
python random_baseline.py
python heuristic_baseline.py
python train.py --episodes 600
python evaluate.py --model results/best_model.pt
```

El entrenamiento se realiza sin ventana para aprovechar mejor el procesador.
La evaluación abre Pygame y muestra los sensores, la velocidad, la acción y el
avance. Los archivos `training_log.csv`, `training_curves.png` y los modelos
entrenados se guardan en `results`.

## Qué deben comparar los estudiantes

- Agente aleatorio frente a DQN.
- Cinco sensores frente a tres sensores.
- Red de 32, 64 o 128 neuronas por capa.
- Descuento gamma y duración de la exploración epsilon-greedy.
- Recompensa original frente a una modificación documentada.
- Resultados en entrenamiento frente a diez episodios de evaluación sin azar.

No debe calificarse una sola ejecución aislada. El aprendizaje por refuerzo es
estocástico y las comparaciones deben repetirse con al menos tres semillas.

## Evaluación reproducible (integrante 4)

`evaluate.py` carga un checkpoint existente y ejecuta la política sin exploración
epsilon-greedy: todas las acciones se solicitan con `training=False`. La red se
coloca en modo evaluación. No se entrena ni se modifica el modelo.

El comando original sigue siendo válido:

```powershell
python evaluate.py --model results/best_model.pt --episodes 10 --no-render
```

Por defecto se usa inicio fijo, cinco episodios y semilla inicial 1000. La
visualización se abre salvo que se indique `--no-render`. La selección de
dispositivo conserva el orden CUDA, MPS y CPU, según disponibilidad.

### Argumentos

| Argumento | Valor predeterminado | Descripción |
|---|---|---|
| `--model` | `results/best_model.pt` | Ruta de un archivo de modelo existente. |
| `--episodes` | `5` | Número de episodios solicitado; entero mayor que cero. |
| `--no-render` | Desactivado | Ejecuta sin ventana Pygame. |
| `--seed` | `1000` | Semilla del primer episodio; entero no negativo. |
| `--random-start` | Desactivado | Activa las pequeñas variaciones de posición y orientación que ya ofrece el simulador. |
| `--output` | Sin exportación | Ruta de un CSV nuevo; crea los directorios necesarios. |
| `--csv-output` | Sin exportación | CSV experimental nuevo de ocho columnas, compatible con el análisis del integrante 3. |
| `--manifest` | Al lado del primer CSV con sufijo `.manifest.jsonl` | Ruta nueva del manifiesto de procedencia. En una ejecución sin CSV puede solicitarse explícitamente. |
| `--configuracion` | Inferida de los pesos | Confirma la arquitectura con una etiqueta como `Base32` o `Base64`. |
| `--semilla-entrenamiento` | Inferida de `config.seed` | Confirma la semilla de entrenamiento del checkpoint, independiente de `--seed`. |

### Inicio fijo e inicio aleatorio

Inicio fijo con semilla explícita:

```powershell
python evaluate.py --model results/best_model.pt --episodes 10 --seed 42 --no-render
```

Inicio aleatorio reproducible y exportación:

```powershell
python evaluate.py --model results/best_model.pt --episodes 10 --seed 42 --random-start --no-render --output results/evaluacion_detallada.csv
```

Cada episodio usa `seed + índice`, comenzando con índice cero. En el ejemplo se
usan las semillas 42 a 51. El inicio fijo conserva la posición predeterminada;
en el simulador actual, variar la semilla sin activar `--random-start` no cambia
las condiciones iniciales. Las repeticiones con inicio fijo pueden ser idénticas.

### CSV y episodios interrumpidos

Se escribe una fila por episodio iniciado y se vacía el búfer del archivo tras
cada fila. Las columnas, en este orden, son:

```text
model,episode,seed,random_start,reward,steps,success,collision,terminated,truncated
```

`episode` comienza en 1; `seed` contiene la semilla efectiva de ese episodio.
`model` contiene solo el nombre del checkpoint y `reward` es la suma sin descuento de las
recompensas. `steps` cuenta las transiciones devueltas por el entorno. Los campos
`random_start`, `success`, `collision`, `terminated` y `truncated` usan 0 o 1.

Un episodio completo tiene `terminated=1` o `truncated=1`. Alcanzar el límite de
pasos es una truncación completa, no una interrupción. Si se cierra la ventana o
se pulsa Ctrl+C durante el episodio, se conserva una fila parcial con ambas
banderas en cero y solo las transiciones confirmadas. La transición interrumpida
antes de que `env.step()` devuelva sus datos no se registra.

Los episodios no iniciados no generan filas. Ctrl+C durante un reinicio que
todavía no devolvió la observación no confirma un episodio nuevo. En ese caso,
el CSV puede contener solo la cabecera. Una interrupción durante la preparación
del agente puede ocurrir antes de crear el CSV. El entorno se cierra mediante
`try/finally`, también ante errores de carga o de ejecución.

Los CSV existentes se rechazan: utiliza un nombre nuevo para cada ejecución.
La creación exclusiva protege también frente a otra ejecución que cree la misma
ruta simultáneamente. No se ofrece una opción para sobrescribir resultados.

### CSV experimental y compatibilidad con el integrante 3

`--output` conserva exactamente las diez columnas del formato detallado.
`--csv-output` produce este segundo formato, con las ocho columnas que valida el
notebook `notebooks/analisis_experimental.ipynb`.

```text
configuracion,semilla,condicion,episodio,reward,success,collision,steps
```

Ambas salidas pueden solicitarse juntas y proceden de una sola evaluación:

```powershell
python evaluate.py --model results/best_model.pt --episodes 10 --seed 1000 --random-start --no-render --configuracion Base64 --semilla-entrenamiento 42 --output results/detalle_Base64_s42_aleatoria_01.csv --csv-output results/experimental_Base64_s42_aleatoria_01.csv
```

En ese ejemplo, `semilla` en el CSV experimental vale **42** en todas las filas
porque identifica el entrenamiento; `seed` en el detallado recorre **1000–1009**
porque controla cada reinicio de evaluación. `condicion` vale `fija` o `aleatoria`,
según `--random-start`. El evaluador comprueba el ancho de las capas en los
tensores y lo contrasta con `config.hidden_size` si existe. Si se proporcionan
la etiqueta o la semilla de entrenamiento, deben coincidir con el checkpoint.
Si falta `config.seed`, se puede usar solo la salida detallada sin declarar
una semilla de entrenamiento. Los metadatos incompatibles detienen la ejecución.

El CSV experimental contiene **solo episodios completos**, incluidas las
truncaciones por límite de pasos. No puede identificar interrupciones con sus
ocho columnas; por ello las filas parciales quedan únicamente en el detallado
y se informa cuántas se omitieron del experimental. Las métricas de terminal
mantienen su cálculo sobre episodios completos, independientemente de las salidas.

Ambas rutas deben ser diferentes y nuevas. A diferencia del evaluador de la rama
experimental, esta integración **no añade filas a CSV existentes**. Usa un archivo
nuevo por modelo, semilla y condición, y coordina una consolidación en otro archivo
nuevo, conservando los originales y evitando duplicar episodios. El CSV de ocho
columnas no registra las semillas de evaluación: acompáñalo del detallado o de
la bitácora de ejecución para reproducir los reinicios.

Si falla la preparación de una salida, el evaluador cierra y retira los archivos
nuevos que alcanzó a crear antes de iniciar los episodios. Una vez preparada
la exportación, cada registro se sincroniza en el manifiesto antes de escribirse
en los CSV. Si falla una escritura, el evaluador conserva el manifiesto y las
salidas disponibles, e indica cuáles coinciden con el registro.

### Verificar y recuperar exportaciones

Conserva juntos el manifiesto y los CSV de cada ejecución. El primer registro
del JSONL incluye el nombre y SHA-256 del checkpoint, arquitectura, semilla de
entrenamiento, configuración disponible, semilla inicial y regla de evaluación,
commit de Git, huellas SHA-256 del código local, Python, PyTorch y dispositivo.
Los registros siguientes incluyen las semillas efectivas y los datos de cada
episodio. El nombre de la salida se guarda sin su ruta personal.

```powershell
python evaluate.py verify-exports results/detalle_Base64_s42_aleatoria_01.csv.manifest.jsonl
python evaluate.py recover-exports results/detalle_Base64_s42_aleatoria_01.csv.manifest.jsonl --output results/detalle_recuperado.csv --csv-output results/experimental_recuperado.csv
python evaluate.py verify-exports results/detalle_Base64_s42_aleatoria_01.csv.manifest.jsonl --output results/detalle_recuperado.csv --csv-output results/experimental_recuperado.csv
```

Si los CSV están en otra carpeta, pasa sus rutas con `--output` y `--csv-output`
al verificar. La verificación compara columnas, orden y valores con el JSONL.
Un manifiesto sin cierre se distingue de uno completado, incluso cuando sus
CSV estén alineados con los episodios que alcanzaron a registrarse. La recuperación
omite solo una línea final inconclusa, exige destinos nuevos y no reanuda la
simulación. Dos CSV independientes no constituyen una escritura atómica.

La columna `steps` conserva los pasos de cada episodio completo. El notebook
publica el promedio sobre todos los episodios y el promedio entre éxitos.
La terminal del evaluador informa este último. Un grupo sin éxitos conserva
vacío su promedio de pasos exitosos.

### Métricas

El resumen de terminal calcula exclusivamente sobre episodios completos:

- **Retorno promedio:** media de las sumas de recompensa por episodio.
- **Desviación estándar del retorno:** poblacional (`ddof=0`); vale cero con un solo episodio completo.
- **Tasa de éxito:** proporción de episodios con `success=1`, según el entorno.
- **Tasa de colisión:** proporción de episodios con `collision=1`, según el entorno.
- **Pasos promedio en episodios exitosos:** media de pasos solo entre los éxitos completos.
- **Episodios completados e interrumpidos:** conteos separados; también se informa cuántos no llegaron a iniciarse.

Sin episodios completos se muestra `no disponible` para las métricas de desempeño.
Sin éxitos, el promedio de pasos exitosos también se muestra como `no disponible`.
Las filas parciales se exportan, pero no alteran estas medias ni tasas. Para
analizar el CSV con la misma convención, filtra `terminated=1` o `truncated=1`.

### Reproducir y comparar modelos

Conserva el mismo checkpoint, versión del código, versiones de las dependencias,
dispositivo, modalidad de inicio y lista de semillas. El manifiesto conserva el
nombre, SHA-256, arquitectura y semilla encontrada en el checkpoint. Registra
también por separado la procedencia del archivo original.
El CSV detallado conserva solo el nombre, así que dos archivos distintos pueden
compartir el identificador. Diferentes dispositivos o versiones de PyTorch pueden producir diferencias
numéricas: una semilla no garantiza igualdad entre plataformas.

La semilla de evaluación controla las condiciones iniciales, no es la semilla
con la que se entrenó el modelo. Para comparar checkpoints de distintas semillas
de entrenamiento usa las mismas semillas de evaluación y nombres de salida únicos:

```powershell
python evaluate.py --model results/seed_42/best_model.pt --episodes 10 --seed 1000 --random-start --no-render --output results/eval_train42_eval1000.csv
python evaluate.py --model results/seed_43/best_model.pt --episodes 10 --seed 1000 --random-start --no-render --output results/eval_train43_eval1000.csv
```

Esas rutas son ejemplos y requieren checkpoints existentes. Registra por separado
las configuraciones y semillas de entrenamiento y repite la comparación con las
semillas acordadas por el equipo. El evaluador usa `TrackConfig()` y `CarConfig()`
predeterminados; no reconstruye automáticamente condiciones del entorno usadas
durante el entrenamiento. Compara resultados obtenidos bajo condiciones equivalentes.

### Carga restringida y adaptable

`DQNAgent.load(path, mode="inference")` lee una vez con `weights_only=True`,
deduce el ancho de los tensores y comprueba que coincida con la configuración
guardada cuando existe. Construye la política con ese ancho y deja la red en
modo de evaluación. El evaluador usa `training=False` y escribe el manifiesto
con el SHA-256 y los metadatos devueltos por esa misma lectura. La inferencia
no restaura la red objetivo, el optimizador ni los pasos. Un archivo histórico
completo sigue deserializándose como un solo checkpoint aunque solo se aplique
su política.

`DQNAgent.load(path)` conserva el comportamiento de carga para reanudar
entrenamiento. También se puede indicar con `mode="resume"`. Restaura la
política, la red objetivo, el optimizador, los pasos y la configuración
numérica guardada. Si faltan los pasos en un archivo antiguo, empieza en cero.
El formato histórico no guarda las transiciones del búfer ni el estado global
de los generadores aleatorios. El búfer se reinicia y la trayectoria exacta del
entrenamiento no se puede reanudar. Los argumentos de `train.py` se conservan.

`requirements.txt` fija `torch==2.15.0`. En Windows se instala la distribución
CPU del índice oficial de PyTorch. El filtro en ejecución conserva el rechazo
de versiones preliminares, versiones anteriores a 2.10 y la serie 2.14.x.
Para ampliar la especificación, verifica la carga restringida y la suite con
las versiones y dispositivos que se quieran admitir. Si ambas variables
`TORCH_FORCE_*` se activan, la carga se detiene antes de deserializar.

Para comparar checkpoints existentes, usa la misma semilla de evaluación
y rutas de salida nuevas:

```powershell
python evaluate.py --model results/Base32_seed42/best_model.pt --episodes 10 --seed 1000 --random-start --no-render --configuracion Base32 --semilla-entrenamiento 42 --output results/detalle_Base32_s42_validacion_01.csv --csv-output results/experimental_Base32_s42_validacion_01.csv
python evaluate.py --model results/Base64_seed42/best_model.pt --episodes 10 --seed 1000 --random-start --no-render --configuracion Base64 --semilla-entrenamiento 42 --output results/detalle_Base64_s42_validacion_01.csv --csv-output results/experimental_Base64_s42_validacion_01.csv
```

Las rutas de entrada son ejemplos: deben apuntar a modelos existentes de la
arquitectura y semilla indicadas. Omitir `--random-start` permite comprobar el
inicio fijo. Cambia los nombres de salida para esa modalidad. La adaptación
Base32 se probó con un checkpoint sintético. El Base64 conocido se contrastó
con sus CSV históricos. Los seis checkpoints del lote experimental no están
disponibles para reproducirlo.

### Pruebas del módulo

```powershell
python -X utf8 -B -m unittest discover -s tests -q
```

La suite combina pruebas simuladas del evaluador con checkpoints temporales
y tensores reales de PyTorch. Cubre la carga restringida, la adaptación Base32,
los estados de AdamW y la gestión de fallos. La verificación con el Base64
conocido usa el archivo original en modo de lectura y salidas temporales.
Consulta la [evidencia de carga centralizada](docs/evaluacion/evidencias/carga_centralizada_2026-11-24.txt)
para ver las versiones ejecutadas y las comparaciones reales.

La evidencia de ejecución y las comprobaciones pendientes se encuentran en
[docs/evaluacion/validacion.md](docs/evaluacion/validacion.md). El borrador del
reporte individual está en
[docs/evaluacion/reporte_integrante4.md](docs/evaluacion/reporte_integrante4.md).

