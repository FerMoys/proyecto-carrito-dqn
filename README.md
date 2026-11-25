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

## Inicio rápido

Desde la raíz y en la misma terminal de PowerShell, puedes abrir la conducción
manual, ejecutar los agentes de referencia o probar dos pasos de entrenamiento:

```powershell
& $ProyectoPython -X utf8 -B manual_drive.py
& $ProyectoPython -X utf8 -B random_baseline.py
& $ProyectoPython -X utf8 -B heuristic_baseline.py
& $ProyectoPython -X utf8 -B train.py --episodes 1 --max-steps 2 --hidden-size 32
```

El último comando es un diagnóstico. Cada entrenamiento sin `--output` crea
un directorio propio `results/entrenamiento-.../` y muestra su ruta al iniciar.
Si indicas `--output`, el directorio final debe ser nuevo, incluso si el destino
existente está vacío. El entrenador conserva cualquier corrida parcial y exige
otra ruta para volver a iniciar. Los checkpoints, el log y la gráfica de cada
corrida se guardan en su directorio. Para evaluar una política ya entrenada,
puedes usar el paquete de demostración directamente.

## Demostración Base64 con el paquete de entrega

Obtén `result_model_prueba.zip` de los materiales entregados por el equipo. El ZIP
se distribuye por separado y la carpeta local `results_demo_base64/` está
excluida de Git. El ZIP entregado contiene `best_model.pt` y `training_log.csv`
en su raíz. Extrae solo una vez, desde la raíz del proyecto, en PowerShell:

```powershell
$Paquete = Read-Host "Ruta al archivo result_model_prueba.zip recibido"
if (-not $Paquete -or -not (Test-Path -LiteralPath $Paquete -PathType Leaf)) {
    throw "Falta el ZIP de entrega. Solicita result_model_prueba.zip al equipo."
}
$DemoBase64 = Join-Path (Get-Location).Path "results_demo_base64"
New-Item -ItemType Directory -Path $DemoBase64 -ErrorAction Stop | Out-Null
Expand-Archive -LiteralPath $Paquete -DestinationPath $DemoBase64 -ErrorAction Stop
```

Si ya recibiste los archivos sueltos, coloca los originales `best_model.pt`
y `training_log.csv` en `results_demo_base64/`. Si ya tienes el paquete extraído,
continúa con este bloque. Vuelve a ejecutarlo si abres otra terminal:

```powershell
$DemoBase64 = Join-Path (Get-Location).Path "results_demo_base64"
$ModeloBase64 = Join-Path $DemoBase64 "best_model.pt"
$LogBase64 = Join-Path $DemoBase64 "training_log.csv"
$HuellasDemo = @{
    $ModeloBase64 = "8fca3c72f147cd94af182417e55e4093ed0f6ff8eaf3448e654650d51c04f5b1"
    $LogBase64 = "2dd65a595cc5378e6fbad9007501ce5731656849f8cf92cb90303962ce187ab8"
}
foreach ($Archivo in $HuellasDemo.Keys) {
    if (-not (Test-Path -LiteralPath $Archivo -PathType Leaf)) {
        throw "Falta $(Split-Path -Leaf $Archivo). Extrae el ZIP de entrega o coloca el archivo original en results_demo_base64/."
    }
    $Actual = (Get-FileHash -LiteralPath $Archivo -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Actual -ne $HuellasDemo[$Archivo]) {
        throw "SHA-256 incorrecto para $(Split-Path -Leaf $Archivo)."
    }
}
```

Ejecuta diez episodios con inicio fijo y diez con inicio aleatorio. Cada modo
genera un CSV detallado, un CSV experimental y un manifiesto JSONL. Las
exportaciones quedan en una carpeta nueva bajo `results_demo_base64/evaluaciones/`:

```powershell
if (-not $ModeloBase64 -or -not (Test-Path -LiteralPath $ModeloBase64 -PathType Leaf)) {
    throw "Primero prepara y verifica el paquete Base64."
}
$CorridaDemo = Join-Path $DemoBase64 ("evaluaciones/" + [guid]::NewGuid().ToString("N"))
$OpcionesDemo = @("--model", $ModeloBase64, "--episodes", "10", "--seed", "42",
                  "--no-render", "--configuracion", "Base64", "--semilla-entrenamiento", "42")
foreach ($Modo in @("fija", "aleatoria")) {
    $Inicio = @()
    if ($Modo -eq "aleatoria") { $Inicio = @("--random-start") }
    $SalidasDemo = @("--output", (Join-Path $CorridaDemo "$Modo-detalle.csv"),
                     "--csv-output", (Join-Path $CorridaDemo "$Modo-experimental.csv"),
                     "--manifest", (Join-Path $CorridaDemo "$Modo.jsonl"))
    & $ProyectoPython -X utf8 -B evaluate.py @OpcionesDemo @Inicio @SalidasDemo
    if ($LASTEXITCODE -ne 0) { throw "Falló la evaluación $Modo." }
}
$CorridaDemo
```

Para la demostración visual en un escritorio con Pygame, usa el mismo checkpoint
verificado y omite `--no-render`. La ventana muestra sensores, acción, velocidad
y avance. Puedes cerrarla o terminar con Ctrl+C.

```powershell
if (-not $ModeloBase64 -or -not (Test-Path -LiteralPath $ModeloBase64 -PathType Leaf)) {
    throw "Primero prepara y verifica el paquete Base64."
}
& $ProyectoPython -X utf8 -B evaluate.py --model $ModeloBase64 --episodes 2 --seed 42 --configuracion Base64 --semilla-entrenamiento 42
```

Las evaluaciones del Base64 conocido se comparan con los CSV `results/evaluacion_best_*`.
El lote de 240 episodios del informe procede de los CSV experimentales publicados.
Los seis checkpoints entrenados de ese lote siguen sin estar disponibles.

## Ejecutar los notebooks

Desde el mismo entorno virtual, instala el complemento y registra su kernel.

```powershell
uv --no-cache pip install --python $ProyectoPython -r requirements-notebooks.txt
& $ProyectoPython -m ipykernel install --sys-prefix --name carrito-dqn --display-name "Python (Carrito DQN)"
```

El notebook experimental lee los 120 episodios DQN de `csv_results/evaluation_results.csv`
sin archivos externos. El integrado de 240 episodios está en
`csv_results/comparison_results_integrado.csv`. La variable `CARRITO_EVALUATION_CSV`
permite elegir otro CSV con las ocho columnas de evaluación.

Para analizar el entrenamiento Base64 de 145 episodios, verifica el paquete como
se indica arriba y configura su log antes de iniciar Jupyter. También puedes
usar otro registro existente, por ejemplo el de tu entrenamiento diagnóstico.

```powershell
if (-not $LogBase64 -or -not (Test-Path -LiteralPath $LogBase64 -PathType Leaf)) {
    throw "Falta el registro Base64. Primero prepara y verifica el paquete de demostración."
}
$env:CARRITO_TRAINING_LOG = $LogBase64
```

Abre JupyterLab, selecciona **Python (Carrito DQN)** y ejecuta todas las celdas.
Las rutas relativas se resuelven desde la raíz al ejecutar allí o en `notebooks/`.

```powershell
& $ProyectoPython -m jupyterlab
```

Para ejecutar sin interfaz desde la raíz, guarda las copias ejecutadas y sus
exportaciones en una carpeta temporal. El análisis de entrenamiento se ejecuta
si configuraste un registro existente.

```powershell
$SalidaNotebook = Join-Path ([System.IO.Path]::GetTempPath()) ("carrito-notebooks-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $SalidaNotebook -ErrorAction Stop | Out-Null
try {
    $env:CARRITO_NOTEBOOK_OUTPUT = Join-Path $SalidaNotebook "experimental"
    & $ProyectoPython -m nbconvert --to notebook --execute notebooks/analisis_experimental.ipynb --output-dir $SalidaNotebook
    if ($LASTEXITCODE -ne 0) { throw "Falló el notebook experimental." }
    if ($env:CARRITO_TRAINING_LOG -and (Test-Path -LiteralPath $env:CARRITO_TRAINING_LOG -PathType Leaf)) {
        $env:CARRITO_NOTEBOOK_OUTPUT = Join-Path $SalidaNotebook "entrenamiento"
        & $ProyectoPython -m nbconvert --to notebook --execute notebooks/analisis_entrenamiento.ipynb --output-dir $SalidaNotebook
        if ($LASTEXITCODE -ne 0) { throw "Falló el notebook de entrenamiento." }
    } else {
        Write-Host "Para ejecutar también el notebook de entrenamiento, configura CARRITO_TRAINING_LOG con un CSV existente."
    }
} finally {
    Remove-Item Env:CARRITO_NOTEBOOK_OUTPUT -ErrorAction SilentlyContinue
}
$SalidaNotebook
```

Sin `CARRITO_NOTEBOOK_OUTPUT` cada notebook crea su propio directorio en
`results_notebooks/`. Los destinos explícitos deben ser nuevos. Las copias
ejecutadas pueden mostrar rutas locales. Para la entrega usa las figuras y
tablas exportadas o limpia las salidas de las celdas antes de distribuirlas.

Consulta la [evidencia de los notebooks](docs/evaluacion/evidencias/validacion_notebooks_2026-11-24.md)
y el [inventario de entrega](docs/evaluacion/cierre_entrega.md).

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

Los comandos listos para ejecutar están en [Demostración Base64](#demostración-base64-con-el-paquete-de-entrega).

Por defecto se usa inicio fijo, cinco episodios y semilla inicial 1000. La
visualización se abre salvo que se indique `--no-render`. La selección de
dispositivo conserva el orden CUDA, MPS y CPU, según disponibilidad.

### Argumentos

| Argumento | Valor predeterminado | Descripción |
|---|---|---|
| `--model` | `results/best_model.pt` | Ruta de un archivo de modelo existente. En un clon nuevo, indica `--model` con la ruta de la demostración. |
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

La demostración usa la misma semilla inicial en ambos modos. Activa el
inicio aleatorio con `--random-start`.

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

La demostración genera las dos salidas con la misma evaluación. En ella,
`semilla` del CSV experimental vale **42** en todas las filas porque identifica
el entrenamiento. `seed` del CSV detallado recorre **42–51** porque controla los
reinicios. `condicion` vale `fija` o `aleatoria`, según `--random-start`.
El evaluador contrasta la arquitectura y la semilla indicada con el checkpoint.
Si falta `config.seed`, usa solo la salida detallada sin declarar una semilla
de entrenamiento.

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

Para verificar y recuperar la salida aleatoria de la demostración, vuelve a
usar el directorio que imprimió la evaluación. La recuperación crea otros CSV.

```powershell
if (-not $CorridaDemo) { throw "Primero ejecuta la evaluación Base64 o indica su carpeta en CorridaDemo." }
$Manifiesto = Join-Path $CorridaDemo "aleatoria.jsonl"
if (-not (Test-Path -LiteralPath $Manifiesto -PathType Leaf)) { throw "No se encontró aleatoria.jsonl en CorridaDemo." }
& $ProyectoPython -X utf8 -B evaluate.py verify-exports $Manifiesto
$Recuperacion = Join-Path $CorridaDemo ("recuperado-" + [guid]::NewGuid().ToString("N"))
$DetalleRecuperado = Join-Path $Recuperacion "detalle.csv"
$ExperimentalRecuperado = Join-Path $Recuperacion "experimental.csv"
& $ProyectoPython -X utf8 -B evaluate.py recover-exports $Manifiesto --output $DetalleRecuperado --csv-output $ExperimentalRecuperado
if ($LASTEXITCODE -ne 0) { throw "Falló la recuperación." }
& $ProyectoPython -X utf8 -B evaluate.py verify-exports $Manifiesto --output $DetalleRecuperado --csv-output $ExperimentalRecuperado
if ($LASTEXITCODE -ne 0) { throw "Las exportaciones recuperadas no coinciden con el manifiesto." }
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

La semilla de evaluación controla las condiciones iniciales. Para comparar
checkpoints de distintas semillas de entrenamiento, conserva las semillas de
evaluación, indica con `--model` cada archivo existente y usa rutas de salida
nuevas. El evaluador usa `TrackConfig()` y `CarConfig()` predeterminados.
Compara resultados obtenidos bajo condiciones equivalentes.

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

La adaptación Base32 se probó con un checkpoint sintético. El Base64 conocido
se contrastó con sus CSV históricos. Los seis checkpoints del lote experimental
siguen sin estar disponibles.

### Pruebas del módulo

```powershell
& $ProyectoPython -X utf8 -B -m unittest discover -s tests -q
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

## Entregables académicos

- [Bitácora del proyecto](docs/bitacora_proyecto.md). Configuración, semillas, incidencias y decisiones verificadas.
- [Evidencia cuantitativa](docs/evidencia_cuantitativa.md). Tablas, figuras, huellas de los insumos y pasos de regeneración.
