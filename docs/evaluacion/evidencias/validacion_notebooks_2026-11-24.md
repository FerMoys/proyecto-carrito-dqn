# Validación de notebooks — 2026-11-24

Rama `feature/integracion-final`, HEAD base `a1ed530`. Las comprobaciones
se ejecutaron en Windows con entornos virtuales limpios creados por uv 0.12.18
y con la distribución CPU de PyTorch del índice oficial.

| Python | PyTorch | NumPy | pandas | Matplotlib | Suite |
| --- | --- | --- | --- | --- | --- |
| 3.11.9 | 2.15.0+cpu | 2.4.6 | 2.3.3 | 3.11.2 | 64/64 (89.151 s) |
| 3.12.14 | 2.15.0+cpu | 1.26.4 | 2.3.3 | 3.8.4 | 64/64 (83.936 s) |

Ambos entornos usaron Gymnasium 1.4.0, Pygame 2.6.1, TensorBoard 2.21.0,
JupyterLab 4.6.6, ipykernel 7.4.0, nbconvert 7.17.2 y nbclient 0.11.0.
`uv pip check` y `python -X utf8 -B check_environment.py` pasaron
en ambos. El diagnóstico usó el controlador de vídeo `dummy` para
validar el render `rgb_array`. La matriz acredita las combinaciones
de la tabla.

## Ejecuciones

- Se ejecutaron ambos notebooks desde la raíz y desde `notebooks/` en
  los dos entornos. Una aserción en el kernel confirmó el directorio
  de trabajo de cada ejecución. Dentro de cada entorno, los CSV y PNG
  generados desde ambos directorios fueron idénticos byte por byte.
- En Python 3.11 también se ejecutaron los comandos `nbconvert` del README,
  con copias ejecutadas y exportaciones en directorios temporales.
- En Python 3.11, una entrada ausente en cada notebook produjo
  `FileNotFoundError` con la variable que permite configurarla, antes de
  crear un directorio de salida. Al reutilizar un directorio, ambos
  notebooks produjeron `FileExistsError` y sus archivos conservaron sus hashes.
- En una copia temporal de los datos se verificó la búsqueda predeterminada
  de ambos CSV, la creación de `results_notebooks/` y la asignación de un
  directorio nuevo por ejecución.
- Las advertencias de transporte local de Jupyter en Windows no impidieron
  ninguna ejecución. La instalación alternativa con pip excedió la ventana
  de espera. La ruta documentada con uv se completó en ambos entornos.

## Contraste matemático y procedencia

El CSV experimental predeterminado tiene 120 filas DQN, con dos arquitecturas,
tres semillas, dos condiciones y diez episodios por grupo. El CSV de agentes
de referencia aporta 120 filas. Las 240 claves del CSV integrado coinciden
con las de ambas fuentes. La mayor diferencia numérica entre archivos
existentes fue de `2.84e-14`. Las comparaciones usaron una tolerancia absoluta
y relativa de `1e-12`.

Se recalcularon las medias por semilla y las medias y desviaciones muestrales
entre semillas con `csv`, `math.fsum` y `statistics.stdev`. Se contrastaron
los 12 grupos DQN y los cuatro agregados con los resúmenes históricos.

| Arquitectura | Inicio | Retorno medio | Pasos medios de todos los episodios | Pasos medios en episodios exitosos |
| --- | --- | ---: | ---: | ---: |
| Base32 | Fijo | 160.171743 | 299.333333 | 299.333333 |
| Base32 | Aleatorio | 160.128077 | 299.333333 | 299.333333 |
| Base64 | Fijo | 160.180167 | 301.333333 | 301.333333 |
| Base64 | Aleatorio | 160.258058 | 301.833333 | 301.833333 |

Los 120 episodios DQN del CSV histórico registraron éxito y cero colisiones.
Al ejecutar el mismo notebook con el CSV integrado, los seis grupos del
agente aleatorio tuvieron pasos totales y dejaron vacío el promedio de
pasos exitosos al no registrar éxitos.

El registro externo de entrenamiento del Base64 conocido tiene 145
episodios. SHA-256 `2dd65a595cc5378e6fbad9007501ce5731656849f8cf92cb90303962ce187ab8`.
Se recalcularon los primeros y últimos 50 episodios y las sumas de las
nueve acciones. El retorno medio pasó de 32.803515 a 105.266466 y la tasa de
éxito del 2 % al 54 % entre esas dos ventanas.

Se conservaron los SHA-256 de los nueve CSV versionados, el registro externo
y los dos checkpoints originales. El entrenamiento Base64 conocido y sus
evaluaciones se documentan en la [evidencia de carga centralizada](carga_centralizada_2026-11-24.txt).
Los 240 episodios se trataron como datos históricos. Los seis checkpoints
originales del lote experimental siguen sin estar disponibles.

## Archivos que exporta cada ejecución

| Notebook | Figuras PNG | Tablas CSV |
| --- | --- | --- |
| Experimental | Tasa de éxito, retorno promedio, distribución de retornos por semilla, pasos totales y pasos en episodios exitosos | `resumen_por_semilla.csv`, `resumen_estadistico.csv` |
| Entrenamiento | Curvas de retorno y éxito móvil, distribución de acciones | `contraste_entrenamiento.csv`, `uso_acciones.csv` |
