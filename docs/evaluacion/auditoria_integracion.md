# Auditoría de integración y evaluación DQN

Fecha: 24 de noviembre de 2026. Rama de trabajo:
`feature/evaluacion-reproducible`, base `064acca` sobre `e8a2d92`.
Las referencias de GitHub se verificaron con `git ls-remote`.

Esta auditoría describe el estado anterior a la corrección de exportaciones.
El integrante 4 asumió el cierre de los cuatro módulos. Las decisiones
vigentes de carga, recuperación y responsabilidades están en la
[preparación de la integración final](preparacion_integracion.md).

El evaluador de esta rama valida la identidad del checkpoint y generó ambos
CSV con el modelo base disponible. Los 240 registros publicados pasan la
auditoría matemática. La aceptación integral permanece bloqueada por la
integración aprobada, los checkpoints experimentales y las rutas de los
notebooks. La [matriz de trazabilidad](trazabilidad_checkpoints.md) registra
los modelos conocidos y los datos faltantes.

## Ramas y conflictos previstos

| Rama | Inventario respecto a main | Dependencias, compatibilidad y conflictos |
|---|---|---|
| `main` · `db1c671` | Entorno, simulador, agente, entrenamiento, evaluador original, referencias, requisitos, README y notebook de entrenamiento en raíz | El agente inicializa 64 neuronas y carga pesos sin adaptar el ancho. |
| `feature/agente-dqn` · `abe9d49` | Cambia `dqn_agent.py`, `train.py` y `.gitignore` | Adapta el ancho guardado y proporciona opciones de entrenamiento. Cambios disjuntos de los archivos editados por el integrante 4. |
| `feature/experimentos` · `1be444f` | Incluye la rama del agente. Cambia `evaluate.py`, mueve el notebook de entrenamiento a `notebooks/` y añade el experimental, dos scripts de referencia y cinco CSV | Conflicto esperado en `evaluate.py`: ambas ramas editaron la inicialización de argumentos, ejecución y exportación. El README y el notebook de entrenamiento de nuestra rama no se tocaron en las dos ramas ajenas y en nuestra rama, respectivamente. Verificar la detección del traslado y conservar sus rutas. |
| `feature/evaluacion-reproducible` · `064acca` | Cambia README y evaluador. Añade pruebas, documentación, evidencias y cuatro CSV propios | Incluye ambos esquemas. Depende del agente adaptable para la carga real de Base32. Los CSV históricos ya están rastreados pese a la nueva regla `results*/` que se integrará. |

Las tres ramas de trabajo tienen cambios pendientes de incorporación a `main`.
Esta matriz se basa en los árboles y diffs, sin ensayar una fusión. Los únicos
archivos compartidos entre los dos integrantes ajenos son los ya heredados del
agente. El simulador, el entorno, la configuración y los requisitos coinciden
en las referencias inspeccionadas.

## Hallazgos

| ID | Severidad | Archivo o área | Responsable | Solución | Estado |
|---|---|---|---|---|---|
| A01 | Alta | `evaluate.py`: etiquetas no verificadas | Integrante 4 | Inferir ancho de los tensores y semilla de `config.seed`, contrastar declaraciones y rechazar metadatos contradictorios | APROBADO |
| A02 | Alta | `evaluate.py` y `DQNAgent.load()`: carga sin restricción en el agente original | Integrantes 4 y 2 | Leer e invocar el agente con restricción de PyTorch y los mismos bytes. Coordinar que el agente adopte una API segura explícita | APROBADO CON OBSERVACIONES |
| A03 | Media | `evaluate.py`: primera salida huérfana si falla la segunda | Integrante 4 | Abrir ambos destinos exclusivamente antes de escribir cabeceras y retirar las salidas propias de una preparación fallida. Mantener filas confirmadas ante interrupción | APROBADO |
| A04 | Media | CSV detallado: ruta personal | Integrante 4 | Identificador por nombre y SHA-256 de procedencia en terminal | APROBADO |
| A05 | Alta | Seis checkpoints del experimento y discrepancia Base64/42 | Integrantes 3 y 2 | Entregar originales y vincular los hashes al entrenamiento y los grupos del CSV | BLOQUEADO |
| A06 | Alta | Notebooks: rutas de CSV y log | Integrante 3 | Parametrizar raíz y entradas. Publicar o entregar los logs correctos | PENDIENTE |
| A07 | Media | `tick_labels`, Matplotlib y Jupyter | Integrante 3 y equipo | Resolver mínimo de Matplotlib o adaptar código y declarar dependencias para el análisis | PENDIENTE |
| A08 | Media | Diferencia semántica entre notebook e informe al promediar pasos | Integrante 3 | Separar pasos totales y exitosos. Acordar rutas nuevas para las tablas derivadas | APROBADO CON OBSERVACIONES |
| A09 | Alta | Ramas sin incorporar y prueba Base32/Base64 | Equipo e integrante 4 | Revisar integración y pasar la matriz de aceptación con los modelos correctos | BLOQUEADO |
| A10 | Media | Requisitos de entrega y demostración | Equipo | Confirmar consigna oficial y reunir video, informe y presentación | PENDIENTE |

## Decisiones del evaluador

- **Identidad.** El formato propio del agente guarda una red de tres matrices
  lineales. Se comprueban las dos capas ocultas y su enlace con la salida.
  `config.hidden_size`, si está presente, debe concordar. La semilla ausente
  permite una evaluación detallada sin atribución de entrenamiento. Una semilla
  inválida, una contradicción o un CSV experimental sin semilla verificable
  fallan antes de abrir el entorno y las salidas. Las etiquetas explícitas
  aceptan `BaseN`, sin distinguir mayúsculas. Los comandos Base64/42 siguen
  funcionando con checkpoints cuyos metadatos coinciden. El hash identifica el
  archivo, no certifica la veracidad histórica de su semilla.
  La futura carga adaptable necesita `config.hidden_size` para cambiar el
  ancho. Verificar los tensores en esta etapa no suple ese metadato en la API
  del agente.
- **Carga.** Confiar solo en el nombre o en metadatos declarados facilita errores
  de atribución. Cargar directamente los pesos en otra red duplicaría
  responsabilidades del agente. Se preserva su cargador, se inspecciona usando
  `weights_only=True` y se le entregan los mismos bytes bajo
  `TORCH_FORCE_WEIGHTS_ONLY_LOAD=1`. La variable se restaura al salir y es
  compartida por el proceso durante la llamada. Esta solución está pensada para
  el comando síncrono. Revisar una API explícita antes de usarla como servicio
  concurrente. El filtro de versiones admite las estables 2.10–2.13 y 2.15+,
  según los avisos de [PyTorch hasta 2.9.1](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p)
  y [PyTorch 2.14](https://github.com/pytorch/pytorch/security/advisories/GHSA-v32g-mr63-3j83).
  Consultar también la [semántica de la variable](https://docs.pytorch.org/docs/stable/notes/serialization).
  La restricción no sustituye comprobar procedencia ni los avisos de seguridad
  de la versión instalada. `requirements.txt` se conserva y su actualización
  requiere coordinación con el equipo.
- **Exportación.** El esquema detallado conserva sus diez columnas. El
  experimental conserva ocho y filtra parciales. Los archivos existentes se
  rechazan y las aperturas son exclusivas. Si la preparación falla, se cierra
  todo y se retiran solo las rutas propias que conservan la identidad del
  archivo creado por esa preparación. Si el sistema impide la limpieza, la
  excepción lo indica. Tras iniciarse los episodios, los archivos se conservan.
  El vaciado del búfer por fila limita la pérdida ante interrupciones normales.
  Dos archivos independientes no ofrecen escritura atómica ante fallos de
  disco, terminación forzada o interrupción durante la escritura.
- **Procedencia.** Se imprime nombre, SHA-256, arquitectura y semilla. El CSV
  detallado usa el nombre, de modo que debe acompañarse de esa línea y de una
  referencia de procedencia separada. Los cuatro CSV previos se conservaron.

## Datos publicados y alcance de la evidencia

Se contrastaron `evaluation_results.csv` (120 DQN), `comparison_results.csv`
(60 aleatorios y 60 heurísticos) y el integrado (240). La clave de agente,
configuración, condición, semilla y episodio es única. Hay 24 grupos de diez
episodios, numerados del 1 al 10, y tres semillas por configuración y condición.
Los 24 resúmenes por semilla y ocho agregados de media y desviación muestral
coinciden con los cálculos a tolerancia absoluta `1e-10`. Se observaron 33
diferencias de representación en `reward`, máximo `2.842170943040401e-14`.
No se editaron los archivos para suprimirlas.

| Grupo | Éxito | Colisión | Retorno medio fijo / aleatorio | Pasos exitosos fijo / aleatorio |
|---|---|---|---|---|
| Base32 | 100 % | 0 % | 160.172 / 160.128 | 299.33 / 299.33 |
| Base64 | 100 % | 0 % | 160.180 / 160.258 | 301.33 / 301.83 |
| Aleatorio | 0 % | 70.00 % / 73.33 % | 5.464 / 4.240 | No aplica |
| Heurístico | 100 % | 0 % | 164.629 / 164.588 | 547.00 / 546.77 |

El informe adjunto coincide a su precisión publicada. El notebook experimental
agrupa DQN por semilla y aplica desviación muestral entre semillas. Su columna
de pasos promedia todos los episodios, que coinciden con los exitosos para
esas 120 filas. Los scripts de referencia sí usan `--seed` como semilla de
evaluación, mientras el evaluador DQN publicado usa 1000 más el índice y
etiqueta la semilla de entrenamiento. Igualar etiquetas en el análisis no
garantiza las mismas perturbaciones entre agentes.

**Consistencia matemática: APROBADO.** El CSV y el informe son evidencia de
resultados entregados, sin bitácoras que acrediten la ejecución individual de
los 240 desde esta auditoría. **Reproducción: BLOQUEADA** hasta recibir los
archivos y metadatos indicados en la matriz.

## Propuestas para los notebooks

La celda de lectura experimental busca `evaluation_results.csv` en el
directorio actual. El archivo rastreado es `csv_results/evaluation_results.csv`.
El otro notebook busca `results/training_log.csv`. No hay un log de entrenamiento
CSV publicado en la rama y `results*/` está ignorado. Propuesta para ambos:

```python
from pathlib import Path

cwd = Path.cwd().resolve()
project = next((p for p in (cwd, *cwd.parents)
                if (p / "config.py").is_file() and (p / "requirements.txt").is_file()), None)
if project is None:
    raise FileNotFoundError("Abre el notebook desde el proyecto o elige su raíz.")
evaluation_csv = project / "csv_results" / "evaluation_results.csv"
training_csv = project / "results" / "training_log.csv"  # Parámetro de entrada.
```

Comprobar la ruta de entrenamiento antes de leerla y permitir seleccionar
el log por modelo y semilla. La raíz explícita es necesaria en kernels
iniciados fuera del repositorio. Probar desde raíz y `notebooks/`, con ambos
archivos presentes y con el log ausente. El log externo disponible sirve
para comprobar gráficas, no acredita los seis entrenamientos experimentales.

`tick_labels` apareció en [Matplotlib 3.9](https://matplotlib.org/3.9.0/api/_as_gen/matplotlib.axes.Axes.boxplot.html)
y el requisito actual admite 3.8. Para mantener el mínimo, el integrante 3
puede usar `ax.boxplot(values, showmeans=True)` seguido de
`ax.set_xticks(range(1, len(labels) + 1), labels)`. Probar en 3.8 y en
la versión de referencia. Elevar el mínimo también resuelve la llamada, pero
reduce la compatibilidad y requiere aprobación. Declarar aparte un kernel
Python 3.11 con `ipykernel`, un frontend como JupyterLab y `nbconvert` o un
ejecutor equivalente para comprobación automatizada. `requirements.txt` cubre
las bibliotecas de las celdas, pero no esas herramientas.

Para reproducir la métrica del informe, introducir una columna temporal
`pasos_exitosos = df["steps"].where(df["success"].eq(1))` y agrupar su media.
Valores ausentes en grupos sin éxitos deben mantenerse como tales. Para
exportaciones controladas, hacer explícito un directorio nuevo y usar rutas
con `mode="x"` en `to_csv`. Así se evita escribir sobre tablas del CWD, con
el efecto esperado de pedir nombres nuevos en cada repetición. Acordar pruebas
con un grupo con fracasos, entradas solo lectura, dos contextos de ejecución
y una repetición de exportación contra un destino existente.

## Aceptación del sistema y estado académico

| Comprobación | Dictamen y evidencia |
|---|---|
| Entorno Python 3.11, importación y `check_environment.py` | APROBADO en el entorno instalado. Instalación desde cero en copia integrada: PENDIENTE. |
| Suite con Python 3.11 y 3.12 | APROBADO. Incluye mocks de agente y entorno, además de una prueba con simulador y agente simulado. |
| Checkpoint Base64 conocido, ambos inicios, doble CSV, métrica y evaluación sin exploración | APROBADO para el modelo base. Las suites verifican explícitamente `training=False`. |
| Base32 y Base64 experimentales | BLOQUEADO por ausencia de originales y carga adaptable pendiente en la rama actual. |
| Agentes de referencia como parte del árbol integrado | BLOQUEADO por integración pendiente. Se revisaron interfaces y datos publicados. |
| Notebooks completos desde clon e instalación limpia | BLOQUEADO por las rutas, el log ausente y la integración. En Python 3.11 faltaban `ipykernel`, `jupyterlab` y `nbconvert`. Con Matplotlib 3.11.2 y backend Agg se ejecutaron en memoria las 11 celdas experimentales y 5 de entrenamiento, ajustando solo rutas y presentación. Se generaron 5 y 1 figuras, respectivamente, y las tablas experimentales se exportaron en temporal. |
| Demostración visual | APROBADO CON OBSERVACIONES. Los dos episodios de ventana local del modelo base constan en evidencia del 23 de noviembre. La prueba gráfica de esta etapa usa SDL dummy. Demostración del sistema integrado y video final pendientes. |
| Reproducción de los CSV de experimentos | BLOQUEADO por la trazabilidad. |
| Repositorio, README y pruebas | El módulo se documentó y verificó. El repositorio final y la instalación integrada siguen pendientes. |
| Experimentos, gráficas e informe | Hay resultados y gráficas de referencia y un informe de experimentación. Informe final conjunto y presentación final pendientes de confirmar. |

La guía del integrante 4 pide repositorio funcional, documentación, evidencias,
video, informe y presentación. El README original enumera comparaciones de
cinco contra tres sensores, redes de 32/64/128, gamma, epsilon-greedy y recompensa,
y sugiere repetir comparaciones con al menos tres semillas. La guía describe
como referencia las dos arquitecturas y los 240 episodios. No se encontró la
consigna oficial del docente entre los archivos revisados. Hasta obtenerla,
los otros experimentos figuran como comparaciones sugeridas en el README y
su obligatoriedad académica queda **PENDIENTE** de confirmación.

La [evidencia de esta auditoría](evidencias/auditoria_2026-11-24.txt) registra
los comandos, versiones y verificaciones observadas. El contraste previo
con el informe y la estructura de las ramas se detalla en
[la integración experimental](integracion_experimentos.md), que describe
el comportamiento anterior al cotejo de etiquetas de esta auditoría.

## Propuesta y responsables al momento de la auditoría

1. **Integrante 2:** confirmar agente, configuraciones, versión admitida de
   PyTorch y los artefactos de entrenamiento. Revisar la carga restringida
   directamente en su API y validar Base32/Base64 cuando lleguen los archivos.
   Revisar primero su rama mediante el procedimiento autorizado del equipo.
2. **Integrante 3:** entregar los seis modelos con su bitácora, corregir rutas
   y exportaciones, conciliar compatibilidad de Matplotlib y precisar qué
   promedio de pasos usa cada tabla. Su rama ya hereda al agente. Una vez
   aprobada la integración del agente, revisar el delta propio del integrante 3
   sobre la base resultante para detectar duplicaciones.
3. **Integrante 1:** verificar que la versión de pista, observaciones y acciones
   corresponda al entorno publicado. Revisar terminación, truncaciones,
   recompensas y arranque aleatorio con prueba de entorno y visualización.
4. **Integrante 4:** conservar `e8a2d92` y `064acca`. Tras autorización, incorporar
   la base aprobada a su rama y resolver `evaluate.py` conservando ambas salidas,
   identidad, semillas, métricas y cierre. Confirmar que los cuatro CSV ya
   rastreados se conservan y que las nuevas evidencias no arrastran modelos,
   rutas personales ni escrituras a los históricos.
5. Ejecutar la matriz de aceptación con Python 3.11 en una instalación limpia
   y usar los comandos Base32/Base64 del README con checkpoints identificados,
   salidas nuevas y ambas condiciones. Abrir los notebooks desde un clon,
   ejecutar las referencias y guardar resultados y demos en las ubicaciones
   acordadas. Reproducir el lote publicado solo cuando esté justificado y
   autorizado.
6. Preparar el borrador de PR con alcance, pruebas, hashes y pendientes para
   revisión. Coordinar la demostración grabada, informe y defensa. Toda fusión,
   publicación y edición de módulos ajenos requiere autorización expresa.
