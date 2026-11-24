# Integración del evaluador con los experimentos

Fecha: 24 de noviembre de 2026.

## Base y alcance

Rama activa: `feature/evaluacion-reproducible`. El commit `e8a2d92` se conserva
como HEAD y base de estos cambios sin commit. El árbol estaba limpio al comenzar.
Se ejecutó `git fetch origin` para consultar las ramas; no se hicieron merges,
rebase, commits, push ni Pull Requests, y no se cambió `main`.

Referencias revisadas:

- Evaluador propio: `e8a2d92`.
- `origin/feature/experimentos`: `1be444f`.
- `origin/feature/agente-dqn`: `abe9d49`.
- Guía de trabajo del integrante 4, páginas 1–3, e informe de experimentación
  suministrados por el usuario. Se leyeron sin modificarlos.

La guía pide verificar y organizar la evidencia ya obtenida, sin repetir los
experimentos. Sus etapas de PR, integración final, instalación desde cero, video
y presentación no autorizan esas operaciones en esta tarea: prevalecen las
restricciones expresas del usuario. No se repitieron los 240 episodios.

## Comparación y solución

| Aspecto | Evaluador propio | Rama de experimentos | Solución en esta rama |
|---|---|---|---|
| Argumentos | Modelo, episodios, render, semilla de evaluación, inicio y salida detallada | Añade salida experimental, configuración y semilla de entrenamiento | Se conservan ambos conjuntos. |
| CSV | Diez columnas y creación exclusiva | Ocho columnas y apertura en modo append | Dos salidas independientes, ambas exclusivamente a archivos nuevos. |
| Reproducibilidad | Semilla inicial configurable más índice de episodio | Semillas de evaluación desde 1000 | Se conserva `--seed`, con 1000 por defecto. |
| Episodios parciales | Marcados por las banderas de finalización; excluidos de métricas | No identifica interrupciones en el CSV | Solo el detallado conserva parciales; el experimental incluye completos. |
| Cierre | `try/finally` y contextos de archivo | Cierre fuera de un bloque protector | Se mantienen el cierre seguro y la liberación de ambos archivos. |

`--output` conserva exactamente:

```text
model,episode,seed,random_start,reward,steps,success,collision,terminated,truncated
```

`--csv-output` genera exactamente:

```text
configuracion,semilla,condicion,episodio,reward,success,collision,steps
```

Los valores `reward`, `success`, `collision` y `steps` se trasladan sin recalcular
al formato experimental; `episode` pasa a llamarse `episodio`. Ambas salidas se
alimentan de la misma ejecución con `training=False`. No se entrena ni se
modifica el checkpoint. La red continúa en modo evaluación después de cargarla.

`--semilla-entrenamiento` alimenta únicamente `semilla` en el experimental y
vale 42 por defecto para mantener la interfaz del integrante 3. `--seed` controla
los reinicios y se registra por episodio en el detallado. `--configuracion`
vale `Base64` por defecto; es una etiqueta, no una orden para construir una red.
Ambos metadatos deben corresponder al checkpoint seleccionado por el operador.

Se rechazan salidas existentes, incluso vacías, y dos rutas que resuelvan al mismo
archivo. No hay append ni sobreescritura de históricos. Si falla la apertura de
la segunda salida, se cierran la primera y el entorno; una salida nueva puede
quedar solo con cabecera. No se borran archivos para ocultar ese fallo.

Para nuevos lotes, producir archivos distintos por configuración, semilla y
condición. La consolidación debe escribir otro archivo nuevo, verificar el
esquema y evitar claves duplicadas; no se implementó un consolidador automático
ni se modificó `csv_results/evaluation_results.csv` publicado.

## Dependencia de la carga de modelos

La rama experimental ya contiene los commits del integrante 2. Su `dqn_agent.py`
es idéntico al de `origin/feature/agente-dqn`: `load()` lee `hidden_size` del
checkpoint y reconstruye redes y optimizador cuando cambia el ancho. Esa mejora
no está integrada en nuestra rama ni en el `main` observado.

Nuestro agente sigue inicializándose con 64 neuronas. El evaluador delega toda la
carga a `DQNAgent.load()` y no copia su implementación, no reconstruye redes ni
adapta pesos. Etiquetar un modelo como Base32 no habilita su carga en esta base.

La validación completa Base32/Base64 requiere la integración aprobada de esos
cambios y los checkpoints correctos. Los comandos posteriores se encuentran en
el README. Se deben confirmar arquitectura y semilla de entrenamiento, usar las
mismas semillas de evaluación y exportar a rutas nuevas.

## Revisión de notebooks y coordinación con el integrante 3

Los índices de celda de esta sección comienzan en cero. No se editaron notebooks.

1. **Experimental, celda 2:** usa `Path("evaluation_results.csv")`, pero el archivo
   publicado está en `csv_results/evaluation_results.csv`. Desde la raíz falta el
   prefijo; desde `notebooks/` también falta ascender a la raíz. El mensaje de
   error que pide colocarlo en la carpeta del proyecto debe ajustarse junto con
   la ruta. Propuesta para coordinar con su autor:

   ```python
   PROJECT_ROOT = Path.cwd().resolve()
   if not (PROJECT_ROOT / "config.py").is_file():
       PROJECT_ROOT = PROJECT_ROOT.parent
   CSV_PATH = PROJECT_ROOT / "csv_results" / "evaluation_results.csv"
   ```

   Este ejemplo admite ejecutar desde la raíz o desde `notebooks/`; para un
   archivo nuevo hay que seleccionar explícitamente su ruta. No mover ni
   sobrescribir el CSV histórico para sortear el problema.

2. **Entrenamiento, celda 2:** busca `results/training_log.csv` respecto al CWD.
   Desde `notebooks/` buscaría `notebooks/results/training_log.csv`. La rama
   publicada no incluye un `training_log.csv`; las carpetas de resultados están
   ignoradas. Solicitar el log ya existente y usar una raíz explícita y una ruta
   parametrizable por ejecución. No ejecutar un entrenamiento nuevo para suplirlo.
   Si se trabajó con un directorio de salida propio, debe seleccionarse su log.

3. **Experimental, celda 12:** `boxplot(..., tick_labels=...)` necesita Matplotlib
   3.9 o posterior, mientras `requirements.txt` permite 3.8. Está confirmado en
   la [documentación oficial de Matplotlib](https://matplotlib.org/3.9.0/api/_as_gen/matplotlib.axes.Axes.boxplot.html).
   Coordinar una de dos soluciones: elevar el mínimo con aprobación del equipo,
   o hacer compatible el notebook usando `boxplot(values, showmeans=True)` y
   configurar las etiquetas mediante `set_xticks`/`set_xticklabels`. No se cambió
   el notebook ni `requirements.txt`.

4. **Experimental, celdas 5 y 16:** el promedio de pasos de la celda 5 usa todos
   los episodios. El informe compara pasos solo entre éxitos. Coinciden en el
   lote DQN publicado porque los 120 episodios son exitosos; si aparecen fracasos,
   el autor debe distinguir ambas métricas. La celda 16 exporta nombres fijos en
   el CWD, aunque el texto dice que guarda junto al notebook; acordar un destino
   explícito y archivos nuevos para evitar sobreescrituras.

El CSV experimental no incluye las semillas de evaluación ni banderas de
interrupción por contrato. Conservar el detallado y el comando de ejecución como
evidencia. No introducir filas parciales en un notebook que las interpretaría
como episodios completos.

## Contraste de datos publicados e informe

Se leyeron los blobs remotos, sin incorporarlos al árbol de trabajo:

- `evaluation_results.csv`: 120 episodios DQN, dos configuraciones, semillas
  42, 123 y 999, dos condiciones y diez episodios por combinación.
- `comparison_results.csv`: 120 episodios de referencias, 60 aleatorios y 60
  heurísticos.
- `comparison_results_integrado.csv`: 240 registros, 24 grupos de diez; las
  claves incluyen agente, configuración, condición, semilla y episodio.
- Los resúmenes de 24 filas por semilla y ocho filas generales se recalcularon
  a partir de esos registros y coinciden con tolerancia absoluta `1e-10`.

Se mantuvieron Base32 y Base64 separadas al agrupar. Las fuentes y el archivo
integrado presentan 33 diferencias numéricas de representación, con diferencia
máxima `2.842170943040401e-14`; no faltan claves ni hay episodios duplicados.
Una comparación inicial de igualdad exacta detectó esas diferencias; se
inspeccionaron y luego se contrastaron con la tolerancia indicada. No se corrigió
ni reescribió ningún CSV.

Las tablas del informe coinciden al redondeo mostrado: Base32 promedia 299.33
pasos exitosos, Base64 301.33–301.83 y el heurístico 546.77–547.00. DQN y heurístico
tienen 100 % de éxito y 0 % de colisión; el aleatorio tiene 0 % de éxito y
70–73.33 % de colisiones. La desviación del informe es muestral entre tres
semillas; la desviación de la terminal es poblacional entre episodios de una
sola evaluación. No deben confundirse.

Los scripts `random_baseline_comparison.py` y `heuristic_baseline_comparison.py`
están en la rama remota, no en esta base. Se revisaron sus interfaces, selección
de acciones, semillas y exportación de nueve columnas. No se ejecutaron aquí
ni se copiaron; su validación completa después de una integración autorizada
sigue pendiente. Sus dependencias ya figuran en `requirements.txt`.

## Pruebas ejecutadas

| Prueba | Resultado | Alcance |
|---|---|---|
| Suite en Python 3.12.14 | 48 aprobadas, 6.069 s | 35 originales y 13 nuevas; agentes y carga simulados, con una prueba del núcleo real y agente simulado. |
| Suite en Python 3.11.9 | 48 aprobadas, 7.520 s | Sin fallos, errores ni omisiones. |
| Checkpoint real de 64 neuronas existente | Dos episodios aleatorios; ambas salidas correctas | No es el lote Base64 del integrante 3 ni valida Base32. |
| Regresión real | Las dos filas detalladas coinciden exactamente con las dos primeras filas de la referencia histórica | Recompensas 161.73355527000086 y 161.78168365349316; 326 pasos en ambos episodios. |
| Celdas 3 y 5 del notebook remoto | Validación y agrupación aprobadas con el CSV experimental real | Pandas real; solo se sustituyó la función de visualización, no datos ni cálculos. No se ejecutaron celdas de escritura. |
| Datos publicados | 240 registros y resúmenes contrastados | Lectura y recálculo, sin repetir experimentos. |

Los CSV de la prueba breve fueron temporales. El checkpoint mantuvo su SHA-256.
No se cambiaron modelos ni CSV históricos, y no se generaron entrenamientos.
La prueba real utilizó semillas de evaluación 42 y 43, y registró la semilla de
entrenamiento 42 en ambas filas experimentales. Las pruebas simuladas verifican
además semillas de evaluación 900–902 frente a una semilla de entrenamiento 7.

El registro de resultados observados está en
[evidencias/integracion_experimentos_2026-11-24.txt](evidencias/integracion_experimentos_2026-11-24.txt).

## Pendientes del equipo

Integrar mediante el procedimiento aprobado el agente adaptable, los notebooks,
scripts y CSV del integrante 3; proporcionar los checkpoints y logs por semilla;
coordinar los ajustes de rutas y compatibilidad de Matplotlib; validar Base32 y
Base64 después de esa integración; comprobar una instalación limpia. Video,
informe final, presentación y publicación quedan para sus etapas autorizadas.
