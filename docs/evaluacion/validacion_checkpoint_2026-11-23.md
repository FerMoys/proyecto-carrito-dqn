# Evaluación real del checkpoint del integrante 2

Fecha: 23 de noviembre de 2026 (America/Guatemala).

Rama: `feature/evaluacion-reproducible`. Esta etapa añade resultados y
documentación; conserva el código del evaluador y los módulos del equipo.

Nota de publicación (24 de noviembre de 2026): se anonimizaron las rutas del
usuario, del intérprete y del checkpoint en documentación, evidencias y CSV.
En los CSV cambió únicamente `model` a `best_model.pt`; las semillas, recompensas,
pasos, banderas y métricas originales se conservaron. Los hashes de los CSV se
recalcularon después de esa sustitución. Los hashes de los checkpoints no cambian.

## Procedencia y compatibilidad

El usuario proporcionó `best_model.pt`, `final_model.pt`, `training_log.csv`,
`evaluation_summary.csv` y `training_curves.png` desde:

```text
<directorio_resultados_integrante2>
```

Se leyeron los archivos originales; no se copiaron checkpoints al repositorio ni
se modificó su contenido. Los dos checkpoints contienen redes con dimensiones
9 → 64 → 64 → 9, semilla de entrenamiento 42 y `total_steps=45552`. Son compatibles
con `DQNAgent.load()` de nuestra base, sin integrar ni copiar cambios de otra rama.

Se eligió **`best_model.pt`** para todas las evaluaciones. `final_model.pt` se
inspeccionó para comprobar metadatos y dimensiones; no se evaluó su desempeño.
No se afirma que ambos modelos sean idénticos por compartir arquitectura y pasos.

SHA-256 de los originales, comprobados antes y después:

| Archivo | SHA-256 |
|---|---|
| `best_model.pt` | `8fca3c72f147cd94af182417e55e4093ed0f6ff8eaf3448e654650d51c04f5b1` |
| `final_model.pt` | `a6aae0e6395290c273f387026509381cc3632d7091ba45c9b3d5994108b3f8b2` |

## Resultados aportados por el integrante 2

Su `evaluation_summary.csv` contiene retorno medio **161.68243519516508**, tasa
de éxito **1.0** y tasa de colisión **0.0**. Ese CSV agregado no contiene registros
por episodio ni semillas para reconstruir por sí solo la ejecución original.

El registro de entrenamiento contiene **145 episodios**. Los últimos 30 tienen
27 éxitos y 3 colisiones: éxito móvil del **90 %** y retorno medio de
**152.44845565474043**. La curva muestra esa evolución; esos son resultados del
entrenamiento del integrante 2 y no se atribuyen al trabajo del integrante 4.

## Ejecuciones nuevas del integrante 4

Se utilizó el Python 3.11 de Microsoft Store validado en la etapa anterior,
PyTorch 2.15.0+cpu y el entorno predeterminado del repositorio. El agente se cargó
realmente desde el checkpoint, sin mocks y sin entrenamiento. El evaluador
conservó `training=False` y colocó la red en modo evaluación.

| Ejecución | Completos | Retorno medio | Desviación poblacional | Éxito | Colisión | Pasos medios en éxitos | Interrumpidos |
|---|---|---|---|---|---|---|---|
| Inicio fijo, semillas 42–51 | 10 | 161.6824351951651 | 0 | 100 % | 0 % | 326.0 | 0 |
| Inicio aleatorio, semillas 42–51 | 10 | 161.8193127579036 | 0.08040261942877586 | 100 % | 0 % | 326.1 | 0 |
| Repetición aleatoria, semillas 42–51 | 10 | 161.8193127579036 | 0.08040261942877586 | 100 % | 0 % | 326.1 | 0 |
| Pygame, inicio fijo, semillas 42–43 | 2 | 161.6824351951651 | 0 | 100 % | 0 % | 326.0 | 0 |

Las dos primeras ejecuciones se hicieron invocando el script como subproceso.
La repetición y la demostración se hicieron llamando a `evaluate.main()` con sus
argumentos CLI en el intérprete real, sin parches de entorno ni agente. En la
demostración se omitió `--no-render`, se ejecutó Pygame y se completaron ambos
episodios; después se cerró el entorno. No se probó un cierre manual a mitad del
episodio en esta etapa.

En total se completaron **32 episodios de evaluación real**, incluidos diez de
repetición y dos visuales. No son 32 escenarios independientes: las semillas se
repiten deliberadamente para comprobar reproducibilidad y renderizado.

## Archivos nuevos y validaciones

Se crearon estos CSV mediante apertura exclusiva, sin sobrescribir resultados:

- [Inicio fijo](../../results/evaluacion_best_fija_py311_20261123_01.csv).
- [Inicio aleatorio](../../results/evaluacion_best_aleatoria_py311_20261123_01.csv).
- [Repetición aleatoria](../../results/evaluacion_best_aleatoria_py311_20261123_02.csv).
- [Demostración visual](../../results/evaluacion_best_visual_py311_20261123_01.csv).

En los cuatro archivos se verificaron las columnas exactas:

```text
model,episode,seed,random_start,reward,steps,success,collision,terminated,truncated
```

También se verificaron cantidad de filas, numeración de episodios, semillas
consecutivas, modalidad de inicio y finalización de todos los episodios. Los tres
primeros tienen diez filas, episodios 1–10 y semillas 42–51. El visual tiene dos
filas, episodios 1–2 y semillas 42–43. Todos terminaron con `terminated=1`,
`success=1`, `collision=0` y `truncated=0`.

Los dos CSV aleatorios anonimizados son **idénticos byte por byte**, con SHA-256
común en los archivos locales (CRLF):

```text
5f3ae68a99a01f347890781f4410921b90bf147283705e165da2c881410c02d9
```

Git normaliza los saltos de línea a LF al preparar estos archivos. La limpieza
conservó los saltos de línea originales del árbol de trabajo y no cambió la
configuración de Git. Para verificar tanto las copias locales como los bytes
preparados para publicación, estos son los SHA-256 recalculados:

| CSV | Archivo local (CRLF) | Publicación / staging (LF) |
|---|---|---|
| `evaluacion_best_fija_py311_20261123_01.csv` | `ee56c24ffb8a5d855bf08ab59b106cb515a40e26ad82f23470a44e9a7f0a9beb` | `7b2660c998cd17f5c1c690ae079a8a0eef81e89188e87f00dd2a83a753c52d18` |
| `evaluacion_best_aleatoria_py311_20261123_01.csv` | `5f3ae68a99a01f347890781f4410921b90bf147283705e165da2c881410c02d9` | `4565d8945270cd22606d1fa78bbed23af69784b216aeee56c3c620f89cf71f40` |
| `evaluacion_best_aleatoria_py311_20261123_02.csv` | `5f3ae68a99a01f347890781f4410921b90bf147283705e165da2c881410c02d9` | `4565d8945270cd22606d1fa78bbed23af69784b216aeee56c3c620f89cf71f40` |
| `evaluacion_best_visual_py311_20261123_01.csv` | `15b2a86919cb8380c39481106d6bf5866d648d0fe9df3cf76c397f1c616fc9a8` | `a9cc0a963b1241b63d2ceb4685832843e3a31400af22e4337b6127777e159d72` |

Los campos `sha256` de la evidencia de consola se actualizaron con los hashes
locales CRLF de esta tabla. Las dos repeticiones también son idénticas en staging.

Las métricas se recalcularon leyendo los CSV con la biblioteca estándar; no se
tomaron únicamente de los mensajes de terminal. El retorno fijo coincide con el
resumen del integrante 2 dentro de tolerancia absoluta `1e-10`: la diferencia
observada es `2.842170943040401e-14`, compatible con redondeo numérico.

La pequeña diferencia de retorno entre modalidades no demuestra superioridad de
una modalidad ni generalización a todas las condiciones. Los inicios fijos son
idénticos en este simulador, por eso su desviación es cero. Las diez semillas
aleatorias solo cubren las pequeñas variaciones iniciales implementadas.

## Reproducir estas ejecuciones

Desde la raíz del repositorio, conservar el mismo modelo y usar nombres de CSV
nuevos. Activar un entorno de Python 3.11 y ajustar la referencia relativa al
checkpoint existente. Los nombres de salida de este ejemplo **ya existen**
después de esta validación:

```powershell
$modeloEvaluacion = 'results/best_model.pt'
python -X utf8 -B evaluate.py --model $modeloEvaluacion --episodes 10 --seed 42 --no-render --output results/evaluacion_best_fija_py311_20261123_01.csv
python -X utf8 -B evaluate.py --model $modeloEvaluacion --episodes 10 --seed 42 --random-start --no-render --output results/evaluacion_best_aleatoria_py311_20261123_01.csv
python -X utf8 -B evaluate.py --model $modeloEvaluacion --episodes 2 --seed 42 --output results/evaluacion_best_visual_py311_20261123_01.csv
```

Cambiar el sufijo antes de repetir: el evaluador rechazará las rutas ocupadas.
El registro completo de inspección, comandos, salidas y verificaciones está en
[evaluacion_best_real_2026-11-23.txt](evidencias/evaluacion_best_real_2026-11-23.txt).

## Alcance y pendientes

La suite de 35 pruebas y `check_environment.py` ya habían pasado con Python 3.11
en la etapa anterior. No se volvieron a ejecutar en esta etapa porque no se
cambió el código. Aquí se añadió la comprobación con un checkpoint entrenado,
CSV reales, reproducibilidad y Pygame; no se sustituyeron por mocks.

Quedan fuera de esta evidencia el desempeño de `final_model.pt`, modelos de otras
semillas de entrenamiento y arquitecturas distintas de 64 neuronas. Estas últimas
dependen de la integración aprobada del trabajo del integrante 2. La prueba real
de interrupción manual durante la conducción tampoco se realizó aquí.

No se hicieron entrenamientos, commits, push, PR ni fusiones. Los archivos del
integrante 2 se conservaron y los cambios nuevos quedaron en resultados y
documentación de este módulo.
