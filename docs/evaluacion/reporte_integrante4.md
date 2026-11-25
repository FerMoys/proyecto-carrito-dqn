# Borrador de reporte individual — integrante 4

Proyecto: Carrito autónomo con Deep Q-Network.

Fecha: 22 de noviembre de 2026.

Actualizado con la evaluación de un checkpoint real el 23 de noviembre de 2026.

Responsabilidad: integración del módulo de evaluación, reproducibilidad,
documentación y preparación de entregables.

## Actividades realizadas

Verifiqué que el repositorio no tuviera cambios locales, actualicé las referencias
de `origin` y comprobé que `main` coincidiera con `origin/main` en el commit
`db1c671`. Creé y seleccioné la rama local `feature/evaluacion-reproducible`
antes de modificar archivos. No realicé commits, push, Pull Requests ni fusiones.

Analicé el evaluador, la interfaz del entorno y el agente. Confirmé que la
selección de acciones ya permitía evaluar sin exploración mediante
`training=False` y que el simulador ya ofrecía inicios aleatorios con semilla.
También identifiqué que las mejoras de carga de arquitecturas del integrante 2
no estaban integradas; conservé esa dependencia sin copiar sus modificaciones.

Implementé en `evaluate.py` los argumentos `--seed`, `--random-start` y
`--output`, manteniendo los argumentos anteriores y sus valores predeterminados.
Añadí validación de episodios positivos y semillas no negativas, así como la
comprobación de que el modelo fuera un archivo existente.

Incorporé la exportación CSV de diez campos por episodio, la creación de sus
directorios y la apertura exclusiva para proteger resultados existentes. La
exportación conserva los datos de cada episodio confirmado y distingue las
filas parciales cuando se cierra la ventana o se interrumpe la evaluación.

Separé el cálculo de métricas de la ejecución: retorno promedio, desviación
estándar poblacional, tasas de éxito y colisión, pasos medios en éxitos y conteos
de episodios completos e interrumpidos. Los episodios parciales se excluyen de
las métricas de desempeño. Incorporé `try/finally` para cerrar el entorno y
coloqué la red en modo evaluación después de cargar el checkpoint.

Creé `tests/test_evaluate.py` con 35 pruebas mediante `unittest`. Actualicé el
README y documenté los comandos, el esquema CSV, las convenciones estadísticas,
la comparación de checkpoints y los límites de reproducibilidad.

## Validación y evidencia

Las 35 pruebas pasaron en Python 3.12.14, sin fallos ni omisiones. Se comprobaron
argumentos, CSV, métricas, ausencia de solicitudes de exploración, protección de
archivos y limpieza ante errores. Una prueba adicional dentro de esa suite usó
el núcleo real del simulador con un agente simulado para verificar que las
mismas semillas reprodujeran los registros.

Los agentes y checkpoints de las pruebas son simulados. No ejecuté
entrenamientos ni evalué el desempeño de un modelo entrenado: no había archivos
`.pt` locales. PyTorch, Gymnasium y Pygame tampoco estaban disponibles en el
intérprete usado. Las pruebas ejecutadas no demuestran la carga real de modelos
ni el funcionamiento gráfico. La evidencia detallada y los pendientes se
encuentran en [validacion.md](validacion.md) y en `evidencias/`.

En una etapa posterior de validación localicé el Python 3.11 registrado en
Windows y confirmé su versión 3.11.9. Las siete dependencias del proyecto se
importaron realmente y sus versiones cumplieron `requirements.txt`. Ejecuté
`check_environment.py` sin mocks: validó espacios, transiciones y render con
código de salida 0. Repetí las 35 pruebas unitarias con Python 3.11; pasaron en
0.727 segundos sin fallos ni omisiones, conservando sus mocks identificados.

El sondeo real de Pygame inicializó el controlador SDL `windows` y detectó tres
escritorios. No hice una demostración de conducción: seguía faltando
`results/best_model.pt` y no había ningún `.pt` local. Un intento real de
evaluación con semilla 42 e inicio aleatorio terminó con `FileNotFoundError`
antes de iniciar episodios. No generé CSV de desempeño ni comparé métricas
inexistentes. En esa etapa solo actualicé documentación y añadí el registro
`evidencias/validacion_real_python311_2026-11-22.txt`.

El 23 de noviembre el usuario suministró los resultados y checkpoints del
integrante 2. Verifiqué que las redes fueran de 64 neuronas, compatibles con el
agente existente, y utilicé directamente su `best_model.pt`, conservando los
archivos originales. Ejecuté diez episodios con inicio fijo y diez con inicio
aleatorio, semillas 42–51, sin exploración ni entrenamiento. Ambas modalidades
alcanzaron diez éxitos y cero colisiones. Los retornos medios fueron
161.6824351951651 y 161.8193127579036, respectivamente; los pasos medios en éxitos
fueron 326.0 y 326.1.

Repetí los diez episodios aleatorios y comprobé identidad byte por byte de sus
CSV. Además ejecuté dos episodios visuales con Pygame, ambos completos y
exitosos. Verifiqué las diez columnas, las filas, las semillas, las modalidades
y la finalización de los cuatro CSV nuevos. Recalculé las métricas desde los
archivos y comprobé que los hashes de los checkpoints originales no cambiaran.
Los detalles están en
[validacion_checkpoint_2026-11-23.md](validacion_checkpoint_2026-11-23.md).

Esta aportación valida y documenta el modelo entrenado por el integrante 2; no
incluye su entrenamiento ni atribuye a mi trabajo sus curvas de aprendizaje.
`final_model.pt` se inspeccionó, pero no se evaluó su desempeño. Las 35 pruebas
unitarias no se repitieron en esta última etapa porque el código no cambió.

## Archivos de la aportación

- `evaluate.py`: ejecución, registros, métricas, argumentos y limpieza.
- `README.md`: instrucciones y explicación del módulo.
- `tests/test_evaluate.py`: pruebas automatizadas.
- `docs/evaluacion/`: validación, evidencias y este borrador.
- `results/evaluacion_best_*_py311_20261123_*.csv`: cuatro CSV de evaluación real (inicio fijo, aleatorio, repetición y visual).

No modifiqué archivos de los integrantes 1, 2 o 3, dependencias ni modelos.

## Trabajo pendiente del equipo

La validación del entorno, la suite con Python 3.11, la carga del checkpoint de
64 neuronas, los CSV reales, la comparación de modalidades, la repetición y la
demostración visual ya se completaron. No se probó una interrupción manual de
Pygame durante un episodio real. La verificación de otras arquitecturas queda
pendiente de la integración aprobada de `feature/agente-dqn`; los modelos de
otras semillas de entrenamiento tampoco se evaluaron aquí. La integración de
esta aportación requiere revisión del equipo mediante Pull Request.

## Propuesta de commit (no ejecutada)

Mensaje sugerido:

```text
feat(evaluacion): añade evaluación reproducible, CSV y pruebas
```

Después de revisar el diff y autorizar el commit, desde la raíz del repositorio:

```powershell
git branch --show-current
git status
git diff --check
git diff -- evaluate.py README.md
git diff --no-index -- /dev/null tests/test_evaluate.py
git add -- evaluate.py README.md tests/test_evaluate.py docs/evaluacion/
git add -- results/evaluacion_best_fija_py311_20261123_01.csv results/evaluacion_best_aleatoria_py311_20261123_01.csv results/evaluacion_best_aleatoria_py311_20261123_02.csv results/evaluacion_best_visual_py311_20261123_01.csv
git diff --cached --stat
git diff --cached
git commit -m "feat(evaluacion): añade evaluación reproducible, CSV y pruebas"
```

`git diff --no-index` devuelve 1 cuando muestra diferencias; `/dev/null` es la
ruta especial reconocida por Git. Confirma que la rama sea
`feature/evaluacion-reproducible` antes de preparar el commit. No se propone
ejecutar push ni crear el PR sin autorización.

## Borrador de Pull Request (no creado)

Título: **Evaluación DQN reproducible con CSV, métricas y pruebas**.

Base prevista: `main`. Rama de trabajo: `feature/evaluacion-reproducible`.

Descripción propuesta:

> El evaluador utilizaba únicamente inicios fijos y no exportaba registros por
> episodio. Este cambio añade semillas configurables, inicios aleatorios y CSV
> con protección contra sobrescritura, conservando los argumentos originales.
>
> El resumen distingue episodios completos, interrumpidos y no iniciados, y
> calcula las métricas de desempeño solo sobre los completos. Se cierra el
> entorno con `try/finally`. Incluye documentación y 35 pruebas aprobadas en
> Python 3.12.14 y Python 3.11.9; las pruebas usan mocks y una comprobación del
> núcleo real del simulador con agente simulado. `check_environment.py` pasó
> realmente con Python 3.11.9, sin mocks.
>
> Se evaluó realmente el `best_model.pt` de 64 neuronas del integrante 2: diez
> episodios fijos y diez aleatorios, con 100 % de éxito y 0 % de colisiones en
> ambas modalidades. La repetición aleatoria produjo un CSV idéntico byte por
> byte y Pygame completó dos episodios visuales. Incluye los cuatro CSV reales,
> procedencia y hash del modelo. No se modificaron ni entrenaron checkpoints.
>
> Pendiente: interrupción manual durante un episodio real y modelos de otras
> semillas. Otras arquitecturas se verificarán tras integrar los cambios del
> integrante 2. Estos resultados corresponden a este checkpoint y a las semillas
> evaluadas; no se afirma generalización a todas las condiciones.
