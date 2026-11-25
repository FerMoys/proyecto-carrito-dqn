# Preparación de la integración final

El integrante 4 asume la integración, corrección y entrega de los cuatro
módulos. La base de trabajo es `feature/evaluacion-reproducible` en `064acca`,
con `e8a2d92` en su historial. Las ramas y artefactos publicados sirven de
entrada. La aceptación del sistema seguirá con los datos disponibles.

## Manifiesto y recuperación

Cuando se exporta un CSV, el evaluador crea un manifiesto JSONL al lado de
la primera salida. `--manifest` permite indicar otra ruta nueva o registrar
una ejecución sin CSV. El contrato de los CSV de diez y ocho columnas sigue
siendo el mismo. El JSONL comienza con una cabecera `evaluation` de esquema
1 y continúa con los registros `episode`. Un registro `finish` distingue
una corrida completada de una que terminó con episodios pendientes o
parciales. La ausencia del cierre indica una salida abrupta.

La cabecera registra el nombre portable y SHA-256 del checkpoint, arquitectura,
semilla de entrenamiento, configuración numérica del entrenamiento guardada y
configuración efectiva del entorno. Registra también la semilla inicial, la
regla de reinicio, la modalidad, Python, PyTorch, dispositivo, HEAD de Git y
SHA-256 de los archivos fuente que afectan a la evaluación. Estos hashes
identifican el código local incluso con cambios sin commit. Cada episodio
incluye su semilla efectiva y las diez columnas detalladas.

La secuencia de escritura es manifiesto, CSV detallado, CSV experimental.
Antes de intentar los CSV se vacía el búfer del manifiesto y se solicita
`fsync` al sistema de archivos. La apertura exclusiva de las salidas
protege los destinos existentes. Un fallo de preparación retira solo los
archivos creados por esa preparación. Un fallo posterior conserva los archivos
y puede dejar un CSV un episodio por detrás del manifiesto.

1. Conservar los originales, incluido el manifiesto.
2. Ejecutar `python evaluate.py verify-exports <manifiesto>`. Si algún CSV
   no está al lado del manifiesto, indicar su ruta con `--output` o
   `--csv-output`. Un desacuerdo en la cabecera, orden o valores devuelve
   código 1. Una verificación que coincide con el prefijo registrado puede
   devolver 0 aunque la corrida no haya alcanzado su cierre.
3. Ejecutar `python evaluate.py recover-exports <manifiesto> --output <detalle-nuevo> --csv-output <experimento-nuevo>`.
   Puede recuperarse una sola salida. Todos los destinos deben ser nuevos.
4. Volver a verificar las copias recuperadas con sus rutas explícitas. El
   manifiesto original sigue acreditando la procedencia.

La recuperación usa el prefijo de registros completos y puede omitir una
última línea sin salto final. Rechaza registros completos inválidos. El
experimental vuelve a omitir episodios parciales. No se reanuda la
simulación ni se recupera un episodio que no alcanzó a registrarse. Los CSV
independientes no son atómicos. Si el manifiesto se pierde o queda corrupto
antes de su último registro, no hay una fuente completa para reconstruirlos.

## Carga restringida y dependencias

Temporalmente, el evaluador lee una sola vez con
`torch.load(..., map_location="cpu", weights_only=True)` y aplica los pesos
a la política que ya construye el agente de esta rama. Para inferencia no
restaura el optimizador ni la red objetivo. La arquitectura adaptativa queda
pendiente. El evaluador no escribe variables de entorno. Con una sola variable
`TORCH_FORCE_*` activa la llamada sigue siendo explícitamente restringida.
Si ambas están activas, se informa del conflicto antes de deserializar.

La política de ejecución acepta versiones estables de 2.10 a 2.13 y desde
2.15. Se propone reemplazar `torch>=2.2` por **`torch>=2.10,!=2.14.*`** al integrar.
Esto excluye instalaciones antiguas y 2.14.x. Quienes ya tienen una versión
aceptada no necesitan cambiarla para cumplir el rango. Esta etapa comprobó
el uso real de 2.15.0+cpu con Python 3.11. La compatibilidad de las otras
versiones admitidas no se validó con un checkpoint real. Al cerrar la
integración conviene registrar la versión probada para repetir la instalación.
`requirements.txt` se mantuvo intacto.

El rango se apoya en los [avisos de PyTorch hasta 2.9.1](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p)
y [de PyTorch 2.14](https://github.com/pytorch/pytorch/security/advisories/GHSA-v32g-mr63-3j83).
La semántica del parámetro y sus variables se describe en la
[documentación de serialización](https://docs.pytorch.org/docs/2.15/notes/serialization.html).

Tras incorporar la carga adaptable, el integrante 4 centralizará la lectura
explícita en `DQNAgent.load()`. El agente deberá devolver la identidad del
archivo leído para el manifiesto, validar las dimensiones y reconstruir la
red si cambia el ancho. Incorporar una prueba Base32 sintética sin entrenamiento
para cubrir la adaptación y otra de regresión con el Base64 conocido.

## Integración aislada para aprobación

1. Revisar este diff completo y aprobar primero cómo se preservará el trabajo
   actual. Una vez autorizado, crear la rama local
   `feature/integracion-final` desde el estado aprobado de
   `feature/evaluacion-reproducible`, en un worktree separado. La rama actual
   conserva sus commits y no se hace una fusión con cambios locales pendientes.
2. Revisar conjuntamente las referencias `abe9d49` y `1be444f`. Con la
   autorización de fusión, incorporar `origin/feature/experimentos` en la
   rama aislada. Esa referencia ya incluye al agente y el entrenamiento
   adaptable. Conservar los CSV históricos y revisar la regla de exclusión de
   resultados y checkpoints.
3. Resolver el conflicto de `evaluate.py` conservando la evaluación
   reproducible, las dos exportaciones, el manifiesto, la recuperación y
   el cierre seguro. Migrar la carga restringida a `DQNAgent.load()`.
4. Corregir los notebooks, el promedio de pasos de episodios exitosos, las
   rutas de entrada y exportaciones, la compatibilidad de Matplotlib y las
   dependencias de Jupyter en la rama aislada. Usar nombres nuevos en los
   scripts de referencia y entrenamientos de diagnóstico autorizados.
5. Repetir la aceptación desde una instalación limpia. Verificar los notebooks
   desde la raíz y desde su carpeta, revisar gráficas y exportaciones, correr
   referencias y grabar la demostración con el Base64 identificado.

## Alcance de la entrega

**Demostrado:** ejecución real del Base64 conocido con inicio fijo y aleatorio,
salidas detallada y experimental, manifiesto y recuperación en destinos nuevos.
Las 20 filas detalladas coinciden con sus referencias históricas. Pasaron
60 pruebas en Python 3.11.9 y 60 en Python 3.12.14. La demostración funcional
final utilizará ese checkpoint. Consulta las
[evidencias de la etapa de corrección](evidencias/correccion_exportaciones_2026-11-24.txt).

**Auditado:** los 240 registros publicados mantienen su respaldo histórico y
su consistencia matemática ya comprobada.

**No reproducible desde los originales:** el lote experimental no se puede
volver a ejecutar con sus seis modelos. La búsqueda queda cerrada en esta
etapa. Las pruebas sintéticas de adaptación podrán comprobar el código sin
atribuir resultados a esos modelos. El informe, las diapositivas y el video
mantendrán explícita la diferencia entre esas evidencias.
