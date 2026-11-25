# Validación del módulo de evaluación

Fecha: 22 de noviembre de 2026 (America/Guatemala).

Actualización: la validación con el checkpoint proporcionado el 23 de noviembre
ya se completó; véase [validacion_checkpoint_2026-11-23.md](validacion_checkpoint_2026-11-23.md).
Las secciones iniciales conservan la evidencia y los bloqueos de cada etapa.

Nota de publicación (24 de noviembre de 2026): las rutas personales se
anonimizaron y los comandos se presentan con referencias portables. No se
modificaron semillas, métricas ni resultados experimentales. Las evidencias
conservan sus mensajes técnicos y versiones; las rutas son referencias genéricas.

## Base y alcance

- Rama local: `feature/evaluacion-reproducible`.
- Base: `main`, commit `db1c671f4d1b854b0fd5691d9706f06e2f6cc410`.
- Se ejecutó `git fetch origin`; `main...origin/main` devolvió `0 0` antes de crear la rama.
- La rama se creó con el árbol limpio y se confirmó antes de editar archivos.
- No se incorporaron cambios de `feature/agente-dqn`.
- No se cambiaron dependencias, hiperparámetros, simulador, entrenamiento ni modelos.

## Entorno de las pruebas iniciales de implementación

Se utilizó el intérprete ya disponible. El comando se muestra con el nombre
portable `python`, correspondiente a la versión indicada debajo:

```powershell
python -B -m unittest discover -s tests -p test_evaluate.py -v
```

Versión observada: Python **3.12.14**, Windows, AMD64. NumPy está disponible.
PyTorch, Gymnasium y Pygame no están disponibles en este intérprete. No se
instalaron paquetes ni se modificó `requirements.txt`.

En esa primera etapa se comprobó la sintaxis compatible con Python 3.11. La
ejecución en 3.11 se completó después, como se documenta en la validación real.

## Resultado automatizado

La ejecución inicial terminó con **35 pruebas aprobadas**, sin fallos, errores
ni omisiones, en 1.731 segundos. El registro persistido de verificación puede
tener otra duración: las mediciones de tiempo corresponden a cada ejecución.

Registro de verificación: [pruebas_2026-11-22.txt](evidencias/pruebas_2026-11-22.txt),
35 pruebas aprobadas en 1.054 segundos.

Cobertura funcional comprobada:

| Área | Evidencia |
|---|---|
| Compatibilidad | Valores originales, comando anterior, visualización por defecto y `--no-render`. |
| Argumentos nuevos | Semilla, inicio aleatorio, ruta CSV y semilla cero. |
| Validación | Rechazo de episodios no positivos, argumentos no enteros y semillas negativas. |
| Acciones | Todas las llamadas usan `training=False`; no se llama entrenamiento ni memoria. |
| Registros | Retorno acumulado, pasos realmente devueltos, semillas consecutivas y numeración desde 1. |
| CSV | Diez filas simuladas, cabecera exacta, valores, ruta Unicode y creación de directorios. |
| Archivos existentes | Contenido preservado; se rechaza salida igual al modelo y se usa apertura exclusiva. |
| Métricas | Valores calculados a mano, desviación poblacional, truncación completa y ausencia de éxitos/datos. |
| Interrupciones | Cierre de ventana, Ctrl+C en selección, pasos y reinicio; conservación de filas parciales. |
| Limpieza | `env.close()` ante ejecución normal, fallos de carga, reinicio, pasos y apertura del CSV. |
| Reproducibilidad | Entorno simulado y núcleo real `CarSimulator`, ambos con agente simulado. |

Las fixtures `.pt` son archivos temporales vacíos, identificados como simulados;
la carga del agente se sustituye con mocks. Los CSV de esas pruebas contienen
datos controlados y temporales, no resultados de un modelo entrenado. No se
publicaron como resultados experimentales en `results/`.

## Comprobaciones de consola

El registro de consola en `evidencias/` conserva la ayuda y los códigos de salida
obtenidos al invocar el script como proceso. En ausencia de
`results/best_model.pt`, los dos comandos solicitados deben fallar indicando que
no existe el modelo, antes de intentar importar PyTorch. No se genera un CSV real.
La ejecución completa de ambos comandos sí se verifica con mocks en la suite.

Registro: [consola_2026-11-22.txt](evidencias/consola_2026-11-22.txt).
Las cinco comprobaciones de consola pasaron. También se analizaron ambos archivos
Python con la gramática de Python 3.11 mediante `ast.parse`; esto verifica la
sintaxis, no reemplaza ejecutar las pruebas con esa versión del intérprete.

La primera captura de consola falló por una diferencia de codificación entre
el proceso de captura y su subproceso (CP1252 frente a UTF-8). Se repitió la
captura usando `-X utf8` en ambos procesos y pasó. No fue un fallo del evaluador
ni requirió cambios en su código.

## Convenciones y límites

- Un episodio completo tiene `terminated=1` o `truncated=1`.
- Una fila con ambas banderas en cero corresponde a un episodio iniciado e interrumpido.
- Los episodios no iniciados no se cuentan como interrumpidos ni generan filas.
- Un reinicio interrumpido antes de devolver datos no confirma un episodio nuevo.
- Solo se registran transiciones que `env.step()` devolvió correctamente.
- Los promedios y tasas excluyen filas parciales; la desviación es poblacional.
- Si falla una llamada por una excepción distinta de Ctrl+C, el error se propaga y las filas ya exportadas se conservan. El episodio que falló no se presenta como completo.
- El CSV se crea después de cargar el agente, antes del primer episodio evaluado. Un fallo de carga no produce CSV.
- Ctrl+C durante la preparación del agente cierra el entorno, pero puede ocurrir antes de que exista un CSV.
- El vaciado del búfer tras cada fila no garantiza recuperación ante una caída del sistema, corte eléctrico o interrupción durante la propia escritura.
- El modelo y el entorno deben permanecer iguales para comparar ejecuciones. Se usa el entorno predeterminado; no se restaura una configuración de pista desde el checkpoint.
- La semilla controla el reinicio del simulador. No se garantiza igualdad numérica entre dispositivos o versiones distintas.

## Validación real posterior con Python 3.11

Esta etapa se ejecutó también el 22 de noviembre de 2026. Se mantuvo activa
`feature/evaluacion-reproducible`, conservando los cambios anteriores sin
prepararlos para commit. En esta etapa se actualizó exclusivamente documentación
en `docs/evaluacion/`; no se modificaron el evaluador, README ni las pruebas.

### Intérprete y dependencias reales

Se encontró Python de Microsoft Store mediante el registro de Windows. No estaba
resuelto por `python` o `py` en el PATH de esta sesión. La ejecución requirió
salir de la restricción de acceso del entorno de herramientas; no fue necesario
instalar Python ni paquetes. El intérprete confirmó **Python 3.11.9**; su
ubicación se reemplazó por esta referencia genérica para publicación:

```text
python
```

Todos los imports de esta tabla se ejecutaron realmente, sin mocks. Sus versiones
cumplen los límites declarados actualmente en `requirements.txt`:

| Dependencia | Versión observada en Python 3.11 |
|---|---|
| NumPy | 2.4.6 |
| pandas | 2.3.3 |
| Matplotlib | 3.11.2 |
| Gymnasium | 1.4.0 |
| Pygame | 2.6.1 |
| PyTorch | 2.15.0+cpu |
| TensorBoard | 2.21.0 |

PyTorch informó `cuda=False`: la disponibilidad comprobada es CPU. El diagnóstico
previo con el intérprete auxiliar 3.12 falló al importar Gymnasium; esa carencia
pertenece a ese intérprete, no al Python 3.11 encontrado posteriormente.

### Resultados de esta etapa

| Comprobación | Resultado real | Pendiente o alcance |
|---|---|---|
| Rama y Git | Rama correcta; cambios anteriores conservados, ninguno preparado para commit. | No se crearon commits, push, PR ni fusiones. |
| Python 3.11 y dependencias | Python 3.11.9; siete dependencias importadas correctamente. | Ninguna dependencia faltante en ese intérprete. |
| `check_environment.py` | Código de salida 0: espacios, transiciones y render validados. | Validación real de Gymnasium y render `rgb_array`, sin mocks. |
| Suite de 35 pruebas | 35 aprobadas en 0.727 segundos, sin errores, fallos ni omisiones. | La suite mantiene sus mocks; no acredita la carga de un checkpoint entrenado. |
| Inicio fijo: 10 episodios | No ejecutado. | Falta `results/best_model.pt`. |
| Inicio aleatorio: 10 episodios, semilla 42 | El intento real terminó con `FileNotFoundError`, antes de iniciar episodios. | Falta el checkpoint; cero episodios evaluados. |
| CSV de evaluación real | No generado; no se validaron filas reales. | Las diez columnas y filas simuladas solo están cubiertas por la suite unitaria. |
| Pygame y demostración visual | `pygame.display.init()` pasó; controlador SDL `windows`, tres escritorios detectados. | No se ejecutó la demostración de dos episodios por falta del modelo. Inicializar el controlador no prueba una ventana de conducción. |
| Comparación fijo/aleatorio | No disponible. | No existen retornos, tasas ni pasos de un modelo entrenado en esta etapa. |

Se volvió a buscar `*.pt` en el repositorio y no se encontró ninguno. No se
sustituyó el modelo por un agente aleatorio, una fixture ni un entrenamiento
nuevo. No se crearon ni sobrescribieron CSV de resultados ni checkpoints.

Registro completo: [validacion_real_python311_2026-11-22.txt](evidencias/validacion_real_python311_2026-11-22.txt).

### Cómo completar las comprobaciones pendientes

Solicitar al integrante 2 un checkpoint ya entrenado, compatible con la red
actual de 64 neuronas. Guardarlo como `results/best_model.pt` solo si esa ruta
sigue libre, o usar su ruta real con `--model`. No se necesita entrenar un modelo
nuevo para esta validación. No se debe modificar el checkpoint suministrado.

Desde la raíz del repositorio, activar el entorno de Python 3.11 antes de usar
estos comandos portables:

```powershell
python -X utf8 -B check_environment.py
python -X utf8 -B -m unittest discover -s tests -p test_evaluate.py -v
```

Cuando exista el modelo, usar nombres de salida nuevos en cada ejecución:

```powershell
python -X utf8 -B evaluate.py --model results/best_model.pt --episodes 10 --seed 42 --no-render --output results/evaluacion_fija_py311_01.csv
python -X utf8 -B evaluate.py --model results/best_model.pt --episodes 10 --seed 42 --random-start --no-render --output results/evaluacion_aleatoria_py311_01.csv
python -X utf8 -B evaluate.py --model results/best_model.pt --episodes 2 --seed 42
```

Esos tres comandos son instrucciones pendientes, no evidencia de episodios
ejecutados. El evaluador rechaza CSV existentes; cambiar el sufijo si las rutas
ya están ocupadas. Para comparar fijo y aleatorio se usan aquí las mismas
semillas 42 a 51. La demostración visual usa inicio fijo y dos episodios.

Tras completar las ejecuciones, comprobar cabecera exacta, episodios 1 a 10,
semillas 42 a 51 y banderas de inicio fijo/aleatorio. Si hubo una interrupción,
registrar el número real de filas y filtrar los episodios completos al comparar
las métricas. No se puede calcular esa comparación mientras falte el checkpoint.

## Pendientes al cierre del 22 de noviembre

1. Disponer de un checkpoint entrenado compatible de 64 neuronas y verificar su carga real.
2. Ejecutar las evaluaciones fija y aleatoria; comprobar diez filas reales cuando no haya interrupción.
3. Repetir la evaluación aleatoria con el mismo modelo y semillas, usando otro nombre CSV; comparar las filas en el mismo entorno y dispositivo.
4. Ejecutar los dos episodios visuales, comprobar la ventana Pygame y el cierre durante un episodio.
5. Comparar métricas reales de ambas modalidades, conservando los archivos originales.
6. Después de integrar y aprobar `feature/agente-dqn`, verificar checkpoints de otras arquitecturas sin adaptar su carga en el evaluador.

Al cierre de esa etapa no se encontraron archivos `.pt` locales. Por eso no se
entregó entonces un CSV de desempeño real ni se atribuyeron tasas de éxito o
colisión a un modelo entrenado.

## Estado al 23 de noviembre: checkpoint disponible y evaluado

El usuario proporcionó un `best_model.pt` de 64 neuronas en almacenamiento
externo. Se utilizó directamente con `--model`, sin copiarlo ni modificarlo.
Esto resolvió el bloqueo anterior sin entrenamientos ni integración de otra rama.

| Comprobación nueva | Resultado |
|---|---|
| Carga real de `best_model.pt` | Compatible con la arquitectura actual; carga aprobada. |
| Diez episodios fijos | Diez éxitos; cero colisiones; retorno medio 161.6824351951651. |
| Diez episodios aleatorios, semillas 42–51 | Diez éxitos; cero colisiones; retorno medio 161.8193127579036. |
| Repetición aleatoria | Diez episodios adicionales; CSV idéntico byte por byte. |
| Demostración real Pygame | Dos episodios completos y exitosos, con inicio fijo. |
| Esquema y contenido de CSV | Diez columnas exactas; filas, semillas, modalidad y finalización verificadas en los cuatro archivos. |
| Protección de originales | Hashes SHA-256 de ambos checkpoints sin cambios. |

Las pruebas nuevas fueron reales, sin mocks. La suite unitaria anterior sigue
siendo evidencia de pruebas con mocks y no se volvió a ejecutar porque no cambió
el código. No se probó un cierre manual de ventana a mitad de episodio; quedan
pendientes esa comprobación y las arquitecturas alternativas, estas últimas
sujetas a la integración aprobada del equipo.

Resultados, comandos, limitaciones y vínculos a los cuatro CSV:
[validacion_checkpoint_2026-11-23.md](validacion_checkpoint_2026-11-23.md).
