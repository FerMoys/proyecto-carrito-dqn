# Proyecto final Carrito autónomo con DQN

Este proyecto implementa una pista 2D, un automóvil con dinámica simplificada y
un agente de aprendizaje por refuerzo profundo. La red recibe nueve variables y
produce un valor Q para cada una de nueve acciones discretas derivadas de los
controles acelerar, frenar, girar a la izquierda y girar a la derecha.

## Instalación recomendada en Windows 11

1. Instala Python 3.11 o crea un entorno de Anaconda con esa versión.
2. Abre una terminal en esta carpeta.
3. Crea y activa un entorno virtual:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
```

4. Instala las dependencias:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

5. Comprueba el entorno:

```powershell
python check_environment.py
```

## Secuencia de uso

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
`model` conserva la ruta suministrada y `reward` es la suma sin descuento de las
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
dispositivo, modalidad de inicio y lista de semillas. Registra el hash del modelo
si necesitas comprobar su identidad; el CSV contiene la ruta, no el contenido ni
su hash. Diferentes dispositivos o versiones de PyTorch pueden producir diferencias
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

La carga actual en esta base utiliza redes de **64 neuronas por capa oculta**.
La validación de arquitecturas de 32 o 128 queda pendiente de que el equipo integre
y apruebe `feature/agente-dqn`. Este módulo utiliza `DQNAgent.load()` sin copiar
sus cambios ni adaptar arquitecturas por su cuenta.

### Pruebas del módulo

```powershell
python -B -m unittest discover -s tests -p test_evaluate.py -v
```

Las pruebas usan `unittest`, agentes y checkpoints simulados en archivos temporales;
no requieren modelos entrenados ni realizan entrenamiento. Una prueba adicional
usa el núcleo real del simulador con un agente simulado y requiere NumPy; se omite
explícitamente si no está instalado. Las pruebas con mocks no acreditan la carga
real de PyTorch ni el funcionamiento de la ventana Pygame.

La evidencia de ejecución y las comprobaciones pendientes se encuentran en
[docs/evaluacion/validacion.md](docs/evaluacion/validacion.md). El borrador del
reporte individual está en
[docs/evaluacion/reporte_integrante4.md](docs/evaluacion/reporte_integrante4.md).

