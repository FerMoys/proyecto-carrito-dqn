# Bitácora del proyecto Carrito Autónomo con DQN

La bitácora describe el código de `e566cdd9b3e57a734028cb0c7f119f3a48f43902`
en `feature/integracion-final`. Distingue los parámetros que se pueden leer en
el código, los registrados en un checkpoint y los campos publicados para los
experimentos. [PR integral #2](https://github.com/FerMoys/proyecto-carrito-dqn/pull/2).

## 1. Simulador y entorno

| Elemento | Configuración del código revisado | Fuente |
| --- | --- | --- |
| Pista | Corona elíptica en una superficie de 1000 × 700 px y centro (500, 350) px. Semiejes exteriores 430 y 285 px, interiores 255 y 125 px. | [TrackConfig](../config.py), [CarSimulator](../simulator_core.py) |
| Sensores | Cinco rayos a −60°, −30°, 0°, 30° y 60° respecto al vehículo. Cada rayo avanza en incrementos de 4 px hasta el borde o el límite de 175 px. | [Sensores y observación](../simulator_core.py) |
| Nueve observaciones | Cinco distancias divididas entre el alcance, velocidad dividida entre la máxima, seno y coseno del error angular respecto de la tangente y posición lateral entre −1 y +1. El vector es `float32`. | [Núcleo](../simulator_core.py), [interfaz Gymnasium](../car_env.py) |
| Nueve acciones | IDs 0 a 8. Inercia, acelerar, frenar, girar a la izquierda, girar a la derecha, acelerar a la izquierda, acelerar a la derecha, frenar a la izquierda y frenar a la derecha. | [ACTION_NAMES y ACTION_TO_CONTROL](../config.py) |
| Dinámica | Paso de 1/30 s. Velocidad limitada a 155 px/s, aceleración de 105 px/s², frenado de 150 px/s² y fricción de 28 px/s² al soltar el control longitudinal. El giro de 2.35 rad/s se escala por `0.30 + 0.70 × velocidad / velocidad máxima`. La posición avanza según la orientación y la velocidad resultantes. | [CarConfig](../config.py), [step](../simulator_core.py) |
| Recompensa | Por transición se suma 18 veces el cambio angular con signo en radianes, 0.035 veces la velocidad normalizada multiplicada por el coseno del error respecto de la tangente limitado inferiormente a cero y 0.01. Se resta 0.004 veces el valor absoluto del mando de giro y 25 si hay colisión. Completar la vuelta añade 35. | [step](../simulator_core.py) |
| Fin del episodio | El objetivo predeterminado es una vuelta. Una colisión o alcanzar el objetivo termina el episodio. Alcanzar 1 800 transiciones sin terminar lo trunca. La detección de colisión aplica un radio de 12 px. | [Núcleo](../simulator_core.py), [TrackConfig y CarConfig](../config.py) |
| Representación | `CarTrackEnv` expone el contrato Gymnasium y renderiza en Pygame. El diagnóstico valida los espacios, las transiciones y el renderizado RGB. | [Entorno](../car_env.py), [diagnóstico](../check_environment.py) |

## 2. Agente DQN y entrenamiento

La red estima nueve valores Q a partir de las nueve observaciones. Tiene dos
capas ocultas del mismo ancho con activación ReLU y una salida lineal por
acción. Base32 usa `9 → 32 → 32 → 9` (1 673 parámetros) y Base64 usa
`9 → 64 → 64 → 9` (5 385 parámetros). El conteo incluye pesos y sesgos de
las tres capas lineales. El ancho se puede configurar con `--hidden-size`.
[QNetwork](../dqn_agent.py), [CLI de entrenamiento](../train.py).

El Replay Buffer guarda copias de `(s, a, r, s', done)` y extrae minilotes
uniformes sin reemplazo con un generador propio. El entrenamiento actual
marca `done` tanto en terminación como en truncación. Para cada transición
muestreada calcula el objetivo de Bellman `r + γ(1 − done) max_a' Q_objetivo(s', a')`,
compara con `Q_política(s,a)` mediante Smooth L1 y actualiza la política con
AdamW y AMSGrad. La red objetivo se inicializa con la política y recibe una
actualización suave `θ_objetivo ← (1 − τ) θ_objetivo + τ θ_política` tras cada
paso de optimización. [ReplayBuffer y train_step](../dqn_agent.py), [bucle](../train.py).

Durante el entrenamiento, la política toma una acción uniforme con
probabilidad `ε`, o selecciona el mayor Q. La fórmula es
`ε(t) = ε_final + (ε_inicial − ε_final) exp(−t / decaimiento)` y `t`
cuenta las acciones de entrenamiento. La evaluación solicita `training=False`
y selecciona la acción de mayor Q. [DQNAgent](../dqn_agent.py), [evaluate_episodes](../evaluate.py).

| Hiperparámetro | Valor predeterminado y confirmado en los dos checkpoints Base64 del paquete |
| --- | --- |
| Ancho y configuración de episodios | 64 por capa y máximo configurado de 600 episodios |
| Semilla y tasa de aprendizaje | 42 y 0.0003 |
| Descuento y red objetivo | γ = 0.99 y τ = 0.005 |
| Replay y muestreo | Capacidad de 50 000 transiciones, minilote de 128 y comienzo de optimización con 1 000 transiciones |
| Exploración | ε inicial 1.0, ε final 0.05 y decaimiento de 45 000 acciones |
| Recorte y parada | Norma de gradiente máxima 10.0, parada temprana al alcanzar al menos 90 % de éxito en una ventana completa de 30 episodios |

Los valores predeterminados se verifican en [TrainConfig](../config.py).
La identificación del modelo entregado y los campos guardados constan en
[trazabilidad](evaluacion/trazabilidad_checkpoints.md) y en la
[prueba de carga restringida](evaluacion/evidencias/carga_centralizada_2026-11-24.txt).

## 3. Semillas, condiciones y diseño

- **Inicio fijo.** El simulador coloca el vehículo en la parte inferior de la
  línea media, en `θ = π/2`, velocidad cero y orientación tangente.
- **Inicio aleatorio.** Añade variación uniforme de ±0.08 rad a la posición
  angular y de ±0.04 rad a la orientación. Mantiene la velocidad inicial en cero.
  Estas condiciones describen el [reinicio implementado](../simulator_core.py).
- **Semillas en el código.** El entrenamiento inicializa Python, NumPy y
  PyTorch con la semilla configurada. El bucle usa la semilla base más el
  número de episodio para los reinicios. El evaluador usa la semilla de
  evaluación más el índice desde cero. En el CSV experimental, `semilla`
  identifica el entrenamiento del DQN o la corrida del agente de referencia.
  [Entrenamiento](../train.py), [evaluación](../evaluate.py),
  [aleatorio](../random_baseline_comparison.py), [heurístico](../heuristic_baseline_comparison.py).
- **Lote histórico.** Base32 y Base64 se cruzaron con 42, 123 y 999 y con
  ambos inicios. Se publicaron 10 episodios por combinación, 120 DQN en total.
  Aleatorio y heurístico siguen el mismo cruce y aportan otros 120 episodios.
  Cada combinación política–inicio resume 30 episodios. Las etiquetas de
  semilla de los CSV no fijan por sí solas las semillas efectivas por episodio
  ni todos los hiperparámetros del lote original. [CSV integrado](../csv_results/comparison_results_integrado.csv),
  [auditoría de los experimentos](evaluacion/integracion_experimentos.md).

## 4. Entrenamiento y modelo Base64 de demostración

`result_model_prueba.zip` contiene `best_model.pt`, `final_model.pt`,
`training_log.csv`, `training_curves.png` y `evaluation_summary.csv`.
La lectura restringida confirmó que ambos checkpoints usan Base64 y semilla
de entrenamiento 42, y guardan **45 552 pasos**. El CSV registra **145
episodios** cuya suma de pasos coincide. Los últimos 30 registran 27 éxitos
(90 %), el umbral que usa el entrenador. Este contraste comprueba la
consistencia entre los archivos del paquete. El checkpoint carece del
commit, las versiones del entorno y los parámetros efectivos de pista y
vehículo usados al entrenarlo. [Trazabilidad del paquete](evaluacion/trazabilidad_checkpoints.md),
[organización de la entrega](evaluacion/cierre_entrega.md).

La demostración lee `best_model.pt`, con SHA-256
`8fca3c72f147cd94af182417e55e4093ed0f6ff8eaf3448e654650d51c04f5b1`.
Las semillas de evaluación 42–51 se usaron en diez episodios fijos y diez
aleatorios. El paquete queda fuera de Git y está previsto adjuntarlo a
GitHub Releases tras integrar `main`. El README documenta la extracción,
las huellas, los comandos y la apertura de Pygame. [README](../README.md),
[evidencia real Base64](evaluacion/evidencias/carga_centralizada_2026-11-24.txt).

## 5. Incidencias y decisiones de integración

| Incidencia | Impacto | Solución comprobada | Referencia |
| --- | --- | --- | --- |
| Un solo formato de salida no servía a todas las verificaciones. | El análisis experimental y la auditoría de episodios requieren campos diferentes. | CSV detallado de 10 columnas y experimental de 8 de la misma corrida. El detallado registra semillas efectivas y banderas de fin. | `064acca`, [integración del evaluador](evaluacion/integracion_experimentos.md) |
| Interrupciones o fallos entre escrituras. | Los CSV pueden quedar desalineados o con datos parciales. | Registro por episodio en JSONL, detección con `verify-exports` y recuperación en rutas nuevas con `recover-exports`. Una fila interrumpida del detallado queda fuera de las métricas y del CSV experimental. | `802fb31`, [regresiones](../tests/test_evaluate.py), [evidencia](evaluacion/evidencias/correccion_exportaciones_2026-11-24.txt) |
| Carga de modelos dispersa y acoplada a la arquitectura. | Identidad y dimensiones ambiguas, y riesgo de deserialización amplia. | Carga centralizada con `weights_only=True`, verificación de dimensiones y procedencia, una lectura por evaluación y modos separados de inferencia y reanudación. En reanudación se recuperan redes, optimizador y pasos. El formato guardado omite el búfer y el estado aleatorio. | `a1ed530`, [cargador](../dqn_agent.py), [pruebas](../tests/test_checkpoint_load.py) |
| Rutas, salidas y dependencias de notebooks. | El arranque dependía del directorio y de la versión de Matplotlib. | Cuadernos reorganizados bajo `notebooks/`, entradas configurables, salidas exclusivas y dependencias opcionales con kernel documentado. Validación desde raíz y carpeta en Python 3.11.9 y 3.12.14 con Matplotlib 3.11.2 y 3.8.4. | `71abfbd`, [instalación](../README.md), [pruebas de notebooks](evaluacion/evidencias/validacion_notebooks_2026-11-24.md) |
| Reutilización del destino de entrenamiento. | Los archivos nuevos podían reemplazar modelos y registros previos. | Reserva exclusiva al arrancar. Se permiten actualizaciones del mejor modelo dentro de la propia corrida. Las cinco regresiones se suman a las 64 pruebas anteriores. | `e566cdd`, [entrenamiento](../train.py), [regresiones](../tests/test_train_output.py), [cierre](evaluacion/cierre_entrega.md) |
| Falta de los seis checkpoints del lote DQN original. | Los CSV por arquitectura y semilla carecen de un modelo verificable para una reejecución exacta. | Conservar el lote como evidencia histórica auditada. Usar el Base64 conocido para la demostración y el Base32 sintético para verificar compatibilidad técnica. | [Trazabilidad](evaluacion/trazabilidad_checkpoints.md), [cotejo cuantitativo](evidencia_cuantitativa.md) |

## 6. Alcance de las verificaciones

La instalación limpia en Windows y la validación del cierre técnico
usaron uv y PyTorch 2.15.0+cpu. La suite registrada para `e566cdd`
aprobó 69/69 pruebas con Python 3.11.9 y 3.12.14. La evidencia Base32
sintética verifica adaptación e inferencia. La comparación histórica de
rendimiento Base32 se apoya en los CSV y sus tres semillas. Los gráficos
del entrenamiento corresponden exclusivamente al modelo Base64 del ZIP.
Las métricas y sus fuentes se desarrollan en la [evidencia cuantitativa](evidencia_cuantitativa.md).
