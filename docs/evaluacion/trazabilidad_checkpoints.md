# Trazabilidad de checkpoints DQN

Auditoría del 24 de noviembre de 2026. Los nombres de directorios se muestran
como referencias relativas. Los seis modelos del experimento de 120 episodios
DQN siguen pendientes de identificación.

| Procedencia | Arquitectura | Semilla | Checkpoint y ubicación | SHA-256 | Configuración | Registro de entrenamiento | CSV de evaluación asociado | Estado |
|---|---|---:|---|---|---|---|---|---|
| Entrenamiento base del integrante 2 | Base64 | 42 | `result_model_prueba/best_model.pt`, entregado fuera de Git | `8fca3c72f147cd94af182417e55e4093ed0f6ff8eaf3448e654650d51c04f5b1` | 9→64→64→9, γ 0.99, LR 0.0003, decaimiento ε 45000, 45552 pasos guardados | `result_model_prueba/training_log.csv`, 145 episodios | `evaluation_summary.csv` del paquete y [evaluación fija del módulo 4](../../results/evaluacion_best_fija_py311_20261123_01.csv) | APROBADO. Identidad y carga real verificadas. |
| Entrenamiento base del integrante 2 | Base64 | 42 | `result_model_prueba/final_model.pt`, entregado fuera de Git | `a6aae0e6395290c273f387026509381cc3632d7091ba45c9b3d5994108b3f8b2` | 9→64→64→9, γ 0.99, LR 0.0003, decaimiento ε 45000, 45552 pasos guardados | Mismo paquete de entrenamiento base | Sin CSV individual atribuido a este archivo | APROBADO CON OBSERVACIONES. Metadatos y hash inspeccionados, desempeño pendiente. |
| Experimento del integrante 3 | Base32 | 42 | Desconocido | No disponible | Ancho y semilla declarados en CSV | No disponible | `csv_results/evaluation_results.csv`: Base32/42 | BLOQUEADO |
| Experimento del integrante 3 | Base32 | 123 | Desconocido | No disponible | Ancho y semilla declarados en CSV | No disponible | mismo CSV: Base32/123 | BLOQUEADO |
| Experimento del integrante 3 | Base32 | 999 | Desconocido | No disponible | Ancho y semilla declarados en CSV | No disponible | mismo CSV: Base32/999 | BLOQUEADO |
| Experimento del integrante 3 | Base64 | 42 | Desconocido | No disponible | Ancho y semilla declarados en CSV | No disponible | mismo CSV: Base64/42 | BLOQUEADO |
| Experimento del integrante 3 | Base64 | 123 | Desconocido | No disponible | Ancho y semilla declarados en CSV | No disponible | mismo CSV: Base64/123 | BLOQUEADO |
| Experimento del integrante 3 | Base64 | 999 | Desconocido | No disponible | Ancho y semilla declarados en CSV | No disponible | mismo CSV: Base64/999 | BLOQUEADO |

Las cuatro ramas inspeccionadas carecen de checkpoints experimentales publicados.
La carpeta de entrega disponible contiene dos archivos DQN y el ZIP repite
sus hashes. Las etiquetas del CSV experimental no contienen hash, nombre de
checkpoint, configuración completa ni versión del código. El archivo base
guarda `TrainConfig`, estados de redes y optimizador y `total_steps`. En los
metadatos inspeccionados faltan el commit del entrenamiento, la versión de
las dependencias y `TrackConfig` y `CarConfig` efectivos.

## Diferencia Base64 con semilla 42

| Evidencia de inicio fijo | Retorno medio | Pasos medios | Éxito |
|---|---:|---:|---:|
| Checkpoint base del integrante 2 evaluado por el integrante 4 | 161.6824351951651 | 326 | 100 % |
| Filas publicadas del integrante 3 | 159.61080719454924 | 294 | 100 % |

La diferencia es de aproximadamente **2.071628 de retorno y 32 pasos**.
El informe del integrante 2, página 4, describe su entrenamiento base
con semilla 42 y su archivo `evaluation_summary.csv` coincide con nuestro
retorno fijo dentro del redondeo. Las dos ramas publicadas usan los mismos
blobs de `car_env.py`, `simulator_core.py` y `config.py`. La rama del agente
permite cambiar ancho, gamma, decaimiento de ε y tasa de aprendizaje, y
`train.py` permite variar la pista. Eso demuestra que existen parámetros
capaces de diferenciar ejecuciones, pero no establece cuáles se usaron.

El inicio fijo del simulador inspeccionado no consume variaciones del
generador de arranque. Cambiar únicamente `--seed` en ese modo no explica
las diferencias observadas con el mismo código. Quedan sin determinar
el checkpoint exacto, mejor frente a final, los parámetros de entrenamiento
y evaluación y los commits y versiones efectivos. Coincidir en arquitectura
y semilla no identifica un entrenamiento.

## Alcance de la entrega final

El integrante 4 continúa la integración con los artefactos disponibles.
La búsqueda de los seis checkpoints está cerrada para esta entrega. Los
240 registros quedan como evidencia histórica auditada matemáticamente.
El Base64 conocido se usará para la demostración funcional y conserva su
propia trazabilidad. El informe separará ambos alcances.
