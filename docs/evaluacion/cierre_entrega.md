# Inventario de entrega — Integrante 4

La guía pide una entrega que se pueda instalar, ejecutar, evaluar y analizar desde
el README. Este inventario cubre los insumos verificados y las piezas por producir.

| Material | Disponible | Pendiente para la entrega |
| --- | --- | --- |
| Informe consolidado | Guía del integrante 4, informes individuales, informe experimental con tablas contrastadas, borrador del integrante 4 y evidencia técnica del repositorio. | Consolidar simulador, agente y entrenamiento, experimentos e integración. Incluir versiones, SHA-256, metodología, tablas, figuras y límites de reproducibilidad. |
| Video de demostración | Checkpoint Base64 comprobado, comando de Pygame y CSV y manifiestos de referencia. | Grabar la verificación SHA-256, una vuelta en Pygame y la comparación fija y aleatoria. Mostrar el origen del registro de 145 episodios. |
| Presentación de defensa | Diseño del entorno, comparación DQN/aleatorio/heurístico, gráficas reproducibles y resultados de la auditoría. | Preparar diapositivas con diseño, metodología, resultados, demostración y límites. Ensayar la explicación de ambas semillas, los pasos exitosos y la separación entre demostración y lote histórico. |
| Evidencias técnicas y experimentales | Nueve CSV históricos, contratos de los CSV de evaluación, manifiesto, recuperación, pruebas y notebooks. | Reunir la consola anonimizada, el estado Git, las huellas, los CSV y manifiestos de la demostración y las figuras y tablas. Limpiar las salidas de celdas antes de compartir notebooks ejecutados. |
| Paquete reproducible Base64 | `result_model_prueba.zip` está en los materiales de entrega. El checkpoint y el registro del ZIP coinciden con los archivos locales originales. | Adjuntar el ZIP fuera de Git por el canal que apruebe el equipo. Incluir el README, las dos huellas y la estructura indicada abajo. |

## Qué acredita cada evidencia

- **Reproducción real.** Diez episodios de inicio fijo y diez de inicio aleatorio
  con `best_model.pt` del entrenamiento Base64 conocido. Semillas de evaluación
  42 a 51. Su registro de 145 episodios acredita ese entrenamiento.
- **Compatibilidad técnica.** Base32 se comprobó con pesos sintéticos. El
  entrenamiento diagnóstico verifica el flujo de guardado y recarga.
- **Auditoría histórica.** Los 240 registros se contrastaron matemáticamente con
  los resúmenes y el informe. Incluyen 120 episodios DQN y 120 de referencia.
  Los seis checkpoints entrenados del lote original siguen sin estar disponibles.

## Estructura de distribución

Distribuye el ZIP por separado, sin modificar su contenido. El repositorio
contiene el código y los CSV históricos. La ruta local queda así tras la extracción:

```text
results_demo_base64/
  best_model.pt
  training_log.csv
```

El ZIP original incluye también `final_model.pt`, `training_curves.png` y
`evaluation_summary.csv`. La demostración usa solo el mejor checkpoint y el
registro. La exclusión de Git cubre todo `results_demo_base64/`.

| Insumo de demostración | SHA-256 |
| --- | --- |
| `best_model.pt` | `8fca3c72f147cd94af182417e55e4093ed0f6ff8eaf3448e654650d51c04f5b1` |
| `training_log.csv` | `2dd65a595cc5378e6fbad9007501ce5731656849f8cf92cb90303962ce187ab8` |

## Validación de este cierre

Las pruebas se ejecutaron sobre los cambios locales de `feature/integracion-final`
con base `71abfbd`, usando PyTorch 2.15.0+cpu.

| Comprobación | Resultado |
| --- | --- |
| Suite completa | 69/69 en Python 3.11.9 y 69/69 en Python 3.12.14. Se conservaron las 64 pruebas previas y se añadieron cinco regresiones. |
| Diagnósticos reales | Dos corridas de un episodio y dos pasos, con destino automático y explícito. Se recargaron ambos checkpoints finales. Reutilizar el destino explícito fue rechazado antes de arrancar y conservó sus seis archivos byte por byte. |
| Actualización de best_model.pt | Una prueba simulada verificó dos guardados del mejor checkpoint dentro de la misma corrida. |
| Demostración Base64 | Diez filas fijas y diez aleatorias idénticas al histórico. Retornos medios 161.682435 y 161.819313. Ambas corridas tuvieron 10/10 éxitos y cero colisiones. Pasaron los dos formatos CSV, ambos JSONL y la recuperación. |
| README y notebooks | Las guardas detectaron el ZIP y los insumos ausentes. Se validaron ambos SHA-256 al extraer el ZIP. El experimental funcionó sin un log. Con el registro de 145 episodios, los dos notebooks generaron siete PNG y cuatro tablas CSV. |
| Demostración visual e integridad | Dos episodios completados con Pygame. Los nueve CSV históricos, los dos checkpoints y el registro de entrenamiento conservaron sus SHA-256. |

## Checklist final

- [x] Verificar el rechazo de destinos de entrenamiento existentes y los
  guardados sucesivos de best_model.pt durante una corrida.
- [x] Pasar la suite y recargar los checkpoints de los diagnósticos.
- [x] Reproducir las veinte filas Base64, verificar los manifiestos y comprobar
  la integridad de los datos originales.
- [x] Probar las instrucciones de extracción, notebooks y demostración visual.
- [ ] Revisar el diff de protección de entrenamiento y las pruebas de regresión.
- [ ] Acordar cómo adjuntar el ZIP de demostración fuera de Git y verificar los
  dos SHA-256 desde el paquete que recibirá quien evalúe la entrega.
- [ ] Consolidar el informe y anexar la evidencia con su procedencia.
- [ ] Grabar el video y preparar la presentación.
- [ ] Revisar que el material distribuido esté libre de rutas personales y que
  la limitación de los seis checkpoints quede explícita.
