# Preparación y evaluación de trayectorias marítimas AIS

## 1. Propósito de la investigación

El proyecto prepara observaciones AIS reales para estudiar la **precisión de la geolocalización predictiva de buques logísticos** y, en una fase posterior, comparar modelos de aprendizaje profundo —por ejemplo, LSTM y GRU— con las mismas rutas de prueba.

La pregunta estadística de interés debe plantearse como una comparación de errores de predicción sobre escenarios no vistos: ¿un modelo produce errores geográficos menores que un baseline sencillo? El repositorio prepara los datos y ahora incluye un experimento reproducible con LSTM contra persistencia. Aún no hay una implementación GRU; sus campos permanecen vacíos.

## 2. Fuente e instrumento de recolección

### Fuente primaria

Se utiliza el archivo diario comprimido del conjunto Nationwide AIS de NOAA Office for Coastal Management (OCM), obtenido del contenedor oficial:

- Índice de archivos 2025: <https://noaaocm.blob.core.windows.net/ais/csv2/csv2025/index.html>
- Patrón de descarga: `https://noaaocm.blob.core.windows.net/ais/csv2/csv{año}/ais-{año}-{mes}-{día}.csv.zst`
- Ejemplo utilizado para la ejecución reproducible: `ais-2025-05-01.csv.zst`

### Instrumento y unidad de observación

AIS (Automatic Identification System) es un sistema de identificación y reporte de movimiento marítimo. El registro analizado es una **transmisión/observación AIS asociada a un MMSI y una hora**, con posición y datos de movimiento. NOAA OCM distribuye los datos agregados como archivos tabulares diarios. Este proyecto no instala receptores ni recolecta señales AIS directamente; usa el producto publicado por NOAA. El archivo no documenta por sí solo todos los aspectos del receptor, la cobertura de estaciones ni la tasa de pérdida de paquetes, por lo que no se infieren esas características.

Las columnas observadas en el archivo NOAA del 1 de mayo de 2025 fueron: `mmsi`, `base_date_time`, `longitude`, `latitude`, `sog`, `cog`, `heading`, `vessel_name`, `imo`, `call_sign`, `vessel_type`, `status`, `length`, `width`, `draft`, `cargo` y `transceiver`. El pipeline conserva para la trayectoria las variables de navegación `mmsi`, `timestamp`, `LAT`, `LON`, `SOG` y `COG`; los demás campos se perfilan en el archivo fuente pero no se incluyen en el Parquet limpio actual.

### Alcance temporal de los datos ejecutados

La corrida documentada utiliza un solo día, 2025-05-01, en UTC, desde `00:00:00` hasta `23:59:59`. Este corte permite probar el flujo y desarrollar el análisis, pero **no representa distintas estaciones, días, zonas oceánicas ni condiciones operacionales**. Para sostener generalización se deberá ampliar la descarga a varios días y periodos, conservando los identificadores de fecha y la procedencia de cada archivo.

## 3. Flujo metodológico reproducible

El proceso se ejecuta por etapas para conservar la trazabilidad desde el archivo NOAA hasta las tablas de análisis:

1. **Adquisición:** descarga el archivo diario oficial y lo guarda comprimido en `data/raw/`.
2. **Perfilado:** registra filas, columnas, tipos, valores ausentes, MMSI distintos y cobertura temporal en `reports/data_profile.json` y `reports/data_profile.md`.
3. **Validación y limpieza:** lee el archivo por bloques para limitar la memoria; normaliza los nombres NOAA; convierte la fecha a UTC; elimina duplicados globalmente mediante la clave MMSI + instante; valida identificador, fecha, latitud, longitud, SOG y COG; cuenta registros válidos, duplicados y rechazados. No completa coordenadas faltantes ni inventa puntos.
4. **Construcción de trayectorias:** ordena los puntos por MMSI y tiempo, y comienza un segmento nuevo cuando la brecha temporal supera 30 minutos o la distancia geodésica entre puntos excede 15 km. Conserva segmentos con al menos 20 puntos y 30 minutos de duración. La distancia se calcula con Haversine.
5. **Construcción de escenarios:** forma ventanas de 30 observaciones de entrada y 10 observaciones objetivo. Se muestrean hasta 369 trayectorias/escenarios con semilla 42 para una prueba manejable y reproducible; no se generan datos sintéticos.
6. **Partición:** asigna train, validation y test por MMSI. Así, las observaciones del mismo buque no quedan repartidas entre particiones. La prueba final de comparación de modelos debe usar exclusivamente el archivo de test.
7. **Exportación y control:** escribe tablas CSV y Parquet, diccionario, resumen de preprocesamiento y controles automáticos de calidad.

### Reglas y decisiones de limpieza

- `mmsi`: debe ser un valor numérico de nueve dígitos distinto de cero. Es una comprobación de formato, no una verificación contra un registro internacional de asignación MMSI.
- `timestamp`: fecha válida, normalizada a UTC.
- `LAT` / `LON`: latitud dentro de $[-90, 90]$ y longitud dentro de $[-180, 180]$.
- `SOG`: velocidad sobre el fondo en nudos; los valores ausentes, centinela o fuera del rango admitido se excluyen.
- `COG`: rumbo sobre el fondo en grados; ausentes, centinelas y valores fuera de rango se excluyen.
- Duplicados: se consideran duplicados las filas que comparten MMSI e instante normalizado. Si hay varios mensajes de un MMSI en el mismo instante con posiciones diferentes, la regla conserva uno; para estudios donde esa situación sea material se deberá revisar una regla de resolución específica.
- Las categorías de rechazo del reporte se contabilizan secuencialmente: cuando una fila incumple varias reglas, se registra en la primera regla fallida. El total de duplicados se informa aparte.

## 4. Resultados de validación del día de prueba

El reporte `reports/preprocessing_report.json` registra los siguientes conteos para el archivo NOAA descargado:

| Control                                |         Registros |
| -------------------------------------- | ----------------: |
| Filas originales                       |         9.181.002 |
| Duplicados MMSI + timestamp            |             4.277 |
| Rechazados por SOG                     |            16.498 |
| Rechazados por COG                     |         1.576.676 |
| MMSI, timestamp o coordenada inválidos | 0 en esta corrida |
| Registros limpios retenidos            |         7.583.551 |
| Retención sobre filas originales       |           82,60 % |

El gran número de registros sin COG utilizable es una decisión metodológica importante: las trayectorias finales corresponden a casos completos de navegación para las variables seleccionadas. La exclusión podría sesgar la muestra hacia determinados tipos de buques, equipos o situaciones de navegación. Antes del análisis de tesis se recomienda informar este patrón y considerar un análisis de sensibilidad que no exija COG como predictor, si el diseño lo permite.

En la muestra final generada hay 369 escenarios, 368 MMSI y 14.760 filas de observaciones largas (40 observaciones por escenario: 30 de entrada + 10 objetivo). Los controles de calidad verifican ausencia de celdas faltantes en las observaciones exportadas, secuencias cronológicas, objetivo posterior a entrada, límites de tiempo/distancia y ausencia de solapamiento de MMSI entre las particiones. Consulte `data/final/dataset_quality.json` para los valores de la ejecución vigente.

## 5. Archivos que se utilizarán

### Datos AIS y tablas para SPSS

| Archivo                                                         | Uso recomendado                              | Unidad por fila / contenido                                                                                                                                                                       |
| --------------------------------------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data/raw/2025-05-01.csv.zst`                                   | Fuente primaria conservada                   | Mensajes AIS NOAA originales comprimidos; no es la tabla analítica limpia.                                                                                                                        |
| `data/intermediate/cleaned.parquet`                             | Punto de partida limpio                      | Un punto AIS válido por fila, con MMSI, timestamp UTC, LAT, LON, SOG y COG. Parquet es eficiente para procesamiento; CSV limpios pueden exportarse si se necesita inspección externa.             |
| `data/trajectories/trajectory_points.parquet`                   | Reconstrucción y consulta de rutas           | Puntos AIS válidos con `trajectory_id`; usar para recrear secuencias.                                                                                                                             |
| `data/trajectories/trajectory_metadata.parquet`                 | Descripción de rutas                         | MMSI, inicio/fin, duración, cantidad de puntos, extremos geográficos, distancia y estadísticas de SOG/COG.                                                                                        |
| `data/scenarios/scenarios.parquet`                              | Índices de ventanas                          | IDs, índices de inicio/fin de entrada/objetivo y horizonte; referencia los puntos, no reemplaza la tabla de observaciones.                                                                        |
| `data/final/observations_long.csv`                              | Análisis descriptivo o longitudinal en SPSS  | Una fila por punto en los escenarios seleccionados; variables AIS más tiempo transcurrido, distancia, SOG km/h y componentes circulares del COG. Hay medidas repetidas dentro de buque/escenario. |
| `data/final/vessel_summary.csv`                                 | Resúmenes por unidad independiente           | Una fila por MMSI representado; incluye resúmenes de navegación y diferencia descriptiva de SOG entrada-objetivo. No es una métrica de precisión del modelo.                                      |
| `data/final/scenario_metadata.csv`                              | Llave y contexto temporal                    | Una fila por escenario, con MMSI, partición e intervalos de entrada y objetivo.                                                                                                                   |
| `data/final/spss_geolocation_study.csv`                         | **Tabla principal para SPSS y modelos**      | Una fila por escenario; entrada, posiciones objetivo, baseline y métricas LSTM en test.                                                                                                           |
| `data/final/lstm_vs_persistence_test_metrics.csv`               | Comparación inferencial                      | ADE, FDE y peor error por escenario para LSTM frente a persistencia, solo test.                                                                                                                   |
| `data/final/lstm_test_point_predictions.csv`                    | Auditoría de predicciones                    | Posición observada, LSTM, persistencia y error por cada punto futuro del test.                                                                                                                    |
| `data/final/observations_long.csv`                              | Análisis longitudinal/descriptivo            | Observaciones por escenario con variables AIS y dependencias temporales explícitas.                                                                                                               |
| `data/final/vessel_summary.csv`                                 | Descripción por buque                        | Resúmenes navegacionales; no usar como métrica de rendimiento de modelo.                                                                                                                          |
| `data/final/scenario_metadata.csv`                              | Trazabilidad de escenarios                   | MMSI, partición, trayectoria e intervalos temporales.                                                                                                                                             |
| `data/final/data_dictionary.csv`                                | Diccionario SPSS                             | Descripción, nivel de medición y unidades de las variables.                                                                                                                                       |
| `data/final/dataset_quality.json`                               | Auditoría de calidad                         | Validaciones de secuencia, rangos, huecos, fuga de MMSI y exportación.                                                                                                                            |
| `data/final/statistical_guidance.md`                            | Guía de interpretación                       | Unidad de análisis, dependencia, COG circular y cautelas inferenciales.                                                                                                                           |
| `data/final/ais_procesado_spss.csv`                             | Tabla pre/post exacta para SPSS              | Una fila por MMSI test; 12 variables, ADE/FDE en metros para persistencia, GRU y LSTM, con diferencias Pre − Post.                                                                                |
| `reports/lstm_test_evaluation.json`                             | Informe del experimento                      | Configuración, entrenamiento, métricas, pruebas pareadas y limitaciones.                                                                                                                          |
| `reports/lstm_training_history.csv`                             | Diagnóstico del ajuste                       | Pérdida de train y ADE de validation por época.                                                                                                                                                   |
| `reports/lstm_geolocation_model.pt`                             | Modelo entrenado                             | Pesos y escaladores para reproducir inferencia.                                                                                                                                                   |
| `data/final/data_dictionary.csv`                                | Etiquetado de variables en SPSS              | Archivo, nombre de variable, descripción, nivel de medición sugerido y unidad.                                                                                                                    |
| `data/final/dataset_quality.json`                               | Auditoría automatizada del dataset           | Conteos, rangos, secuencia, tamaños y controles de fuga entre MMSI.                                                                                                                               |
| `reports/data_profile.md` y `reports/preprocessing_report.json` | Perfil del origen y trazabilidad de limpieza | Esquema/missingness del archivo fuente y conteos de limpieza.                                                                                                                                     |

Los CSV son los más prácticos para importar en SPSS. Los archivos Parquet se conservan para análisis reproducibles y procesamiento eficiente. En SPSS, `MMSI`, `ID_Escenario_Ruta`, `ID_Trayectoria`, `Particion`, `split`, `scenario_id` y `sequence_role` son identificadores/categorías; no se deben analizar como variables numéricas continuas.

## 6. Variables relevantes para el objetivo de tesis

### A. Variables predictoras AIS

1. **Posición de entrada:** `LAT` y `LON` en cada uno de los 30 pasos (`input_01_lat`, `input_01_lon`, …, `input_30_lat`, `input_30_lon`). Son las variables centrales para aprender movimiento espacial.
2. **Velocidad:** `SOG` (nudos) y `sog_kmh` (km/h). Escoger una unidad y mantenerla igual entre modelos.
3. **Dirección:** `COG`, de 0 a menos de 360 grados. Es circular: 359° está cerca de 1°, no lejos. En `observations_long.csv` están `cog_sin` y `cog_cos`; los modelos deben usar esos componentes o una codificación circular consistente en vez de tratar grados como escala lineal.
4. **Tiempo y muestreo:** `timestamp_utc`, `step` y `delta_time_seconds`. Las transmisiones AIS son irregulares; el paso 30 no necesariamente equivale al mismo tiempo transcurrido en todos los casos. Comparar horizontes por cantidad de puntos puede significar horizontes temporales distintos.
5. **Movimiento derivado:** `distance_from_previous_km`, calculado entre puntos consecutivos, útil para describir movimiento y revisar saltos. No es una variable independiente de LAT/LON y no representa por sí sola la distancia total de una ruta.

### B. Variable objetivo de geolocalización

Para cada uno de los 10 pasos futuros, las coordenadas observadas (`target_01_lat`, `target_01_lon`, …, `target_10_lat`, `target_10_lon`) son el **ground truth AIS** contra el que se comparan las coordenadas predichas por el modelo. No se debe medir precisión restando grados de latitud/longitud sin transformación: los grados no equivalen a una distancia fija en km. Calcule para cada paso la distancia geodésica entre punto predicho y punto observado (por ejemplo, Haversine) y resuma a nivel de escenario.

### C. Resultados recomendados para evaluar la precisión predictiva

La variable independiente es el **modelo evaluado** (LSTM, GRU y, como referencia, un baseline de persistencia que mantiene la última posición observada). Las variables dependientes comparan la posición predicha con la posición AIS observada para exactamente los mismos escenarios y tiempos.

Para cada escenario $s$ y punto objetivo $k=1,\dots,10$, calcular primero el error geodésico $e_{s,k}$ en kilómetros mediante Haversine entre la posición predicha y la observada. No restar directamente latitud/longitud en grados.

Propongo estas variables, en orden de prioridad:

1. **ADE — error promedio de trayectoria** (`ADE_LSTM_km`, `ADE_GRU_km`): media de los diez errores geodésicos por escenario. Es el resultado primario: representa la precisión espacial durante todo el pronóstico.
2. **FDE — error del punto final** (`FDE_LSTM_km`, `FDE_GRU_km`): error geodésico en el punto 10; representa la desviación al final del horizonte.
3. **Error máximo por escenario** (`Max_Error_LSTM_km`, `Max_Error_GRU_km`): el mayor de los diez errores. Es una métrica secundaria de peor caso y sensible a valores atípicos. No recomiendo un P95 separado por escenario con solo diez puntos; el P95 puede resumirse sobre errores punto-a-punto agrupados y con intervalos que remuestreen buques.
4. **Horizonte temporal observado** (`Horizonte_final_min`, `Span_objetivo_min`): tiempo real transcurrido, ya que diez mensajes AIS no equivalen al mismo número de minutos para cada escenario.
5. **Latencia de inferencia** (`Latencia_LSTM_ms`, `Latencia_GRU_ms`): resultado operativo secundario; medir bajo hardware, batch, precisión numérica, calentamiento y protocolo de repetición iguales. No mide exactitud geográfica.

Para SPSS, importar `spss_geolocation_study.csv`. No se mantienen copias separadas train/validation/test ni plantillas vacías: la columna `split` identifica la partición, y las métricas ya calculadas están en las filas test. GRU no está implementado; sus campos vacíos no son ceros. `lstm_vs_persistence_test_metrics.csv` contiene el subconjunto estrecho para la comparación inferencial.

El CSV SPSS contiene 369 filas (221 train, 74 validation, 74 test) y 218 columnas en la corrida documentada. Los datos AIS y baseline están completos; solo quedan vacíos los resultados GRU y las métricas LSTM de train/validation. Los campos de tiempo transcurrido son numéricos en segundos. Al importarlo, seleccionar coma como separador, primera fila como nombres, punto decimal y tratar MMSI/IDs/partición como nominales.

La precisión de geolocalización no equivale a optimizar una ruta operacional. Para evaluar una ruta optimizada harían falta además longitud/costo respecto a una ruta de referencia factible, tiempo o combustible, cumplimiento de restricciones y exposición a riesgos. Las fuentes actuales no contienen restricciones, costos ni una ruta óptima etiquetada; por tanto, el alcance medible ahora es la predicción de posición.

### D. Factores de agrupación y covariables

- `Modelo`: categoría LSTM/GRU/baseline de persistencia; las métricas quedan en columnas pareadas para el mismo escenario.
- `Particion`: train/validation/test. Son particiones técnicas, **no tratamientos experimentales**.
- `MMSI`, `ID_Trayectoria`, `ID_Escenario_Ruta`: identificadores para agrupación, emparejamiento y control de dependencia.
- Contexto sugerido para análisis de sensibilidad: SOG media, intervalos entre transmisiones, longitud/duración de trayectoria, latitud/región, tipo/clase de buque y estación/fecha. Algunos atributos existen en el CSV original pero no están en el Parquet limpio actual; no deben presentarse como covariables disponibles en los CSV finales sin ampliar la retención y validación del pipeline.

## 7. Plan de análisis estadístico recomendado

1. Entrene y ajuste hiperparámetros usando solo train y validation. Incluya el baseline de persistencia; no seleccione modelos basándose en test.
2. Para cada modelo, prediga exactamente los mismos escenarios test y guarde predicciones punto-a-punto junto a `scenario_id`, instante, LAT/LON observado y predicho. Los resultados de LSTM ya están en `lstm_test_point_predictions.csv` y `lstm_vs_persistence_test_metrics.csv`.
3. Calcule ADE/FDE/error máximo usando las mismas fórmulas en todos los modelos. Compare el ADE emparejado por escenario y reporte media/mediana, IC 95 %, tamaño de efecto y p-valor.
4. Evalúe normalidad sobre las **diferencias emparejadas de ADE**, no sobre todas las filas AIS. Use Q-Q y Shapiro-Wilk como diagnóstico. Si la normalidad es plausible, t pareada; si no, Wilcoxon o bootstrap agrupado por MMSI.
5. Declare ADE como métrica primaria y FDE/error máximo como secundarios. Ajuste multiplicidad (por ejemplo, Holm) si contrasta varias métricas/horizontes. Si hay varios escenarios por MMSI, agregue por buque o modele explícitamente esa dependencia.
6. Interprete la diferencia de desempeño predictivo de modelos como comparación experimental sobre casos compartidos. Sin asignación aleatoria de una intervención sobre buques, no es correcto atribuir causalidad general al “efecto de implementación” fuera de la muestra evaluada.

`vessel_summary.csv` se reserva para descripción y covariables AIS. Sus diferencias de velocidad no evalúan ni seleccionan modelos y no deben usarse para responder la hipótesis principal.

### Experimento GRU/LSTM pre/post ejecutado

El comando `run-spss-dl` entrena ambos modelos con un máximo de 60 épocas y parada temprana según ADE de validación, y evalúa MMSI reales de test. La referencia Pre es persistencia (mantener la última posición AIS). En la corrida más reciente: ADE media Pre 776,64 m, GRU 626,31 m y LSTM 571,05 m. Sin embargo, solo 24 buques mejoraron con GRU y 23 con LSTM; las medianas de diferencias Pre − Post fueron negativas. Las diferencias no fueron normales y Wilcoxon no encontró diferencias significativas en ADE (GRU $p=0,848$, LSTM $p=0,968$). Las medias menores no demuestran mejora general; el resultado de una fecha y muestra pequeña es exploratorio. El script imprime métricas de cada ejecución y nunca fuerza mejoras.

El resultado se exporta a `data/final/ais_procesado_spss.csv` con las 12 columnas especificadas, una fila por MMSI y ADE/FDE en metros. `Dif = Pre - Post`: valores positivos significan menor error posterior; negativos significan que el modelo empeoró. No se generan filas ni ganancias sintéticas.

## 8. Limitaciones para optimización de rutas

Los archivos actuales representan trayectorias AIS observadas y ventanas para predicción de movimiento. No incluyen en su esquema limpio actual meteorología, corrientes, batimetría, zonas restringidas, puertos de origen/destino, consumo de combustible ni una función de costo. Por ello permiten preparar y evaluar **predicción de geolocalización**, pero no bastan por sí solos para recomendar rutas óptimas operacionales. La optimización requerirá sumar esas fuentes, definir restricciones y costos, y validar seguridad y factibilidad marítima.

Además, una sola fecha y 369 escenarios son insuficientes para afirmar robustez anual o poblacional. La partición por MMSI limita fuga entre train/validation/test, pero al compartir fecha y dominio geográfico del archivo no sustituye una validación temporal y geográfica externa. El test actual tiene 74 escenarios y 74 MMSI; amplíe a varios días/meses y, de ser posible, retenga periodos o regiones no usados durante el entrenamiento.

## 9. Ejecución del pipeline

Desde la raíz del repositorio, en Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
./install
./test
./download --date 2025-05-01
./profile --input data/raw/2025-05-01.csv.zst
./preprocess --input data/raw/2025-05-01.csv.zst --output data/intermediate/cleaned.parquet
./build-trajectories --input data/intermediate/cleaned.parquet
./build-scenarios --input data/trajectories/trajectory_metadata.parquet --points data/trajectories/trajectory_points.parquet
./build-dataset --scenarios data/scenarios/scenarios.parquet --points data/trajectories/trajectory_points.parquet
./train-model
./run-spss-dl --reuse-derived
```

La descarga diaria ronda cientos de MB comprimidos y el procesamiento puede tardar. `./run-spss-dl` procesa desde `--raw` por defecto; usa `--reuse-derived` si los Parquet limpios y segmentados ya están disponibles. Exporta el CSV final a `data/final/ais_procesado_spss.csv`. Las diferencias negativas son resultados posibles y deben conservarse.

## 10. Reproducibilidad y pruebas

Los parámetros principales están en `.env.example` y `src/config/settings.py`: umbrales de segmentación, mínimos de trayectoria, longitudes de entrada/horizonte, cantidad de escenarios y semilla aleatoria. La descarga y limpieza no generan observaciones sintéticas. Ejecuta `./test` para las pruebas unitarias y end-to-end con datos controlados; estas pruebas validan la lógica del pipeline, no reemplazan la validación del dataset NOAA ni una evaluación de modelos.

Los reportes y metadatos describen la ejecución actual. Al cambiar fechas, umbrales o semilla, vuelve a ejecutar las etapas posteriores y conserva una copia de los reportes y archivos de configuración asociados a cada dataset.
