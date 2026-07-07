# Reporte de resultados - churn de comunicación CRM multicanal

_Generado: 20260519_152108_

## Metodología (síntesis para el paper)

- **D1 Ventana temporal**: observación = 2021-04-23T23:59:29 → 2022-10-23T23:59:29 (18 meses). Evaluación = 2022-10-23T23:59:29 → 2023-04-23T23:59:29 (6 meses). Conteo hacia atrás desde `max(sent_at)`.
- **D2 Atribución de compra**: se usa `is_purchased` del dataset tal como viene (atribución CRM origen).
- **D3 Features**: lista cerrada del paper sec. IV.C. Sin features adicionales no documentadas.
- **D4 Entorno**: Python 3.11.4, stack pandas/sklearn/xgboost/imblearn. SEED = 42 global.
- **D5 Dataset**: `messages.csv.gz` (24 meses, 721M filas en bruto), descargado de `data.rees46.com`. Se procesa por streaming gzip + chunks (sin descomprimir a disco).

## Split y desbalance

- Tamaño train: 7,924,808 | Tamaño test: 1,981,203
- Prevalencia churn en train: 0.5700
- Prevalencia churn en test:  0.5700
- SMOTE aplicado: **False** (threshold = 0.2)

## Resultados comparativos (threshold óptimo F1)

Métricas con intervalo de confianza 95% por bootstrap (200 iteraciones).

| Modelo | AUC-ROC | PR-AUC | F1 | Precision | Recall | Threshold |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.771 [0.771, 0.772] | 0.773 [0.772, 0.774] | 0.791 [0.790, 0.791] | 0.675 [0.674, 0.675] | 0.954 [0.954, 0.955] | 0.448 |
| Random Forest | 0.807 [0.806, 0.808] | 0.821 [0.821, 0.822] | 0.804 [0.803, 0.804] | 0.701 [0.700, 0.701] | 0.943 [0.942, 0.943] | 0.416 |
| XGBoost | 0.806 [0.805, 0.806] | 0.820 [0.819, 0.820] | 0.803 [0.803, 0.804] | 0.698 [0.698, 0.699] | 0.946 [0.945, 0.946] | 0.868 |

**Modelo ganador (mejor AUC-ROC):** Random Forest.

## Resultados con threshold por defecto (0.5)

| Modelo | AUC-ROC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression | 0.771 | 0.773 | 0.790 | 0.681 | 0.941 |
| Random Forest | 0.807 | 0.821 | 0.801 | 0.711 | 0.918 |
| XGBoost | 0.806 | 0.820 | 0.778 | 0.639 | 0.994 |

## Figuras

![logreg](outputs/figures/confusion_logreg_20260519_152108.png)

![rf](outputs/figures/confusion_rf_20260519_152108.png)

![rf](outputs/figures/feature_importance_rf_20260519_152108.png)

![xgb](outputs/figures/confusion_xgb_20260519_152108.png)

![xgb](outputs/figures/feature_importance_xgb_20260519_152108.png)

![comparativas](outputs/figures/roc_curves_20260519_152108.png)

![comparativas](outputs/figures/pr_curves_20260519_152108.png)

