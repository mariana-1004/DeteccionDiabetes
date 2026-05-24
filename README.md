# Predicción de Diabetes con Machine Learning
 
---
 
## 📁 Estructura del repositorio
 
```
├── Dataset/
│   └── diabetes_binary_5050split_health_indicators_BRFSS2015.csv
├── Paper/
│   └── Original
│        └── BRFSS2015_paper.pdf
├── Codigo/
│   └── preprocesamiento.ipynb
└── README.md
```
 
---
 
## 📊 Dataset
 
**CDC Diabetes Health Indicators — BRFSS 2015**
 
- **Fuente:** Centers for Disease Control and Prevention (CDC)
- **Instancias:** 70,692
- **Target:** `Diabetes_binary` — 0 = No diabetes, 1 = Sí diabetes
- **Balance:** 50% / 50% 
---
 
## 🔬 Primer Avance — Rama `Primer-Avance`
 
Preprocesamiento del dataset:
 
- Carga y exploración de datos
- Separación de features (X) y target (y)
- División train/test (80% / 20%)
