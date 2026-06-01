# Predicción de Diabetes (Clasificación)
 
---
## 📊 Dataset

**CDC Diabetes Health Indicators — BRFSS 2015**

- **Fuente:** Centers for Disease Control and Prevention (CDC) — Behavioral Risk Factor Surveillance System (BRFSS) 2015
- **Instancias:** 70,692
- **Features:** 21
- **Target:** `Diabetes_binary` — 0 = No diabetes, 1 = Sí diabetes
- **Balance:** 50% / 50% ✅

### Descripción de las columnas

| Columna | Descripción | Tipo de dato |
|---|---|---|
| `Diabetes_binary` | **(Target)** Si la persona tiene diabetes (1) o no (0) | Categórico |
| `HighBP` | Tiene presión arterial alta (1) o no (0) | Categórico |
| `HighChol` | Tiene colesterol alto (1) o no (0) | Categórico |
| `CholCheck` | Se ha revisado el colesterol en los últimos 5 años | Categórico |
| `BMI` | Índice de masa corporal | Numérico |
| `Smoker` | Ha fumado al menos 100 cigarros en su vida | Categórico |
| `Stroke` | Ha sufrido un derrame cerebral | Categórico |
| `HeartDiseaseorAttack` | Tiene enfermedad coronaria o ha tenido un infarto | Categórico |
| `PhysActivity` | Ha hecho actividad física en los últimos 30 días | Categórico |
| `Fruits` | Consume fruta al menos una vez al día | Categórico |
| `Veggies` | Consume verduras al menos una vez al día | Categórico |
| `HvyAlcoholConsump` | Consumo alto de alcohol | Categórico |
| `AnyHealthcare` | Tiene algún tipo de cobertura médica | Categórico |
| `NoDocbcCost` | No pudo ver al doctor por costo en el último año | Categórico |
| `GenHlth` | Salud general percibida (escala 1 = excelente a 5 = mala) | Numérico  |
| `MentHlth` | Días de mala salud mental en el último mes | Numérico |
| `PhysHlth` | Días de mala salud física en el último mes  | Numérico |
| `DiffWalk` | Tiene dificultad para caminar o subir escaleras | Categórico |
| `Sex` | Sexo (0 = mujer, 1 = hombre) | Categórico |
| `Age` | Rango de edad en 13 categorías | Numérico (ordinal) |
| `Education` | Nivel educativo (escala 1 a 6) | Numérico  |
| `Income` | Nivel de ingresos  | Numérico  |


[1] A. Teboul, "Diabetes Health Indicators Dataset," Kaggle, 2021. [Online]. Available: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data

[2] Centers for Disease Control and Prevention, "Behavioral Risk Factor Surveillance System Survey Data," U.S. Department of Health and Human Services, Atlanta, GA, 2015.





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
