# Computacion Cientifica - Consecuencias de la IA en las personas

Repositorio del trabajo de Computacion Cientifica de la Universidad Austral de Chile. El proyecto analiza, de forma exploratoria y descriptiva, distintas consecuencias sociales, educativas, laborales e informacionales asociadas al uso de inteligencia artificial.

El trabajo usa datasets abiertos, notebooks en Python, graficos generados y una presentacion (PPT).

## 🚀 Cómo ejecutar el Dashboard

El **Dashboard Interactivo (`Cuadernillo_dashboard_interactivo_v23_mapas.ipynb`)** es el archivo central de esta entrega. Para visualizarlo y usar sus herramientas de Machine Learning e interactividad, siga estos pasos:

### Paso 1: Verificar la estructura de archivos
Asegúrese de haber descomprimido todo el `.zip` y de que la estructura se vea exactamente así:
```text
Entrega_Final/
├── cuadernillos/
│   └── Cuadernillo_dashboard_interactivo_v23_mapas.ipynb
├── Dataset/
│   ├── Data_set_01.csv
│   ├── Data_set_02.csv
│   ├── Data_set_03.csv
│   ├── Data_set_05.csv
│   └── Data_set_07.csv
└── patentes_ia_2018_2024.csv
```
*(Nota: El archivo de patentes debe ir suelto en la carpeta principal, NO dentro de Dataset).*

### Paso 2: Instalar los requisitos
Abra una terminal (o PowerShell) en la carpeta principal del proyecto y asegúrese de tener instaladas las siguientes librerías de Python:
```powershell
pip install pandas numpy matplotlib seaborn geopandas scikit-learn ipywidgets jupyter
```

### Paso 3: Abrir Jupyter Notebook
En la misma terminal, inicie Jupyter:
```powershell
jupyter notebook
```
Esto abrirá una pestaña en su navegador web.

### Paso 4: Ejecutar el Dashboard
1. Navegue hasta la carpeta `cuadernillos/` y haga clic en **`Cuadernillo_dashboard_interactivo_v23_mapas.ipynb`**.
2. **Importante:** Asegúrese de estar conectado a Internet (el cuaderno necesita descargar las fronteras geométricas del mapa mundi desde *Natural Earth*).
3. En el menú superior de Jupyter, haga clic en la pestaña **"Kernel"** y luego en **"Restart & Run All"** (Reiniciar y Ejecutar Todo).
4. Desplácese hasta el final del cuaderno. Verá aparecer la interfaz gráfica completa con múltiples pestañas (Gráficos, General, Mapas, ML V1 y ML V2).

---

## Contenido principal

- `ppt trabajo.pptx`: presentacion del trabajo.
- `Dataset/`: datasets utilizados en el analisis.
- `cuadernillos/`: notebooks de exploracion y generacion de graficos.

## Instalacion rapida (Entorno Virtual)

En Windows PowerShell, si prefiere crear un entorno aislado:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install pandas numpy matplotlib seaborn geopandas scikit-learn ipywidgets jupyter ipykernel openpyxl
```

## Integrantes

- Martin Arrigo
- Nicolas Toro
- Diego Mora
- Benjamin Neira

---

Curso: Computacion Cientifica - Universidad Austral de Chile
