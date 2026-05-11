# Computacion Cientifica - Consecuencias de la IA en las personas

Repositorio del trabajo de Computacion Cientifica de la Universidad Austral de Chile. El proyecto analiza, de forma exploratoria y descriptiva, distintas consecuencias sociales, educativas, laborales e informacionales asociadas al uso de inteligencia artificial.

El trabajo usa datasets abiertos, notebooks en Python, graficos generados para el informe, articulos de respaldo y un informe final escrito en LaTeX.

## Contenido principal

- `informe_trabajo.tex`: fuente LaTeX del informe final.
- `informe_trabajo.pdf`: version compilada del informe.
- `ppt trabajo.pptx`: presentacion del trabajo.
- `Dataset/`: datasets utilizados en el analisis.
- `cuadernillos/`: notebooks de exploracion y generacion de graficos.
- `Papper/`: papers y documentos consultados como respaldo teorico.

## Ejes del analisis

El informe organiza los resultados en cuatro lineas principales:

- Percepcion social y trabajo.
- Educacion y rendimiento academico.
- Produccion cientifica y desarrollo geografico.
- Seguridad informacional y desinformacion.

## Estructura del repositorio

```text
.
|- README.md
|- informe_trabajo.tex
|- informe_trabajo.pdf
|- ppt trabajo.pptx
|- Dataset/
|  |- Data_set_01.csv
|  |- Data_set_02.csv
|  |- Data_set_03.csv
|  |- Data_set_04.csv
|  |- Data_set_05.csv
|  |- Data_set_07.csv
|- cuadernillos/
|  |- Cuadernillo_dataset_01.ipynb
|  |- Cuadernillo_dataset_02.ipynb
|  |- Cuadernillo_dataset_03.ipynb
|  |- Cuadernillo_dataset_04.ipynb
|  |- Cuadernillo_dataset_05.ipynb
|  |- Cuadernillo_dataset_06.ipynb
|  |- Cuadernillo_dataset_07.ipynb
|- Papper/
```

## Requisitos

- Python 3.10 o superior.
- Jupyter Notebook o JupyterLab.
- Una distribucion de LaTeX para compilar el informe, por ejemplo MiKTeX o TeX Live.

Paquetes de Python usados principalmente:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `geopandas`
- `jupyter`
- `ipykernel`
- `openpyxl`

## Instalacion rapida

En Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install pandas numpy matplotlib seaborn geopandas jupyter ipykernel openpyxl
```

Opcionalmente, registrar el entorno como kernel de Jupyter:

```powershell
python -m ipykernel install --user --name computacion-cientifica --display-name "Python (Computacion Cientifica)"
```

## Uso

Para abrir los notebooks:

```powershell
jupyter notebook
```

Luego revisar los archivos dentro de `cuadernillos/`. Cada cuadernillo corresponde al procesamiento de uno de los datasets o a la generacion de figuras usadas en el informe.

Para compilar el informe:

```powershell
pdflatex informe_trabajo.tex
pdflatex informe_trabajo.tex
```

Se ejecuta dos veces para actualizar correctamente referencias, numeracion de figuras y bibliografia interna.

## Fuentes y respaldo

El analisis se apoya en datasets abiertos y documentos academicos o institucionales, incluyendo trabajos sobre impacto laboral de IA generativa, educacion, riesgos sociales y registros de incidentes asociados a IA. Los documentos de respaldo se encuentran en `Papper/` y las fuentes especificas estan citadas dentro del informe.

## Integrantes

- Martin Arrigo
- Nicolas Toro
- Diego Mora
- Benjamin Neira

---

Curso: Computacion Cientifica - Universidad Austral de Chile
