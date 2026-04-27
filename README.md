# Computacion Cientifica - Analisis de impacto de la IA

Repositorio del curso de Computacion Cientifica (UACh) enfocado en un analisis exploratorio sobre las consecuencias de la inteligencia artificial en las personas, usando Python, Jupyter y visualizacion de datos.

## Descripcion

El proyecto integra y analiza datos de distintas fuentes para estudiar cuatro dimensiones principales:

- Salud mental
- Educacion
- Privacidad y seguridad
- Productividad y empleo

El trabajo incluye notebooks de analisis, datasets en formato CSV/Excel, graficos generados y un informe en LaTeX.

## Estructura del repositorio

```text
.
|- README.md
|- prueba.ipynb
|- prueba_02.ipynb
|- informe_trabajo.tex
|- Data/
|  |- Data_set_01.xlsx
|  |- Data_set_02.csv
|  |- Data_set_03.csv
|  |- Data_set_04.csv
|  |- Data_set_05.csv
|  |- tabla2_ordenada_es.csv
|  |- tabla3_ordenada_es.csv
|  |- histograma_sexo_tabla2.png
|  |- histograma_edad_tabla2.png
|  |- grafico_barras_tabla3_rubros.png
|  |- output.png
```

## Requisitos

- Python 3.10 o superior
- Jupyter Notebook / JupyterLab
- Paquetes de Python:
	- pandas
	- numpy
	- matplotlib
	- seaborn
	- scikit-learn
	- jupyter
	- ipykernel
	- openpyxl
	- requests

## Instalacion rapida

### 1) Crear y activar entorno virtual

En Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

En Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

### 2) Instalar dependencias

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter ipykernel openpyxl requests
```

### 3) (Opcional) Registrar kernel

```bash
python -m ipykernel install --user --name computacion-cientifica --display-name "Python (Computacion Cientifica)"
```

## Uso

### Ejecutar notebooks

```bash
jupyter notebook
```

Luego abrir:

- `prueba.ipynb`
- `prueba_02.ipynb`

### Ejecutar crawler de papers de IA

```bash
python crawler_arxiv_ia.py
```

Esto descarga papers recientes relacionados con IA desde la API oficial de arXiv y guarda los resultados en `Data/arxiv_ia/`.

### Ejecutar crawler de indicadores crudos para IA

```bash
python crawler_worldbank_ia.py
```

Esto descarga indicadores crudos del Banco Mundial utiles para hacer graficos sobre adopcion digital, I+D, exportaciones tecnologicas y patentes en `Data/worldbank_ia/`.

### Ejecutar crawler de modelos de IA desde Hugging Face

```bash
python crawler_huggingface_ia.py
```

Esto descarga metadata cruda de modelos de IA reales desde la API oficial de Hugging Face y guarda CSV/JSON en `Data/huggingface_ia/`.

El crawler extrae columnas mas utiles para analisis y graficos, por ejemplo:
- `downloads` y `downloadsAllTime`
- `likes` y `trendingScore`
- `pipeline_tag`, `model_type`, `architecture`
- `parameter_count_billions`
- `used_storage_gb`, `repo_file_count`
- `space_count`, `inference_status`
- `license`, `base_models`, `card_languages`, `arxiv_refs`

### Compilar el informe

Si tienes LaTeX instalado:

```bash
pdflatex informe_trabajo.tex
```

Esto genera el PDF del informe a partir de `informe_trabajo.tex`.

## Fuentes de datos

- Brookings Institution: encuesta sobre uso de IA en EE.UU.
- Kaggle: The Impact of Artificial Intelligence on Society

Los archivos procesados y ordenados utilizados en el analisis se encuentran en la carpeta `Data/`.

## Integrantes

- Martin Arrigo
- Nicolas Toro
- Diego Mora
- Benjamin Neira

---

Curso: Computacion Cientifica - Universidad Austral de Chile
