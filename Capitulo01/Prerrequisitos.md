# Prerrequisitos del equipo

Este archivo es para quien **prepara** el equipo antes del taller (tecnología o facilitador).

Quien hace el taller no instala programas. Solo abre Visual Studio Code y sigue el [taller](README.md).

Al terminar esta guía, el equipo queda así:

- Windows Server con Python 3.10.12 y Visual Studio Code.
- Un entorno de Python en `C:\Taller\entorno`, con las librerías del taller.
- Visual Studio Code abre los archivos `.py` con ese Python.
- Hay salida a internet hacia Hugging Face y PyTorch (se usa durante el taller, no ahora).

## Equipo mínimo

| Recurso | Mínimo | Recomendado |
|---------|--------|-------------|
| Sistema | Windows Server 2019 o posterior | Windows Server 2022 |
| CPU | 4 núcleos | 8 núcleos |
| RAM | 16 GB | 32 GB |
| Disco libre | 20 GB | 30 GB |
| Internet | 10 Mbps | 50 Mbps |

La tarjeta de video no es necesaria. El taller corre en CPU.

## Sitios que deben estar permitidos

Durante el taller el participante descarga dos modelos de lenguaje. El firewall y el proxy del servidor deben permitir:

- `https://www.python.org`
- `https://code.visualstudio.com`
- `https://pypi.org`
- `https://files.pythonhosted.org`
- `https://download.pytorch.org`
- `https://huggingface.co`
- `https://cdn-lfs.huggingface.co`

## 1. Instalar Python 3.10.12

1. Descargue el instalador de 64 bits:

   [https://www.python.org/ftp/python/3.10.12/python-3.10.12-amd64.exe](https://www.python.org/ftp/python/3.10.12/python-3.10.12-amd64.exe)

2. Ejecútelo como administrador.
3. En la primera pantalla marque **Add python.exe to PATH**.
4. Elija **Customize installation**.
5. Deje marcadas las opciones por defecto y pulse **Next**.
6. Marque **Install Python 3.10 for all users**.
7. En **Customize install location** escriba:

   ```text
   C:\Program Files\Python310
   ```

8. Pulse **Install**.
9. Al final, si aparece **Disable path length limit**, púlselo. Los modelos usan rutas largas y en Windows Server fallan si ese límite sigue activo.

Compruebe en PowerShell:

```powershell
py -3.10 --version
```

Debe mostrar `Python 3.10.12`.

## 2. Instalar Visual Studio Code

1. Descargue el instalador de sistema (System Installer) de 64 bits desde [https://code.visualstudio.com/download](https://code.visualstudio.com/download).
2. Ejecútelo como administrador.
3. Marque **Add to PATH** y **Register Code as an editor for supported file types**.
4. Termine la instalación.

## 3. Instalar la extensión de Python

1. Abra Visual Studio Code.
2. En la barra izquierda pulse **Extensiones** (icono de cuadrados).
3. Busque **Python**.
4. Instale la extensión publicada por **Microsoft** (identificador `ms-python.python`).

No hace falta instalar Jupyter. El taller se ejecuta en la terminal de Visual Studio Code.

## 4. Permitir el entorno de Python en PowerShell

PowerShell, por defecto, puede bloquear la activación del entorno. En una ventana de PowerShell **como administrador**:

```powershell
Set-ExecutionPolicy -Scope LocalMachine -ExecutionPolicy RemoteSigned
```

Confirme con `S`.

## 5. Crear la carpeta del taller y el entorno

En PowerShell:

```powershell
New-Item -ItemType Directory -Force -Path "C:\Taller" | Out-Null
py -3.10 -m venv C:\Taller\entorno
```

Active el entorno:

```powershell
C:\Taller\entorno\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

El texto de la línea debe empezar por `(entorno)`.

## 6. Instalar las librerías

Siga en la misma ventana, con el entorno activo.

Primero PyTorch para CPU (no instale la versión con CUDA: en un servidor sin tarjeta NVIDIA la instalación es mucho más pesada y no aporta en este taller):

```powershell
pip install torch==2.1.2+cpu --index-url https://download.pytorch.org/whl/cpu
```

Cree el archivo `C:\Taller\requirements.txt` con este contenido exacto:

```text
transformers==4.38.2
sentencepiece==0.2.0
datasets==2.18.0
accelerate==0.27.2
evaluate==0.4.1
rouge-score==0.1.2
PyPDF2==3.0.1
nltk==3.8.1
matplotlib==3.8.3
seaborn==0.13.2
numpy==1.26.4
pandas==2.2.1
scikit-learn==1.4.1
tokenizers==0.15.2
```

Instálelo:

```powershell
pip install -r C:\Taller\requirements.txt
```

La descarga puede tardar varios minutos.

## 7. Dejar ese Python como el de Visual Studio Code

Así el participante no tiene que buscar el intérprete.

1. En Visual Studio Code pulse `Ctrl + Shift + P`.
2. Escriba `Preferences: Open User Settings (JSON)` y ábralo.
3. Agregue esta línea dentro de las llaves (si ya hay otras líneas, separe con coma):

```json
"python.defaultInterpreterPath": "C:\\Taller\\entorno\\Scripts\\python.exe"
```

4. Guarde con `Ctrl + S`.

## 8. Comprobar que el equipo quedó listo

Cierre Visual Studio Code y ábralo de nuevo. Abra la carpeta `C:\Taller` (**Archivo > Abrir carpeta**).

Abra la terminal integrada: menú **Terminal > Nueva terminal**.

Ejecute:

```powershell
python --version
python -c "import torch, transformers, nltk, evaluate, sklearn, matplotlib, PyPDF2; print('Entorno listo'); print(torch.__version__); print(transformers.__version__)"
```

Resultado esperado:

```text
Python 3.10.12
Entorno listo
2.1.2+cpu
4.38.2
```

Si `python --version` no muestra 3.10.12, la terminal no está usando `C:\Taller\entorno`. Repita el paso 7 y abra una terminal nueva.

No descargue los modelos de lenguaje en este momento. Esa descarga forma parte del taller.

## Si algo falla

| Lo que aparece | Qué hacer |
|----------------|-----------|
| `py` no se reconoce | Vuelva a instalar Python y marque **Add python.exe to PATH**. Cierre y abra PowerShell. |
| La activación del entorno está deshabilitada | Repita el paso 4 en PowerShell como administrador. |
| `No matching distribution found` para `torch` | Confirme que el comando usa el índice `https://download.pytorch.org/whl/cpu` y que el proxy deja pasar ese sitio. |
| Error al instalar `sentencepiece` | Instale [Microsoft Visual C++ Redistributable 2015-2022 x64](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist) y repita `pip install -r C:\Taller\requirements.txt`. |
| Rutas demasiado largas al usar el taller | En el instalador de Python pulse **Disable path length limit**, o habilite rutas largas en la directiva del sistema. |
| El participante ve otro Python | Repita el paso 7. El archivo debe ser `C:\Taller\entorno\Scripts\python.exe`. |
