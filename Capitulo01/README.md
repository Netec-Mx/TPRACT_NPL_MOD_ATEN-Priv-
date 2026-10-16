# Construcción de un asistente para resumir y consultar documentos bancarios

## En pocas palabras

Vas a construir un asistente que lee tres documentos de un banco de ejemplo, escribe un resumen de cada uno y responde preguntas sobre su contenido.

Los documentos son de práctica (no son contratos reales):

- Términos de una tarjeta de crédito.
- Contrato de una cuenta corriente.
- Reglamento de préstamos personales.

Trabajarás en **Visual Studio Code**, en **Windows Server**. No instalas programas. El equipo ya está preparado.

| | |
|---|---|
| **Duración** | 105 minutos |
| **Dónde** | Windows Server y Visual Studio Code |
| **Qué entregas** | Resúmenes, respuestas, un mapa de lo que el asistente “miró” y tus conclusiones |

## Qué vas a lograr

- [ ] Crear los tres documentos de ejemplo.
- [ ] Partir cada documento en trozos que el asistente pueda leer.
- [ ] Pedir dos tipos de resumen y compararlos.
- [ ] Hacer preguntas y ver qué tan seguro está el asistente de cada respuesta.
- [ ] Abrir el mapa que muestra qué palabras usó para responder.
- [ ] Probar el asistente con tus propias palabras y anotar si lo usarías con un documento real.

## Palabras que vas a ver

| Palabra | Significado en este taller |
|---------|----------------------------|
| Resumen que copia frases | Elige las oraciones más importantes y las deja tal cual. |
| Resumen redactado | Escribe un texto nuevo, más corto. En este taller sale en inglés. |
| Confianza | Un número entre 0 y 1. Cerca de 1, el asistente está más seguro. Igual hay que revisar la respuesta en el documento. |
| Mapa de atención | Una imagen. Los cuadros más oscuros son las palabras que el asistente relacionó para contestar. |
| ROUGE | Un puntaje de parecido entre los dos resúmenes. Un número bajo es normal aquí, porque uno está en español y el otro en inglés. |

## Antes de empezar

El equipo debe tener Python y Visual Studio Code listos. Esa preparación está en [Prerrequisitos.md](Prerrequisitos.md) y **no forma parte de este taller**.

Comprueba solo esto:

1. Abre **Visual Studio Code**.
2. Menú **Terminal > Nueva terminal**.
3. Escribe el siguiente comando y presiona **Enter**:

```powershell
python --version
```

Debe aparecer `Python 3.10.12`.

Si aparece un mensaje de que Python no se reconoce, o la versión es otra, detente y pide apoyo con el archivo [Prerrequisitos.md](Prerrequisitos.md).

## Cómo está Visual Studio Code

Vas a usar tres zonas:

1. **Panel izquierdo.** Ahí están las carpetas y los archivos. Si no lo ves, menú **Ver > Explorador**.
2. **Centro.** Ahí pegas cada programa y lo guardas.
3. **Terminal de abajo.** Ahí escribes las órdenes y lees los resultados. Menú **Terminal > Nueva terminal**.

Usa siempre esa terminal de abajo. Debe verse una ruta que termina en `AsistenteBancario` antes de cada orden. Si no es así, escribe:

```powershell
cd C:\Taller\AsistenteBancario
```

Para que las tildes se vean bien, escribe esto una vez en la terminal, al empezar:

```powershell
$env:PYTHONIOENCODING = "utf-8"
chcp 65001 | Out-Null
```

## Cómo pegar un programa

Cada paso te da un programa completo. Haz siempre lo mismo:

1. En el panel izquierdo, clic derecho sobre la carpeta `src`.
2. Elige **Nuevo archivo**.
3. Escribe el nombre exacto que indica el paso (por ejemplo `crear_documentos.py`) y presiona **Enter**.
4. Borra cualquier texto que aparezca solo.
5. Copia **todo** el programa del paso y pégalo.
6. Guarda con **Ctrl + S**.
7. En la terminal escribe la orden del paso y presiona **Enter**.

No hace falta modificar el programa. Pégalo completo.

Cuando un paso dice que puede tardar, espera. La terminal se queda quieta mientras trabaja. No cierres Visual Studio Code.

---

## Paso 1. Crear la carpeta de trabajo (10 min)

### Qué vas a hacer

Crear el lugar donde vivirá el asistente y descargar los dos modelos que va a usar: uno para resumir y otro para responder.

### Crear la carpeta

1. En Visual Studio Code, menú **Archivo > Abrir carpeta**.
2. Entra a `C:\Taller`.
3. Clic derecho en un espacio vacío > **Nueva carpeta**.
4. Nómbrala `AsistenteBancario` y ábrela con **Seleccionar carpeta**.

En la terminal:

```powershell
cd C:\Taller\AsistenteBancario
New-Item -ItemType Directory -Force -Path "data\raw","data\processed","models","outputs\mapas_atencion","src" | Out-Null
```

Si no sale texto en rojo, las carpetas quedaron creadas. En el panel izquierdo debes ver `data`, `models`, `outputs` y `src`.

### Descargar los modelos

Crea el archivo `src\descargar_modelos.py` y pega este programa:

```python
import os
import sys
from pathlib import Path

import nltk

BASE = Path(__file__).resolve().parent.parent
MODELOS = BASE / "models"
NLTK_DIR = MODELOS / "nltk_data"
MODELOS.mkdir(parents=True, exist_ok=True)
NLTK_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(MODELOS)
os.environ["TRANSFORMERS_CACHE"] = str(MODELOS)
os.environ["HF_HUB_CACHE"] = str(MODELOS)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

nltk.data.path.append(str(NLTK_DIR))
for recurso in ("punkt", "punkt_tab", "stopwords"):
    try:
        nltk.download(recurso, download_dir=str(NLTK_DIR), quiet=True)
    except Exception:
        pass

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
)

print("Descargando el modelo de resumen. Puede tardar varios minutos...")
T5Tokenizer.from_pretrained("t5-base")
T5ForConditionalGeneration.from_pretrained("t5-base")

print("Descargando el modelo de preguntas. Puede tardar varios minutos...")
NOMBRE_PREGUNTAS = "bert-large-uncased-whole-word-masking-finetuned-squad"
AutoTokenizer.from_pretrained(NOMBRE_PREGUNTAS)
AutoModelForQuestionAnswering.from_pretrained(NOMBRE_PREGUNTAS)

print("Listo. Los modelos quedaron en la carpeta models.")
```

Guarda y ejecuta:

```powershell
python src\descargar_modelos.py
```

### Qué debes ver

Al final:

```text
Listo. Los modelos quedaron en la carpeta models.
```

La carpeta `models` queda con archivos grandes. La primera descarga es la más lenta. Si más adelante un paso vuelve a cargar el mismo modelo, ya no lo descarga otra vez.

---

## Paso 2. Crear los tres documentos del banco (15 min)

### Qué vas a hacer

Generar los tres textos de ejemplo. Son documentos ficticios, con cifras concretas, para poder comprobar después si el asistente responde bien.

Crea el archivo `src\crear_documentos.py` y pega este programa:

```python
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA_RAW = BASE / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

terminos_tc = """TÉRMINOS Y CONDICIONES - TARJETA DE CRÉDITO ORO
Banco Nacional S.A. - Vigencia: Enero 2024

CAPÍTULO 1: DEFINICIONES
El presente documento establece los términos y condiciones aplicables a la Tarjeta de Crédito Oro emitida por Banco Nacional S.A. (en adelante "el Banco"). El titular de la tarjeta (en adelante "el Cliente") acepta las siguientes condiciones al momento de activar el plástico.

CAPÍTULO 2: TASAS Y COMISIONES
La tasa de interés anual ordinaria es del 24.5% sobre saldos no pagados. La tasa moratoria aplicable será del 36% anual. La comisión por anualidad es de $1,200.00 MXN, cobrada en el mes de aniversario de la cuenta. La comisión por disposición de efectivo en cajero automático es del 5% sobre el monto dispuesto, con un mínimo de $150.00 MXN. No se cobra comisión por consulta de saldo en cajeros propios.

CAPÍTULO 3: PAGOS Y ESTADOS DE CUENTA
El estado de cuenta se genera mensualmente con fecha de corte el día 15 de cada mes. El pago mínimo corresponde al 5% del saldo total o $200.00 MXN, lo que resulte mayor. La fecha límite de pago es 20 días naturales después de la fecha de corte. El pago para no generar intereses requiere liquidar el saldo total antes de la fecha límite.

CAPÍTULO 4: LÍMITE DE CRÉDITO
El límite de crédito inicial se determina con base en el análisis crediticio del solicitante. El Banco se reserva el derecho de modificar el límite de crédito, notificando al Cliente con 30 días de anticipación. El Cliente puede solicitar un aumento de límite después de 6 meses de uso responsable.

CAPÍTULO 5: SEGUROS Y PROTECCIONES
La tarjeta incluye seguro de fraude por transacciones no reconocidas, con cobertura hasta el límite de crédito. El Cliente debe reportar cualquier transacción no reconocida dentro de los 30 días posteriores a la fecha del estado de cuenta. El seguro de viajero aplica para compras de boletos aéreos realizadas con la tarjeta, con cobertura de hasta $2,000,000.00 MXN.

CAPÍTULO 6: CANCELACIÓN Y TERMINACIÓN
El Cliente puede solicitar la cancelación de la tarjeta en cualquier momento, sin penalización, siempre que no existan saldos pendientes. El Banco puede cancelar la tarjeta por falta de pago de tres mensualidades consecutivas. En caso de cancelación, el saldo pendiente deberá liquidarse en un plazo máximo de 12 meses.
"""

contrato_cc = """CONTRATO DE APERTURA DE CUENTA CORRIENTE
Banco Nacional S.A. - Contrato No. CC-2024-00891

PRIMERA: PARTES CONTRATANTES
Celebran el presente contrato, por una parte, Banco Nacional S.A., institución de banca múltiple, con domicilio en Avenida Reforma 500, Ciudad de México (en adelante "el Banco"), y por otra parte, la empresa Distribuidora Comercial del Norte S.A. de C.V., con RFC DCN-980515-AB3 (en adelante "el Cuentahabiente").

SEGUNDA: OBJETO DEL CONTRATO
El Banco se obliga a recibir depósitos de dinero en moneda nacional y extranjera, así como a realizar transferencias, pagos y demás operaciones bancarias que el Cuentahabiente solicite, conforme a las condiciones establecidas en este contrato.

TERCERA: REQUISITOS DE APERTURA
Para la apertura de la cuenta se requiere: identificación oficial vigente del representante legal, acta constitutiva de la empresa, comprobante de domicilio fiscal no mayor a 3 meses, constancia de situación fiscal actualizada, y un depósito inicial mínimo de $50,000.00 MXN.

CUARTA: MANEJO DE LA CUENTA
El saldo mínimo promedio mensual requerido es de $25,000.00 MXN. Si el saldo promedio mensual es inferior, se cobrará una comisión de $500.00 MXN. Las transferencias electrónicas nacionales tienen un costo de $5.80 MXN por operación. Las transferencias internacionales tienen un costo de $350.00 MXN más el tipo de cambio vigente.

QUINTA: TASAS DE RENDIMIENTO
Los depósitos en cuenta corriente generan un rendimiento anual del 3.2% sobre el saldo promedio mensual cuando este supere los $100,000.00 MXN. Para saldos entre $50,000.00 y $100,000.00 MXN, el rendimiento es del 1.8% anual. Los intereses se calculan diariamente y se abonan mensualmente.

SEXTA: CHEQUERA Y MEDIOS DE DISPOSICIÓN
El Banco proporcionará una chequera de 50 documentos sin costo al momento de la apertura. Chequeras adicionales tendrán un costo de $250.00 MXN cada una. Se proporcionará acceso a banca electrónica y aplicación móvil sin costo adicional.

SÉPTIMA: RESPONSABILIDADES DEL CUENTAHABIENTE
El Cuentahabiente se obliga a mantener actualizados sus datos fiscales y de contacto. Cualquier cambio debe notificarse al Banco en un plazo no mayor a 10 días hábiles. El Cuentahabiente es responsable del uso adecuado de los medios de disposición.

OCTAVA: VIGENCIA Y TERMINACIÓN
El presente contrato tiene vigencia indefinida. Cualquiera de las partes puede darlo por terminado mediante notificación escrita con 30 días de anticipación. En caso de terminación, el saldo disponible se entregará al Cuentahabiente dentro de los 5 días hábiles siguientes.
"""

reglamento_prestamos = """REGLAMENTO GENERAL DE PRÉSTAMOS PERSONALES
Banco Nacional S.A. - Versión 3.1 - Actualizado: Marzo 2024

ARTÍCULO 1: ÁMBITO DE APLICACIÓN
El presente reglamento aplica a todos los préstamos personales otorgados por Banco Nacional S.A. a personas físicas con actividad empresarial o asalariados con antigüedad laboral mínima de 1 año.

ARTÍCULO 2: MONTOS Y PLAZOS
El monto mínimo de préstamo es de $20,000.00 MXN y el máximo de $500,000.00 MXN. Los plazos disponibles son: 12, 24, 36 y 48 meses. El monto máximo se determina en función del ingreso mensual comprobable del solicitante, sin exceder el 40% de su capacidad de pago.

ARTÍCULO 3: TASAS DE INTERÉS
La tasa de interés anual fija para préstamos personales varía según el plazo:
- 12 meses: 18.5% anual
- 24 meses: 21.0% anual
- 36 meses: 23.5% anual
- 48 meses: 25.0% anual
El Costo Anual Total (CAT) incluye comisión por apertura, seguro de vida y gastos de investigación.

ARTÍCULO 4: COMISIÓN POR APERTURA
Se cobra una comisión por apertura del 2.5% sobre el monto total del préstamo, la cual puede financiarse dentro del mismo crédito o pagarse al momento del desembolso.

ARTÍCULO 5: REQUISITOS DE SOLICITUD
El solicitante debe presentar: identificación oficial vigente, comprobante de domicilio no mayor a 3 meses, últimos 3 recibos de nómina o declaración fiscal del último ejercicio, estado de cuenta bancario de los últimos 3 meses, y referencias personales y comerciales.

ARTÍCULO 6: GARANTÍAS
Los préstamos de hasta $200,000.00 MXN no requieren garantía real. Para montos superiores, se requiere un aval con capacidad de pago comprobable o garantía prendaria equivalente al 120% del monto solicitado.

ARTÍCULO 7: PAGOS ANTICIPADOS
El Cliente puede realizar pagos anticipados parciales o totales sin penalización. Los pagos anticipados se aplican directamente al capital. En caso de liquidación anticipada total, se recalculan los intereses al día de pago.

ARTÍCULO 8: MORA Y COBRANZA
En caso de mora, se aplicará una tasa moratoria del 40% anual sobre el saldo vencido. Después de 90 días de mora, el crédito se considerará vencido y se iniciará el proceso de cobranza extrajudicial. Los gastos de cobranza serán a cargo del deudor.

ARTÍCULO 9: SEGURO DE VIDA
Todo préstamo personal incluye un seguro de vida deudor que cubre el saldo insoluto en caso de fallecimiento o invalidez total y permanente del titular. La prima del seguro es del 0.45% mensual sobre el saldo insoluto.

ARTÍCULO 10: MODIFICACIONES
El Banco se reserva el derecho de modificar el presente reglamento, notificando a los clientes con 60 días de anticipación a través de los medios oficiales de comunicación.
"""

documentos = {
    "terminos_tarjeta_credito.txt": terminos_tc,
    "contrato_cuenta_corriente.txt": contrato_cc,
    "reglamento_prestamos.txt": reglamento_prestamos,
}

for nombre, texto in documentos.items():
    ruta = DATA_RAW / nombre
    ruta.write_text(texto, encoding="utf-8")
    print(f"{nombre}: {ruta.stat().st_size} bytes")

print("Listo. Los tres documentos quedaron en data\\raw.")
```

Guarda y ejecuta:

```powershell
python src\crear_documentos.py
```

### Qué debes ver

Tres líneas con el nombre de cada archivo y su tamaño, y al final:

```text
Listo. Los tres documentos quedaron en data\raw.
```

Abre `data\raw\terminos_tarjeta_credito.txt` con doble clic y confirma que se lee en español. Haz lo mismo con los otros dos si quieres conocer las cifras antes de preguntarle al asistente.

---

## Paso 3. Preparar los documentos para el asistente (15 min)

### Qué vas a hacer

El asistente no lee el archivo de una sola vez. Este paso limpia el texto y lo parte en trozos de un tamaño que el modelo de preguntas puede aceptar (como máximo, unas 512 piezas internas llamadas tokens).

También acepta un PDF: si más adelante dejas un `.pdf` en `data\raw`, el mismo programa extrae el texto. En este taller los tres archivos son `.txt`.

Crea el archivo `src\preparar_documentos.py` y pega este programa:

```python
import json
import os
import re
import sys
from pathlib import Path

import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer

BASE = Path(__file__).resolve().parent.parent
NLTK_DIR = BASE / "models" / "nltk_data"
os.environ["HF_HOME"] = str(BASE / "models")
os.environ["TRANSFORMERS_CACHE"] = str(BASE / "models")
os.environ["HF_HUB_CACHE"] = str(BASE / "models")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

NLTK_DIR.mkdir(parents=True, exist_ok=True)
nltk.data.path.append(str(NLTK_DIR))
for recurso in ("punkt", "punkt_tab"):
    try:
        nltk.download(recurso, download_dir=str(NLTK_DIR), quiet=True)
    except Exception:
        pass


class PreparadorDocumentos:
    """Lee un documento, lo limpia y lo parte en trozos."""

    def __init__(self):
        self.max_tokens = 512
        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-large-uncased-whole-word-masking-finetuned-squad"
        )
        self.raw_dir = BASE / "data" / "raw"
        self.processed_dir = BASE / "data" / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def leer_texto(self, nombre):
        return (self.raw_dir / nombre).read_text(encoding="utf-8")

    def leer_pdf(self, nombre):
        try:
            from PyPDF2 import PdfReader
            lector = PdfReader(str(self.raw_dir / nombre))
            return "\n".join((pagina.extract_text() or "") for pagina in lector.pages)
        except FileNotFoundError:
            return self.leer_texto(nombre.replace(".pdf", ".txt"))

    def limpiar(self, texto):
        texto = re.sub(r"\n{3,}", "\n\n", texto)
        texto = re.sub(r"[ \t]+", " ", texto)
        texto = "\n".join(linea.strip() for linea in texto.split("\n"))
        texto = re.sub(r"\n{3,}", "\n\n", texto)
        return texto.strip()

    def partir(self, texto):
        oraciones = sent_tokenize(texto, language="spanish")
        trozos = []
        actual = []
        largo = 0
        limite = self.max_tokens - 2

        for oracion in oraciones:
            tokens = self.tokenizer.encode(oracion, add_special_tokens=False)
            if len(tokens) > limite:
                if actual:
                    trozos.append(" ".join(actual))
                    actual = []
                    largo = 0
                sub = []
                sub_largo = 0
                for palabra in oracion.split():
                    n = len(self.tokenizer.encode(palabra, add_special_tokens=False))
                    if sub_largo + n > limite:
                        trozos.append(" ".join(sub))
                        sub = [palabra]
                        sub_largo = n
                    else:
                        sub.append(palabra)
                        sub_largo += n
                actual = sub
                largo = sub_largo
                continue

            if largo + len(tokens) > limite:
                trozos.append(" ".join(actual))
                actual = [oracion]
                largo = len(tokens)
            else:
                actual.append(oracion)
                largo += len(tokens)

        if actual:
            trozos.append(" ".join(actual))
        return trozos

    def procesar(self, nombre, doc_id):
        texto = self.leer_pdf(nombre) if nombre.endswith(".pdf") else self.leer_texto(nombre)
        limpio = self.limpiar(texto)
        trozos = self.partir(limpio)
        resultado = {
            "document_id": doc_id,
            "source_file": nombre,
            "full_text": limpio,
            "num_chunks": len(trozos),
            "chunks": [
                {
                    "chunk_id": i,
                    "text": trozo,
                    "num_tokens": len(self.tokenizer.encode(trozo)),
                }
                for i, trozo in enumerate(trozos)
            ],
        }
        destino = self.processed_dir / f"chunks_{doc_id}.json"
        destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
        return resultado


def main():
    preparador = PreparadorDocumentos()
    documentos = [
        ("terminos_tarjeta_credito.txt", "terminos"),
        ("contrato_cuenta_corriente.txt", "contrato"),
        ("reglamento_prestamos.txt", "reglamento"),
    ]
    print("Preparando documentos...")
    for nombre, doc_id in documentos:
        resultado = preparador.procesar(nombre, doc_id)
        print(f"{nombre}: {resultado['num_chunks']} trozos")
        for trozo in resultado["chunks"]:
            print(f"  trozo {trozo['chunk_id']}: {trozo['num_tokens']} tokens")
    print(r"Listo. Los archivos quedaron en data\processed.")


if __name__ == "__main__":
    main()
```

Guarda y ejecuta:

```powershell
python src\preparar_documentos.py
```

### Qué debes ver

Cada documento queda en **al menos un trozo**. El número de trozos y de tokens puede variar un poco. Lo importante es el cierre:

```text
Listo. Los archivos quedaron en data\processed.
```

En `data\processed` deben estar `chunks_terminos.json`, `chunks_contrato.json` y `chunks_reglamento.json`. Abre uno y comprueba que el texto sigue siendo el del documento, ahora dividido.

---

## Paso 4. Resumir los documentos (20 min)

### Qué vas a hacer

Pedir dos resúmenes de cada documento:

- Uno **copia** las tres frases que parecen más importantes.
- El otro **redacta** un texto nuevo con el modelo T5. Ese modelo se entrenó sobre todo en inglés, así que el resumen sale en inglés. Es el resultado esperado.

Después se comparan con un puntaje llamado ROUGE. Un número bajo no significa que el resumen esté mal: están en idiomas distintos.

Crea el archivo `src\resumir.py` y pega este programa:

```python
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
os.environ["HF_HOME"] = str(BASE / "models")
os.environ["TRANSFORMERS_CACHE"] = str(BASE / "models")
os.environ["HF_HUB_CACHE"] = str(BASE / "models")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json

import evaluate
import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import T5ForConditionalGeneration, T5Tokenizer

NLTK_DIR = BASE / "models" / "nltk_data"
NLTK_DIR.mkdir(parents=True, exist_ok=True)
nltk.data.path.append(str(NLTK_DIR))
for recurso in ("punkt", "punkt_tab"):
    try:
        nltk.download(recurso, download_dir=str(NLTK_DIR), quiet=True)
    except Exception:
        pass


class ResumenQueCopia:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words=None)

    def resumir(self, texto, num_oraciones=3):
        oraciones = sent_tokenize(texto, language="spanish")
        if len(oraciones) <= num_oraciones:
            return texto
        matriz = self.vectorizer.fit_transform(oraciones)
        puntajes = matriz.sum(axis=1).A1
        elegidas = sorted(np.argsort(puntajes)[-num_oraciones:])
        return " ".join(oraciones[i] for i in elegidas)


class ResumenRedactado:
    def __init__(self):
        print("Cargando el modelo de resumen...")
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.model = T5ForConditionalGeneration.from_pretrained("t5-base")
        self.model.eval()
        print("Modelo de resumen listo.")

    def resumir(self, texto, max_length=150, min_length=40):
        entrada = self.tokenizer.encode(
            "summarize: " + texto,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )
        ids = self.model.generate(
            entrada,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
        return self.tokenizer.decode(ids[0], skip_special_tokens=True)


class Comparador:
    def __init__(self):
        self.rouge = evaluate.load("rouge")

    def comparar(self, redactado, copiado):
        resultados = self.rouge.compute(predictions=[redactado], references=[copiado])
        return {
            "rouge1": round(resultados["rouge1"], 4),
            "rouge2": round(resultados["rouge2"], 4),
            "rougeL": round(resultados["rougeL"], 4),
        }


def main():
    procesados = BASE / "data" / "processed"
    salidas = BASE / "outputs"
    salidas.mkdir(parents=True, exist_ok=True)

    copia = ResumenQueCopia()
    redacta = ResumenRedactado()
    comparador = Comparador()
    filas = []

    print("Resumiendo documentos. Cada uno puede tardar unos minutos...")
    for archivo in ("chunks_terminos.json", "chunks_contrato.json", "chunks_reglamento.json"):
        datos = json.loads((procesados / archivo).read_text(encoding="utf-8"))
        doc_id = datos["document_id"]
        texto = datos["chunks"][0]["text"]
        print(f"Trabajando en: {doc_id}")

        resumen_copia = copia.resumir(texto, num_oraciones=3)
        resumen_redactado = redacta.resumir(texto)
        puntajes = comparador.comparar(resumen_redactado, resumen_copia)

        print(f"Resumen que copia frases: {resumen_copia}")
        print(f"Resumen redactado: {resumen_redactado}")
        print(
            f"Parecido ROUGE-1: {puntajes['rouge1']} | "
            f"ROUGE-2: {puntajes['rouge2']} | ROUGE-L: {puntajes['rougeL']}"
        )
        filas.append({
            "document": doc_id,
            "rouge1": puntajes["rouge1"],
            "rouge2": puntajes["rouge2"],
            "rougeL": puntajes["rougeL"],
            "resumen_redactado": resumen_redactado,
            "resumen_que_copia": resumen_copia,
        })

    tabla = pd.DataFrame(filas)
    destino = salidas / "resultados_rouge.csv"
    tabla[["document", "rouge1", "rouge2", "rougeL"]].to_csv(destino, index=False, encoding="utf-8")
    print(r"Listo. La comparación quedó en outputs\resultados_rouge.csv")
    print(tabla[["document", "rouge1", "rouge2", "rougeL"]].to_string(index=False))


if __name__ == "__main__":
    main()
```

Guarda y ejecuta:

```powershell
python src\resumir.py
```

### Qué debes ver

Para cada documento, un resumen en español (frases del original) y un resumen en inglés. Al final, una tabla con tres filas: `terminos`, `contrato` y `reglamento`.

Abre `outputs\resultados_rouge.csv`. Debe tener una fila por documento.

Lee el resumen en inglés y compáralo con el documento. Busca si menciona la tasa, la comisión o el plazo. Anota mentalmente si se entiende el sentido, aunque el idioma cambió.

---

## Paso 5. Hacer preguntas y ver el mapa (20 min)

### Qué vas a hacer

Hacer diez preguntas de negocio. El asistente:

1. Busca el trozo del documento más relacionado con la pregunta.
2. Señala la frase que usa como respuesta.
3. Entrega un número de confianza.
4. Guarda tres imágenes del mapa de atención.

Crea el archivo `src\preguntar.py` y pega este programa:

```python
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
os.environ["HF_HOME"] = str(BASE / "models")
os.environ["TRANSFORMERS_CACHE"] = str(BASE / "models")
os.environ["HF_HUB_CACHE"] = str(BASE / "models")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

NOMBRE_PREGUNTAS = "bert-large-uncased-whole-word-masking-finetuned-squad"


class BuscadorDeTrozos:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def buscar(self, pregunta, trozos):
        textos = [trozo["text"] for trozo in trozos]
        matriz = self.vectorizer.fit_transform([pregunta] + textos)
        similitudes = cosine_similarity(matriz[0:1], matriz[1:])[0]
        mejor = int(np.argmax(similitudes))
        return textos[mejor], mejor, float(similitudes[mejor])


class ModuloPreguntas:
    def __init__(self):
        print("Cargando el modelo de preguntas...")
        self.tokenizer = AutoTokenizer.from_pretrained(NOMBRE_PREGUNTAS)
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            NOMBRE_PREGUNTAS,
            output_attentions=True,
        )
        self.model.eval()
        self.pipeline = pipeline(
            "question-answering",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        self.buscador = BuscadorDeTrozos()
        print("Modelo de preguntas listo.")

    def responder_en_texto(self, pregunta, contexto):
        resultado = self.pipeline(question=pregunta, context=contexto)
        return {
            "answer": resultado["answer"],
            "score": round(resultado["score"], 4),
            "start": resultado["start"],
            "end": resultado["end"],
        }

    def responder_documento(self, pregunta, documento):
        texto, trozo_id, similitud = self.buscador.buscar(pregunta, documento["chunks"])
        respuesta = self.responder_en_texto(pregunta, texto)
        return {
            "question": pregunta,
            "answer": respuesta["answer"],
            "confidence": respuesta["score"],
            "chunk_id": trozo_id,
            "chunk_similarity": round(similitud, 4),
            "context_snippet": texto[:200] + "...",
        }

    def pesos_de_atencion(self, pregunta, contexto):
        entradas = self.tokenizer(
            pregunta,
            contexto,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )
        with torch.no_grad():
            salidas = self.model(**entradas)
        promedio = salidas.attentions[-1].mean(dim=1).squeeze(0)
        tokens = self.tokenizer.convert_ids_to_tokens(entradas["input_ids"][0])
        return promedio.numpy(), tokens

    def guardar_mapa(self, pregunta, contexto, ruta):
        atencion, tokens = self.pesos_de_atencion(pregunta, contexto)
        limite = min(40, len(tokens))
        figura, eje = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            atencion[:limite, :limite],
            xticklabels=tokens[:limite],
            yticklabels=tokens[:limite],
            cmap="YlOrRd",
            ax=eje,
            square=True,
        )
        eje.set_title(f"Mapa de atención\n{pregunta[:60]}")
        eje.set_xlabel("Palabras relacionadas")
        eje.set_ylabel("Palabras de origen")
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.yticks(fontsize=7)
        plt.tight_layout()
        ruta.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(ruta, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Mapa guardado: {ruta.name}")


def nombre_seguro(texto):
    limpio = re.sub(r"[^0-9A-Za-zÁÉÍÓÚáéíóúÑñ]+", "_", texto)
    return limpio.strip("_")[:40]


def main():
    procesados = BASE / "data" / "processed"
    mapas = BASE / "outputs" / "mapas_atencion"
    mapas.mkdir(parents=True, exist_ok=True)
    modulo = ModuloPreguntas()

    documentos = {}
    for archivo in ("chunks_terminos.json", "chunks_contrato.json", "chunks_reglamento.json"):
        datos = json.loads((procesados / archivo).read_text(encoding="utf-8"))
        documentos[datos["document_id"]] = datos

    preguntas = [
        ("terminos", "¿Cuál es la tasa de interés anual?"),
        ("terminos", "¿Qué comisiones aplican por disposición de efectivo?"),
        ("terminos", "¿Cuál es el pago mínimo?"),
        ("contrato", "¿Cuál es el saldo mínimo requerido?"),
        ("contrato", "¿Cuánto cuesta una transferencia internacional?"),
        ("contrato", "¿Qué rendimiento genera la cuenta?"),
        ("reglamento", "¿Cuál es el monto máximo de préstamo?"),
        ("reglamento", "¿Qué tasa aplica para un préstamo a 36 meses?"),
        ("reglamento", "¿Se puede hacer pago anticipado sin penalización?"),
        ("reglamento", "¿Qué pasa después de 90 días de mora?"),
    ]

    print("Respondiendo preguntas. Puede tardar varios minutos...")
    for numero, (doc_id, pregunta) in enumerate(preguntas, start=1):
        resultado = modulo.responder_documento(pregunta, documentos[doc_id])
        print(f"{numero}. [{doc_id}] {pregunta}")
        print(f"   Respuesta: {resultado['answer']}")
        print(f"   Confianza: {resultado['confidence']}")
        print(f"   Trozo usado: {resultado['chunk_id']}")

    print("Creando mapas de atención...")
    for doc_id, pregunta in (
        ("terminos", "¿Cuál es la tasa de interés anual?"),
        ("contrato", "¿Cuál es el saldo mínimo requerido?"),
        ("reglamento", "¿Cuál es el monto máximo de préstamo?"),
    ):
        contexto, _, _ = modulo.buscador.buscar(pregunta, documentos[doc_id]["chunks"])
        ruta = mapas / f"atencion_{doc_id}_{nombre_seguro(pregunta)}.png"
        modulo.guardar_mapa(pregunta, contexto[:300], ruta)

    print(r"Listo. Las imágenes quedaron en outputs\mapas_atencion.")


if __name__ == "__main__":
    main()
```

Guarda y ejecuta:

```powershell
python src\preguntar.py
```

### Qué debes ver

Diez respuestas. Contrástalas con el documento. Estas son las cifras correctas del texto de ejemplo:

| Pregunta | Dato que está en el documento |
|----------|-------------------------------|
| Tasa de la tarjeta | 24.5% anual |
| Disposición de efectivo | 5%, mínimo $150.00 MXN |
| Pago mínimo | 5% del saldo o $200.00 MXN |
| Saldo mínimo de la cuenta | $25,000.00 MXN |
| Transferencia internacional | $350.00 MXN más el tipo de cambio |
| Rendimiento de la cuenta | 3.2% si el saldo supera $100,000.00 MXN |
| Monto máximo del préstamo | $500,000.00 MXN |
| Tasa a 36 meses | 23.5% anual |
| Pago anticipado | Sí, sin penalización |
| Después de 90 días de mora | El crédito se marca vencido y empieza la cobranza |

La confianza y la redacción pueden variar. Si una respuesta no coincide con la tabla, el asistente se equivocó: anótala.

### Ver el mapa

1. En el panel izquierdo abre `outputs`, luego `mapas_atencion`.
2. Deben aparecer **3 imágenes** `.png`.
3. Haz doble clic en una. Visual Studio Code la muestra.
4. Los cuadros más oscuros son las palabras que el modelo relacionó con más fuerza. Busca si esas palabras están cerca de la cifra de la respuesta (una tasa, un monto, un plazo).

---

## Paso 6. Usar el asistente completo (25 min)

### Qué vas a hacer

Juntar el resumen y las preguntas en un solo asistente. Primero recorre solo los tres documentos y guarda un informe. Después te deja escribir, en la misma terminal, para probarlo tú.

Crea el archivo `src\asistente.py` y pega este programa:

```python
import json
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
BASE = SRC.parent
sys.path.insert(0, str(SRC))

os.environ["HF_HOME"] = str(BASE / "models")
os.environ["TRANSFORMERS_CACHE"] = str(BASE / "models")
os.environ["HF_HUB_CACHE"] = str(BASE / "models")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from preguntar import ModuloPreguntas
from resumir import Comparador, ResumenQueCopia, ResumenRedactado


class AsistenteBancario:
    def __init__(self):
        print("Iniciando el asistente. La carga puede tardar varios minutos...")
        self.copia = ResumenQueCopia()
        self.redacta = ResumenRedactado()
        self.preguntas = ModuloPreguntas()
        self.comparador = Comparador()
        self.documentos = self._cargar()
        print("Asistente listo.")
        print("Documentos: " + ", ".join(self.documentos.keys()))

    def _cargar(self):
        carpeta = BASE / "data" / "processed"
        documentos = {}
        for ruta in carpeta.glob("*.json"):
            datos = json.loads(ruta.read_text(encoding="utf-8"))
            documentos[datos["document_id"]] = datos
        return documentos

    def resumir(self, doc_id, metodo="redactado", num_oraciones=3):
        if doc_id not in self.documentos:
            return {"error": f"No existe '{doc_id}'. Usa: {', '.join(self.documentos)}"}
        texto = self.documentos[doc_id]["chunks"][0]["text"]
        if metodo == "redactado":
            resumen = self.redacta.resumir(texto)
        elif metodo == "copia":
            resumen = self.copia.resumir(texto, num_oraciones)
        else:
            return {"error": "El método debe ser 'redactado' o 'copia'."}
        return {
            "document_id": doc_id,
            "method": metodo,
            "summary": resumen,
            "source_length": len(texto.split()),
            "summary_length": len(resumen.split()),
        }

    def preguntar(self, pregunta, doc_id):
        if doc_id not in self.documentos:
            return {"error": f"No existe '{doc_id}'. Usa: {', '.join(self.documentos)}"}
        return self.preguntas.responder_documento(pregunta, self.documentos[doc_id])

    def evaluar(self, doc_id):
        copiado = self.resumir(doc_id, metodo="copia")
        redactado = self.resumir(doc_id, metodo="redactado")
        if "error" in copiado:
            return copiado
        puntajes = self.comparador.comparar(redactado["summary"], copiado["summary"])
        return {
            "document_id": doc_id,
            "resumen_que_copia": copiado["summary"],
            "resumen_redactado": redactado["summary"],
            "rouge": puntajes,
        }

    def conversar(self):
        print("")
        print("Puedes escribir estos comandos:")
        print("  docs")
        print("  resumir terminos")
        print("  preguntar contrato")
        print("  evaluar reglamento")
        print("  salir")
        print("Los nombres válidos son: terminos, contrato, reglamento.")

        while True:
            try:
                orden = input("\nAsistente > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nSesión terminada.")
                break
            if not orden:
                continue
            if orden == "salir":
                print("Sesión terminada.")
                break
            if orden == "docs":
                print("Documentos: " + ", ".join(self.documentos.keys()))
                continue
            if orden.startswith("resumir "):
                resultado = self.resumir(orden.split(" ", 1)[1].strip())
                if "error" in resultado:
                    print(resultado["error"])
                else:
                    print(resultado["summary"])
                    print(
                        f"Palabras del trozo: {resultado['source_length']}. "
                        f"Palabras del resumen: {resultado['summary_length']}."
                    )
                continue
            if orden.startswith("preguntar "):
                doc_id = orden.split(" ", 1)[1].strip()
                pregunta = input("Escribe la pregunta: ").strip()
                if not pregunta:
                    print("La pregunta está vacía.")
                    continue
                resultado = self.preguntar(pregunta, doc_id)
                if "error" in resultado:
                    print(resultado["error"])
                else:
                    print(f"Respuesta: {resultado['answer']}")
                    print(f"Confianza: {resultado['confidence']}")
                continue
            if orden.startswith("evaluar "):
                resultado = self.evaluar(orden.split(" ", 1)[1].strip())
                if "error" in resultado:
                    print(resultado["error"])
                else:
                    print(f"ROUGE-1: {resultado['rouge']['rouge1']}")
                    print(f"ROUGE-2: {resultado['rouge']['rouge2']}")
                    print(f"ROUGE-L: {resultado['rouge']['rougeL']}")
                continue
            print("Comando no reconocido. Usa: docs, resumir, preguntar, evaluar o salir.")


def evaluar_todo():
    asistente = AsistenteBancario()
    lineas = ["INFORME DEL ASISTENTE BANCARIO", ""]
    puntajes = []

    lineas.append("RESUMENES")
    for doc_id in ("terminos", "contrato", "reglamento"):
        resultado = asistente.evaluar(doc_id)
        puntajes.append(resultado["rouge"])
        print(f"Documento: {doc_id}")
        print(f"Resumen que copia frases: {resultado['resumen_que_copia']}")
        print(f"Resumen redactado: {resultado['resumen_redactado']}")
        print(
            f"ROUGE-1: {resultado['rouge']['rouge1']} | "
            f"ROUGE-2: {resultado['rouge']['rouge2']} | "
            f"ROUGE-L: {resultado['rouge']['rougeL']}"
        )
        lineas.append(f"Documento: {doc_id}")
        lineas.append(f"Resumen que copia frases: {resultado['resumen_que_copia']}")
        lineas.append(f"Resumen redactado: {resultado['resumen_redactado']}")
        lineas.append(
            f"ROUGE-1: {resultado['rouge']['rouge1']} | "
            f"ROUGE-2: {resultado['rouge']['rouge2']} | "
            f"ROUGE-L: {resultado['rouge']['rougeL']}"
        )
        lineas.append("")

    pruebas = [
        ("terminos", "¿Cuál es la tasa de interés anual?"),
        ("terminos", "¿Qué comisiones aplican por disposición de efectivo?"),
        ("terminos", "¿Cuál es el pago mínimo?"),
        ("terminos", "¿Cuándo se genera el estado de cuenta?"),
        ("contrato", "¿Cuál es el saldo mínimo requerido?"),
        ("contrato", "¿Cuánto cuesta una transferencia internacional?"),
        ("contrato", "¿Qué rendimiento genera la cuenta?"),
        ("reglamento", "¿Cuál es el monto máximo de préstamo?"),
        ("reglamento", "¿Se puede hacer pago anticipado sin penalización?"),
        ("reglamento", "¿Qué pasa después de 90 días de mora?"),
    ]

    lineas.append("PREGUNTAS")
    confianzas = []
    for numero, (doc_id, pregunta) in enumerate(pruebas, start=1):
        resultado = asistente.preguntar(pregunta, doc_id)
        confianzas.append(resultado["confidence"])
        texto = (
            f"{numero}. [{doc_id}] {pregunta}\n"
            f"   Respuesta: {resultado['answer']}\n"
            f"   Confianza: {resultado['confidence']}"
        )
        print(texto)
        lineas.append(texto)

    promedio_confianza = sum(confianzas) / len(confianzas)
    altas = sum(1 for valor in confianzas if valor > 0.5)
    promedio_rouge = sum(item["rouge1"] for item in puntajes) / len(puntajes)
    cierre = [
        "",
        "LECTURA RAPIDA",
        f"Confianza promedio: {promedio_confianza:.4f}",
        f"Respuestas con confianza mayor a 0.5: {altas} de {len(confianzas)}",
        f"ROUGE-1 promedio: {promedio_rouge:.4f}",
        "",
        "QUE TENER EN CUENTA",
        "1. El resumen redactado sale en inglés. El modelo se entrenó sobre todo con textos en inglés.",
        "2. El modelo de preguntas también se entrenó en inglés. Con cifras y términos claros suele acertar; con frases largas en español puede fallar.",
        "3. La búsqueda del trozo compara palabras, no el sentido completo de la pregunta.",
        "4. Si el dato queda partido entre dos trozos, el asistente puede no verlo.",
        "",
        "QUE HARIA FALTA PARA UN DOCUMENTO REAL",
        "1. Entrenar o cambiar a un modelo que trabaje bien en español.",
        "2. Buscar el trozo por significado, no solo por palabras repetidas.",
        "3. Hacer que los trozos se solapen, para no cortar una cláusula a la mitad.",
        "4. Revisar siempre la respuesta contra el documento antes de usarla con un cliente.",
    ]
    for linea in cierre:
        print(linea)
    lineas.extend(cierre)

    destino = BASE / "outputs" / "informe.txt"
    destino.write_text("\n".join(lineas), encoding="utf-8")
    print(r"Listo. El informe quedó en outputs\informe.txt")
    return asistente


if __name__ == "__main__":
    asistente = evaluar_todo()
    print("Puedes abrir outputs\\informe.txt y leerlo con calma.")
    print("Cuando termines de leer, vuelve a esta terminal y prueba el asistente.")
    asistente.conversar()
```

Guarda y ejecuta:

```powershell
python src\asistente.py
```

La carga y las respuestas pueden tardar bastante. Espera a ver `Asistente >`.

### Leer el informe

Sin cerrar la terminal, abre `outputs\informe.txt` en el panel izquierdo.

Revisa:

- Si el resumen en inglés conserva las cifras del documento.
- Cuántas respuestas tienen confianza mayor a 0.5.
- Qué respuestas no coinciden con el documento. Esas no se deben usar.

### Probarlo tú

Vuelve a la terminal. Escribe cada línea y presiona **Enter**. Cuando pida la pregunta, escríbela y vuelve a presionar **Enter**.

```text
docs
resumir terminos
preguntar terminos
¿Cuál es la comisión por anualidad?
evaluar contrato
preguntar reglamento
¿Los préstamos de hasta 200000 requieren garantía?
salir
```

Haz al menos **una pregunta que se te ocurra**, distinta a las de la lista. Mira la confianza y comprueba la respuesta abriendo el documento en `data\raw`.

---

## Paso 7. Escribir tus conclusiones (10 min)

Crea un archivo en la carpeta del proyecto (clic derecho en el espacio vacío del panel izquierdo > **Nuevo archivo**) llamado `mis_conclusiones.txt`.

Responde con tus palabras:

1. ¿El resumen en inglés dice lo mismo que el documento, o se pierde alguna cifra o condición?
2. De las respuestas del informe, ¿cuáles coinciden con el documento y cuáles no?
3. ¿Usarías este asistente con un contrato real del banco? Escribe qué tendría que mejorar antes de mostrarle una respuesta a un cliente.

Guarda con **Ctrl + S**.

Con eso el taller está completo. Debes tener:

- Tres documentos en `data\raw`.
- Tres archivos preparados en `data\processed`.
- `outputs\resultados_rouge.csv`.
- Tres imágenes en `outputs\mapas_atencion`.
- `outputs\informe.txt`.
- `mis_conclusiones.txt`.

## Si algo no sale

| Lo que ves | Qué hacer |
|------------|-----------|
| Python no se reconoce, o no es 3.10.12 | El equipo no está preparado. Usa [Prerequisitos.md](Prerequisitos.md). No instales programas por tu cuenta durante el taller. |
| La terminal no está en `AsistenteBancario` | Escribe `cd C:\Taller\AsistenteBancario` y repite la orden. |
| Un mensaje dice que no existe un archivo | Revisa el nombre en la carpeta `src`. Debe coincidir exactamente con el del paso. Guarda con **Ctrl + S** antes de ejecutar. |
| La terminal lleva varios minutos sin texto nuevo | Está trabajando. Espera. La primera carga de cada modelo es la más lenta. |
| Las tildes se ven raras | Vuelve a escribir `$env:PYTHONIOENCODING = "utf-8"` y `chcp 65001`, y ejecuta otra vez. |
| Falta una carpeta | Repite la orden del Paso 1 que crea `data`, `models`, `outputs` y `src`. |
