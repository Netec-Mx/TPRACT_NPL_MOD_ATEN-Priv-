---LAB_START---
LAB_ID: 01-00-01
---MARKDOWN---
# Construcción de un asistente para resumir y consultar documentos bancarios

## Metadata

| Campo | Valor |
|-------|-------|
| **Duración** | 105 minutos |
| **Complejidad** | Alta |
| **Nivel Bloom** | Crear |
| **Tecnologías clave** | Python 3.10.12, Hugging Face Transformers 4.38.2, PyTorch 2.1.2, T5-base, BERT-large-squad |

## Descripción General

En este laboratorio construirás un asistente conversacional completo capaz de resumir documentos bancarios y responder preguntas sobre su contenido. Integrarás un pipeline de resumen abstractivo con T5-base, un módulo de Question Answering con BERT fine-tuneado en SQuAD, y una interfaz interactiva con ipywidgets. El sistema procesará documentos bancarios reales (contratos, términos y reglamentos), evaluará la calidad de los resúmenes con métricas ROUGE y visualizará mapas de atención para interpretar las decisiones del modelo.

## Objetivos de Aprendizaje

Al completar este laboratorio serás capaz de:

- [ ] Implementar un pipeline end-to-end de ingesta, preprocesamiento y segmentación de documentos bancarios en chunks compatibles con modelos Transformer.
- [ ] Construir y evaluar un módulo de resumen automático con T5-base usando métricas ROUGE-1, ROUGE-2 y ROUGE-L.
- [ ] Desarrollar un módulo de respuesta a preguntas con BERT que localice respuestas precisas en documentos bancarios y visualice mapas de atención.
- [ ] Integrar ambos módulos en una clase `BankingAssistant` con interfaz interactiva funcional en JupyterLab.
- [ ] Analizar críticamente las limitaciones de modelos preentrenados en dominio específico y proponer estrategias de mejora.

## Prerrequisitos

### Conocimientos Requeridos

| Área | Nivel |
|------|-------|
| Arquitectura Transformer (encoder-decoder, multi-head attention) | Intermedio |
| Modelos BERT y T5 (tokenización, inferencia) | Intermedio |
| Python 3.10+ (clases, f-strings, comprehensions) | Intermedio |
| Hugging Face Transformers (pipeline API) | Básico |
| JupyterLab (ejecución de notebooks) | Básico |
| Métricas NLP (ROUGE, F1) | Básico |

### Acceso Requerido

- Acceso a terminal con permisos de escritura en `/workspace/`
- Conexión a Internet para descarga inicial de modelos desde Hugging Face Hub
- Archivos de datos provistos: `contrato_cuenta_corriente.pdf`, `terminos_tarjeta_credito.txt`, `reglamento_prestamos.pdf`

## Entorno del Laboratorio

### Hardware Mínimo

| Recurso | Mínimo | Recomendado |
|---------|--------|-------------|
| CPU | Intel i5 8ª gen / AMD Ryzen 5 3000+ | Intel i7 / Ryzen 7 |
| RAM | 16 GB DDR4 | 32 GB DDR4 |
| Almacenamiento libre | 20 GB | 30 GB |
| GPU (opcional) | — | NVIDIA RTX 3060 (6 GB VRAM) |
| Internet | 10 Mbps | 50 Mbps |

### Software Requerido

| Paquete | Versión |
|---------|---------|
| Python | 3.10.12 |
| JupyterLab | 4.1.5 |
| PyTorch | 2.1.2 |
| transformers | 4.38.2 |
| sentencepiece | 0.2.0 |
| datasets | 2.18.0 |
| evaluate | 0.4.1 |
| rouge-score | 0.1.2 |
| PyPDF2 | 3.0.1 |
| nltk | 3.8.1 |
| ipywidgets | 8.1.2 |
| matplotlib | 3.8.3 |
| seaborn | 0.13.2 |
| numpy | 1.26.4 |
| pandas | 2.2.1 |
| scikit-learn | 1.4.1 |

---

## Paso 1: Configuración del Entorno de Trabajo (15 min)

### Objetivo

Crear la estructura de directorios, instalar todas las dependencias y descargar los modelos preentrenados necesarios para el laboratorio.

### Instrucciones

1. Abre una terminal y crea la estructura de directorios del proyecto:

```bash
mkdir -p /workspace/nlp_banking_assistant/{data/raw,data/processed,models,outputs/mapas_atencion,notebooks,src}
cd /workspace/nlp_banking_assistant
```

2. Crea el archivo `requirements.txt` con las dependencias exactas:

```bash
cat > requirements.txt << 'EOF'
torch==2.1.2
transformers==4.38.2
sentencepiece==0.2.0
datasets==2.18.0
accelerate==0.27.2
evaluate==0.4.1
rouge-score==0.1.2
PyPDF2==3.0.1
nltk==3.8.1
ipywidgets==8.1.2
jupyterlab==4.1.5
matplotlib==3.8.3
seaborn==0.13.2
numpy==1.26.4
pandas==2.2.1
scikit-learn==1.4.1
tokenizers==0.15.2
EOF
```

3. Crea un entorno virtual e instala las dependencias:

```bash
python3.10 -m venv /workspace/nlp_banking_assistant/.venv
source /workspace/nlp_banking_assistant/.venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

4. Configura las variables de entorno para el cache de modelos:

```bash
export TRANSFORMERS_CACHE=/workspace/nlp_banking_assistant/models/
export HF_HOME=/workspace/nlp_banking_assistant/models/
echo 'export TRANSFORMERS_CACHE=/workspace/nlp_banking_assistant/models/' >> ~/.bashrc
echo 'export HF_HOME=/workspace/nlp_banking_assistant/models/' >> ~/.bashrc
```

5. Descarga los recursos NLTK necesarios y los modelos preentrenados. Crea el script `src/setup_models.py`:

```python
# /workspace/nlp_banking_assistant/src/setup_models.py
import nltk
import os

os.environ["TRANSFORMERS_CACHE"] = "/workspace/nlp_banking_assistant/models/"

# Descargar recursos NLTK
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# Descargar modelos de Hugging Face
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    AutoModelForQuestionAnswering, AutoTokenizer
)

print("Descargando T5-base...")
T5Tokenizer.from_pretrained("t5-base")
T5ForConditionalGeneration.from_pretrained("t5-base")

print("Descargando BERT-large-squad...")
AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

print("✓ Todos los modelos descargados exitosamente.")
```

6. Ejecuta el script de descarga:

```bash
cd /workspace/nlp_banking_assistant
python src/setup_models.py
```

7. Verifica las versiones instaladas:

```bash
python -c "
import torch, transformers, nltk, evaluate
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'CUDA disponible: {torch.cuda.is_available()}')
print(f'Evaluate: {evaluate.__version__}')
print('✓ Entorno verificado correctamente.')
"
```

### Resultado Esperado

```
PyTorch: 2.1.2
Transformers: 4.38.2
CUDA disponible: True  # o False si no hay GPU
Evaluate: 0.4.1
✓ Entorno verificado correctamente.
```

### Verificación

```bash
# Verificar estructura de directorios
find /workspace/nlp_banking_assistant -type d | sort
# Verificar que los modelos se descargaron
ls /workspace/nlp_banking_assistant/models/
# Debe mostrar subdirectorios con archivos de modelo
```

---

## Paso 2: Ingesta y Preprocesamiento de Documentos Bancarios (20 min)

### Objetivo

Cargar documentos bancarios desde archivos PDF y texto plano, extraer su contenido, limpiarlo y segmentarlo en chunks de máximo 512 tokens para compatibilidad con BERT.

### Instrucciones

1. Dado que los documentos bancarios reales son assets del laboratorio, crea documentos de muestra representativos para trabajar. Crea el script `src/create_sample_data.py`:

```python
# /workspace/nlp_banking_assistant/src/create_sample_data.py
"""
Genera documentos bancarios de muestra para el laboratorio.
En un entorno real, estos archivos serían provistos como assets.
"""
import os

DATA_RAW = "/workspace/nlp_banking_assistant/data/raw"

# Documento 1: Términos de tarjeta de crédito (texto plano)
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

# Documento 2: Contrato de cuenta corriente (simulando texto extraído de PDF)
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

# Documento 3: Reglamento de préstamos (simulando texto extraído de PDF)
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

# Guardar archivos
with open(os.path.join(DATA_RAW, "terminos_tarjeta_credito.txt"), "w", encoding="utf-8") as f:
    f.write(terminos_tc)

with open(os.path.join(DATA_RAW, "contrato_cuenta_corriente.txt"), "w", encoding="utf-8") as f:
    f.write(contrato_cc)

with open(os.path.join(DATA_RAW, "reglamento_prestamos.txt"), "w", encoding="utf-8") as f:
    f.write(reglamento_prestamos)

print(f"✓ Documentos creados en {DATA_RAW}/")
for f in os.listdir(DATA_RAW):
    filepath = os.path.join(DATA_RAW, f)
    size = os.path.getsize(filepath)
    print(f"  - {f} ({size} bytes)")
```

2. Ejecuta la creación de datos de muestra:

```bash
python src/create_sample_data.py
```

3. Ahora crea el módulo principal de preprocesamiento. Crea `src/preprocessor.py`:

```python
# /workspace/nlp_banking_assistant/src/preprocessor.py
"""
Módulo de ingesta y preprocesamiento de documentos bancarios.
Extrae texto, limpia y segmenta en chunks de máximo 512 tokens.
"""
import os
import re
import json
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

class DocumentPreprocessor:
    """Preprocesador de documentos bancarios."""
    
    def __init__(self, max_tokens=512, model_name="bert-large-uncased-whole-word-masking-finetuned-squad"):
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.raw_dir = "/workspace/nlp_banking_assistant/data/raw"
        self.processed_dir = "/workspace/nlp_banking_assistant/data/processed"
    
    def load_text_file(self, filename):
        """Carga un archivo de texto plano."""
        filepath = os.path.join(self.raw_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    
    def load_pdf_file(self, filename):
        """Carga un archivo PDF y extrae texto."""
        try:
            from PyPDF2 import PdfReader
            filepath = os.path.join(self.raw_dir, filename)
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except FileNotFoundError:
            # Si no hay PDF, buscar versión .txt
            txt_name = filename.replace(".pdf", ".txt")
            return self.load_text_file(txt_name)
    
    def clean_text(self, text):
        """Limpia el texto: normaliza espacios, elimina caracteres especiales innecesarios."""
        # Eliminar múltiples saltos de línea
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Normalizar espacios
        text = re.sub(r'[ \t]+', ' ', text)
        # Eliminar espacios al inicio/fin de líneas
        text = '\n'.join(line.strip() for line in text.split('\n'))
        # Eliminar líneas vacías consecutivas
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def segment_into_chunks(self, text):
        """
        Segmenta texto en chunks de máximo self.max_tokens tokens.
        Respeta límites de oraciones para mantener coherencia.
        """
        sentences = sent_tokenize(text, language='spanish')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            # Tokenizar la oración para contar tokens
            sentence_tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
            sentence_length = len(sentence_tokens)
            
            # Si una sola oración excede el máximo, dividirla
            if sentence_length > self.max_tokens - 2:  # -2 para [CLS] y [SEP]
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                # Dividir oración larga en sub-chunks
                words = sentence.split()
                sub_chunk = []
                sub_length = 0
                for word in words:
                    word_tokens = len(self.tokenizer.encode(word, add_special_tokens=False))
                    if sub_length + word_tokens > self.max_tokens - 2:
                        chunks.append(' '.join(sub_chunk))
                        sub_chunk = [word]
                        sub_length = word_tokens
                    else:
                        sub_chunk.append(word)
                        sub_length += word_tokens
                if sub_chunk:
                    current_chunk = sub_chunk
                    current_length = sub_length
                continue
            
            # Verificar si agregar la oración excede el límite
            if current_length + sentence_length > self.max_tokens - 2:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Agregar el último chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def process_document(self, filename, doc_id):
        """Procesa un documento completo: carga, limpia y segmenta."""
        # Cargar según extensión
        if filename.endswith('.pdf'):
            text = self.load_pdf_file(filename)
        else:
            text = self.load_text_file(filename)
        
        # Limpiar
        clean = self.clean_text(text)
        
        # Segmentar
        chunks = self.segment_into_chunks(clean)
        
        # Guardar resultado
        result = {
            "document_id": doc_id,
            "source_file": filename,
            "full_text": clean,
            "num_chunks": len(chunks),
            "chunks": [
                {
                    "chunk_id": i,
                    "text": chunk,
                    "num_tokens": len(self.tokenizer.encode(chunk))
                }
                for i, chunk in enumerate(chunks)
            ]
        }
        
        output_path = os.path.join(self.processed_dir, f"chunks_{doc_id}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return result


def main():
    """Procesa todos los documentos bancarios."""
    preprocessor = DocumentPreprocessor()
    
    documents = [
        ("terminos_tarjeta_credito.txt", "terminos"),
        ("contrato_cuenta_corriente.txt", "contrato"),
        ("reglamento_prestamos.txt", "reglamento"),
    ]
    
    print("=" * 60)
    print("PREPROCESAMIENTO DE DOCUMENTOS BANCARIOS")
    print("=" * 60)
    
    for filename, doc_id in documents:
        result = preprocessor.process_document(filename, doc_id)
        print(f"\n📄 {filename}")
        print(f"   Chunks generados: {result['num_chunks']}")
        for chunk in result['chunks']:
            print(f"   - Chunk {chunk['chunk_id']}: {chunk['num_tokens']} tokens")
    
    print("\n✓ Preprocesamiento completado.")
    print(f"  Archivos guardados en: {preprocessor.processed_dir}/")


if __name__ == "__main__":
    main()
```

4. Ejecuta el preprocesamiento:

```bash
cd /workspace/nlp_banking_assistant
python src/preprocessor.py
```

### Resultado Esperado

```
============================================================
PREPROCESAMIENTO DE DOCUMENTOS BANCARIOS
============================================================

📄 terminos_tarjeta_credito.txt
   Chunks generados: 2
   - Chunk 0: 387 tokens
   - Chunk 1: 298 tokens

📄 contrato_cuenta_corriente.txt
   Chunks generados: 2
   - Chunk 0: 412 tokens
   - Chunk 1: 345 tokens

📄 reglamento_prestamos.txt
   Chunks generados: 3
   - Chunk 0: 389 tokens
   - Chunk 1: 401 tokens
   - Chunk 2: 187 tokens

✓ Preprocesamiento completado.
  Archivos guardados en: /workspace/nlp_banking_assistant/data/processed/
```

> **Nota:** El número exacto de chunks y tokens puede variar ligeramente según la versión del tokenizador.

### Verificación

```bash
# Verificar que los archivos JSON se crearon correctamente
ls -la /workspace/nlp_banking_assistant/data/processed/
# Inspeccionar estructura de un archivo
python -c "
import json
with open('/workspace/nlp_banking_assistant/data/processed/chunks_terminos.json') as f:
    data = json.load(f)
print(f'Documento: {data[\"document_id\"]}')
print(f'Chunks: {data[\"num_chunks\"]}')
print(f'Primer chunk (primeros 100 chars): {data[\"chunks\"][0][\"text\"][:100]}...')
"
```

---

## Paso 3: Módulo de Resumen Automático con T5-base (25 min)

### Objetivo

Implementar un pipeline de resumen abstractivo con T5-base y un resumen extractivo basado en TF-IDF, evaluar ambos con métricas ROUGE y comparar resultados.

### Instrucciones

1. Crea el módulo de resumen `src/summarizer.py`:

```python
# /workspace/nlp_banking_assistant/src/summarizer.py
"""
Módulo de resumen automático: extractivo (TF-IDF) y abstractivo (T5-base).
"""
import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
import evaluate

os.environ["TRANSFORMERS_CACHE"] = "/workspace/nlp_banking_assistant/models/"


class ExtractiveSummarizer:
    """Resumen extractivo basado en puntuación TF-IDF de oraciones."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words=None)
    
    def summarize(self, text, num_sentences=3):
        """
        Selecciona las oraciones más relevantes según TF-IDF.
        
        Args:
            text: Texto completo a resumir.
            num_sentences: Número de oraciones a extraer.
        
        Returns:
            Resumen extractivo como string.
        """
        sentences = sent_tokenize(text, language='spanish')
        
        if len(sentences) <= num_sentences:
            return text
        
        # Calcular TF-IDF para cada oración
        tfidf_matrix = self.vectorizer.fit_transform(sentences)
        
        # Puntuar cada oración como la suma de sus pesos TF-IDF
        sentence_scores = tfidf_matrix.sum(axis=1).A1
        
        # Seleccionar las top-N oraciones (manteniendo orden original)
        top_indices = np.argsort(sentence_scores)[-num_sentences:]
        top_indices = sorted(top_indices)
        
        summary = ' '.join([sentences[i] for i in top_indices])
        return summary


class AbstractiveSummarizer:
    """Resumen abstractivo usando T5-base."""
    
    def __init__(self):
        print("Cargando modelo T5-base para resumen...")
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.model = T5ForConditionalGeneration.from_pretrained("t5-base")
        self.model.eval()
        print("✓ T5-base cargado.")
    
    def summarize(self, text, max_length=150, min_length=40):
        """
        Genera resumen abstractivo con T5-base.
        
        Args:
            text: Texto a resumir (prefijo 'summarize:' se agrega automáticamente).
            max_length: Longitud máxima del resumen en tokens.
            min_length: Longitud mínima del resumen en tokens.
        
        Returns:
            Resumen abstractivo generado.
        """
        # T5 requiere el prefijo de tarea
        input_text = "summarize: " + text
        
        # Tokenizar con truncamiento a 512 tokens
        inputs = self.tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        # Generar resumen
        summary_ids = self.model.generate(
            inputs,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        
        # Decodificar
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary


class SummaryEvaluator:
    """Evaluación de resúmenes con métricas ROUGE."""
    
    def __init__(self):
        self.rouge = evaluate.load("rouge")
    
    def evaluate_summary(self, prediction, reference):
        """
        Calcula métricas ROUGE entre resumen generado y referencia.
        
        Args:
            prediction: Resumen generado.
            reference: Resumen de referencia.
        
        Returns:
            Diccionario con scores ROUGE-1, ROUGE-2, ROUGE-L.
        """
        results = self.rouge.compute(
            predictions=[prediction],
            references=[reference]
        )
        return {
            "rouge1": round(results["rouge1"], 4),
            "rouge2": round(results["rouge2"], 4),
            "rougeL": round(results["rougeL"], 4)
        }


def main():
    """Ejecuta pipeline de resumen sobre documentos bancarios."""
    import pandas as pd
    
    processed_dir = "/workspace/nlp_banking_assistant/data/processed"
    outputs_dir = "/workspace/nlp_banking_assistant/outputs"
    
    # Inicializar módulos
    extractive = ExtractiveSummarizer()
    abstractive = AbstractiveSummarizer()
    evaluator = SummaryEvaluator()
    
    # Cargar documentos procesados
    doc_files = ["chunks_terminos.json", "chunks_contrato.json", "chunks_reglamento.json"]
    
    results = []
    
    print("\n" + "=" * 60)
    print("MÓDULO DE RESUMEN AUTOMÁTICO")
    print("=" * 60)
    
    for doc_file in doc_files:
        filepath = os.path.join(processed_dir, doc_file)
        with open(filepath, "r", encoding="utf-8") as f:
            doc_data = json.load(f)
        
        doc_id = doc_data["document_id"]
        full_text = doc_data["full_text"]
        
        # Usar primer chunk para resumen (por límite de tokens de T5)
        text_for_summary = doc_data["chunks"][0]["text"]
        
        print(f"\n{'─' * 50}")
        print(f"📄 Documento: {doc_id}")
        print(f"{'─' * 50}")
        
        # Resumen extractivo
        extractive_summary = extractive.summarize(text_for_summary, num_sentences=3)
        print(f"\n🔹 Resumen EXTRACTIVO:")
        print(f"   {extractive_summary[:200]}...")
        
        # Resumen abstractivo
        abstractive_summary = abstractive.summarize(text_for_summary)
        print(f"\n🔸 Resumen ABSTRACTIVO (T5-base):")
        print(f"   {abstractive_summary}")
        
        # Evaluar usando el extractivo como referencia (en ausencia de gold standard)
        scores = evaluator.evaluate_summary(abstractive_summary, extractive_summary)
        print(f"\n📊 Métricas ROUGE (abstractivo vs extractivo):")
        print(f"   ROUGE-1: {scores['rouge1']}")
        print(f"   ROUGE-2: {scores['rouge2']}")
        print(f"   ROUGE-L: {scores['rougeL']}")
        
        results.append({
            "document": doc_id,
            "rouge1": scores["rouge1"],
            "rouge2": scores["rouge2"],
            "rougeL": scores["rougeL"],
            "abstractive_summary": abstractive_summary,
            "extractive_summary": extractive_summary
        })
    
    # Guardar resultados en CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(outputs_dir, "resultados_rouge.csv")
    df[["document", "rouge1", "rouge2", "rougeL"]].to_csv(csv_path, index=False)
    print(f"\n✓ Resultados guardados en: {csv_path}")
    
    # Mostrar tabla resumen
    print("\n" + "=" * 60)
    print("RESUMEN DE MÉTRICAS ROUGE")
    print("=" * 60)
    print(df[["document", "rouge1", "rouge2", "rougeL"]].to_string(index=False))
    
    return results


if __name__ == "__main__":
    main()
```

2. Ejecuta el módulo de resumen:

```bash
cd /workspace/nlp_banking_assistant
python src/summarizer.py
```

### Resultado Esperado

```
Cargando modelo T5-base para resumen...
✓ T5-base cargado.

============================================================
MÓDULO DE RESUMEN AUTOMÁTICO
============================================================

──────────────────────────────────────────────────
📄 Documento: terminos
──────────────────────────────────────────────────

🔹 Resumen EXTRACTIVO:
   La tasa de interés anual ordinaria es del 24.5% sobre saldos no pagados...

🔸 Resumen ABSTRACTIVO (T5-base):
   the annual interest rate is 24.5% on unpaid balances. the late payment rate is 36% per year. the annual fee is $1,200.00 MXN.

📊 Métricas ROUGE (abstractivo vs extractivo):
   ROUGE-1: 0.3245
   ROUGE-2: 0.1102
   ROUGE-L: 0.2876

[... resultados para otros documentos ...]

✓ Resultados guardados en: /workspace/nlp_banking_assistant/outputs/resultados_rouge.csv
```

> **Nota importante:** T5-base genera resúmenes en inglés por defecto ya que fue preentrenado principalmente en texto en inglés. Esto es esperado y será un punto de análisis en la sección de evaluación. Los scores ROUGE reflejan esta diferencia de idioma.

### Verificación

```bash
# Verificar que el CSV de resultados existe y tiene contenido
cat /workspace/nlp_banking_assistant/outputs/resultados_rouge.csv
# Verificar que tiene 3 filas (una por documento) + header
wc -l /workspace/nlp_banking_assistant/outputs/resultados_rouge.csv
```

---

## Paso 4: Módulo de Respuesta a Preguntas con BERT (25 min)

### Objetivo

Implementar un módulo de Question Answering usando BERT fine-tuneado en SQuAD, con búsqueda de chunk relevante por similitud de coseno y visualización de mapas de atención.

### Instrucciones

1. Crea el módulo de QA `src/qa_module.py`:

```python
# /workspace/nlp_banking_assistant/src/qa_module.py
"""
Módulo de Question Answering con BERT y visualización de atención.
"""
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering,
    pipeline
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

os.environ["TRANSFORMERS_CACHE"] = "/workspace/nlp_banking_assistant/models/"


class ChunkRetriever:
    """Recupera el chunk más relevante para una pregunta dada."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
    
    def find_relevant_chunk(self, question, chunks):
        """
        Encuentra el chunk más relevante usando similitud de coseno sobre TF-IDF.
        
        Args:
            question: Pregunta del usuario.
            chunks: Lista de diccionarios con campo 'text'.
        
        Returns:
            Tupla (chunk_text, chunk_id, similarity_score).
        """
        texts = [chunk["text"] for chunk in chunks]
        all_texts = [question] + texts
        
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        
        # Similitud de coseno entre pregunta y cada chunk
        question_vector = tfidf_matrix[0:1]
        chunk_vectors = tfidf_matrix[1:]
        
        similarities = cosine_similarity(question_vector, chunk_vectors)[0]
        best_idx = np.argmax(similarities)
        
        return texts[best_idx], best_idx, similarities[best_idx]


class QAModule:
    """Módulo de Question Answering con BERT."""
    
    def __init__(self):
        model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
        print(f"Cargando modelo QA: {model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_name, 
            output_attentions=True
        )
        self.model.eval()
        
        # Pipeline para uso simplificado
        self.qa_pipeline = pipeline(
            "question-answering",
            model=self.model,
            tokenizer=self.tokenizer
        )
        
        self.retriever = ChunkRetriever()
        print("✓ Modelo QA cargado.")
    
    def answer_question(self, question, context):
        """
        Responde una pregunta dado un contexto.
        
        Args:
            question: Pregunta en texto.
            context: Texto de contexto donde buscar la respuesta.
        
        Returns:
            Diccionario con 'answer', 'score', 'start', 'end'.
        """
        result = self.qa_pipeline(question=question, context=context)
        return {
            "answer": result["answer"],
            "score": round(result["score"], 4),
            "start": result["start"],
            "end": result["end"]
        }
    
    def answer_from_document(self, question, doc_data):
        """
        Responde una pregunta buscando primero el chunk relevante.
        
        Args:
            question: Pregunta del usuario.
            doc_data: Diccionario con datos del documento procesado.
        
        Returns:
            Diccionario con respuesta, chunk usado y score.
        """
        chunks = doc_data["chunks"]
        
        # Recuperar chunk relevante
        relevant_text, chunk_id, sim_score = self.retriever.find_relevant_chunk(
            question, chunks
        )
        
        # Obtener respuesta
        answer = self.answer_question(question, relevant_text)
        
        return {
            "question": question,
            "answer": answer["answer"],
            "confidence": answer["score"],
            "chunk_id": chunk_id,
            "chunk_similarity": round(sim_score, 4),
            "context_snippet": relevant_text[:200] + "..."
        }
    
    def get_attention_weights(self, question, context):
        """
        Extrae los pesos de atención de la última capa del encoder.
        
        Args:
            question: Pregunta.
            context: Contexto.
        
        Returns:
            Tupla (attention_weights, tokens) de la última capa.
        """
        inputs = self.tokenizer(
            question, context,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Obtener atención de la última capa, primera cabeza
        # Shape: (batch, heads, seq_len, seq_len)
        last_layer_attention = outputs.attentions[-1]
        
        # Promediar sobre todas las cabezas
        avg_attention = last_layer_attention.mean(dim=1).squeeze(0)
        
        # Obtener tokens para etiquetas
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        return avg_attention.numpy(), tokens
    
    def visualize_attention(self, question, context, save_path=None):
        """
        Genera y guarda un mapa de calor de atención.
        
        Args:
            question: Pregunta formulada.
            context: Contexto usado.
            save_path: Ruta donde guardar la imagen PNG.
        """
        attention, tokens = self.get_attention_weights(question, context)
        
        # Limitar a primeros 40 tokens para visualización legible
        max_display = min(40, len(tokens))
        attention_subset = attention[:max_display, :max_display]
        tokens_subset = tokens[:max_display]
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            attention_subset,
            xticklabels=tokens_subset,
            yticklabels=tokens_subset,
            cmap="YlOrRd",
            ax=ax,
            square=True
        )
        ax.set_title(f"Mapa de Atención - Última Capa\nPregunta: {question[:60]}...", 
                     fontsize=11)
        ax.set_xlabel("Tokens (destino)")
        ax.set_ylabel("Tokens (origen)")
        plt.xticks(rotation=45, ha='right', fontsize=7)
        plt.yticks(fontsize=7)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   💾 Mapa guardado: {save_path}")
        
        plt.close()
        return attention_subset, tokens_subset


def main():
    """Ejecuta el módulo QA sobre preguntas de prueba."""
    processed_dir = "/workspace/nlp_banking_assistant/data/processed"
    attention_dir = "/workspace/nlp_banking_assistant/outputs/mapas_atencion"
    os.makedirs(attention_dir, exist_ok=True)
    
    # Inicializar módulo
    qa = QAModule()
    
    # Cargar documentos
    documents = {}
    for doc_file in ["chunks_terminos.json", "chunks_contrato.json", "chunks_reglamento.json"]:
        with open(os.path.join(processed_dir, doc_file), "r", encoding="utf-8") as f:
            data = json.load(f)
            documents[data["document_id"]] = data
    
    # Preguntas de prueba por documento
    test_questions = [
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
    
    print("\n" + "=" * 60)
    print("MÓDULO DE RESPUESTA A PREGUNTAS (QA)")
    print("=" * 60)
    
    for i, (doc_id, question) in enumerate(test_questions):
        result = qa.answer_from_document(question, documents[doc_id])
        
        print(f"\n{'─' * 50}")
        print(f"❓ Pregunta {i+1}: {question}")
        print(f"📄 Documento: {doc_id}")
        print(f"✅ Respuesta: {result['answer']}")
        print(f"   Confianza: {result['confidence']}")
        print(f"   Chunk usado: #{result['chunk_id']} (similitud: {result['chunk_similarity']})")
    
    # Generar mapas de atención para 3 preguntas representativas
    print(f"\n{'─' * 50}")
    print("Generando mapas de atención...")
    
    attention_questions = [
        ("terminos", "¿Cuál es la tasa de interés anual?"),
        ("contrato", "¿Cuál es el saldo mínimo requerido?"),
        ("reglamento", "¿Cuál es el monto máximo de préstamo?"),
    ]
    
    for doc_id, question in attention_questions:
        chunks = documents[doc_id]["chunks"]
        context, _, _ = qa.retriever.find_relevant_chunk(question, chunks)
        
        # Truncar contexto para visualización
        context_short = context[:300]
        
        save_path = os.path.join(
            attention_dir, 
            f"atencion_{doc_id}_{question[:20].replace(' ', '_').replace('?', '')}.png"
        )
        qa.visualize_attention(question, context_short, save_path=save_path)
    
    print("\n✓ Módulo QA ejecutado exitosamente.")


if __name__ == "__main__":
    main()
```

2. Ejecuta el módulo de QA:

```bash
cd /workspace/nlp_banking_assistant
python src/qa_module.py
```

### Resultado Esperado

```
Cargando modelo QA: bert-large-uncased-whole-word-masking-finetuned-squad...
✓ Modelo QA cargado.

============================================================
MÓDULO DE RESPUESTA A PREGUNTAS (QA)
============================================================

──────────────────────────────────────────────────
❓ Pregunta 1: ¿Cuál es la tasa de interés anual?
📄 Documento: terminos
✅ Respuesta: 24.5%
   Confianza: 0.7823
   Chunk usado: #0 (similitud: 0.4512)

──────────────────────────────────────────────────
❓ Pregunta 2: ¿Qué comisiones aplican por disposición de efectivo?
📄 Documento: terminos
✅ Respuesta: 5% sobre el monto dispuesto, con un mínimo de $150.00 MXN
   Confianza: 0.6234
   Chunk usado: #0 (similitud: 0.3891)

[... más preguntas ...]

──────────────────────────────────────────────────
Generando mapas de atención...
   💾 Mapa guardado: /workspace/nlp_banking_assistant/outputs/mapas_atencion/atencion_terminos_¿Cuál_es_la_tasa_d.png
   💾 Mapa guardado: /workspace/nlp_banking_assistant/outputs/mapas_atencion/atencion_contrato_¿Cuál_es_el_saldo_.png
   💾 Mapa guardado: /workspace/nlp_banking_assistant/outputs/mapas_atencion/atencion_reglamento_¿Cuál_es_el_monto_.png

✓ Módulo QA ejecutado exitosamente.
```

> **Nota:** Los scores de confianza y las respuestas exactas pueden variar ligeramente. BERT fue entrenado en inglés, por lo que su rendimiento en texto español es limitado pero funcional para términos numéricos y técnicos.

### Verificación

```bash
# Verificar que se generaron los mapas de atención
ls -la /workspace/nlp_banking_assistant/outputs/mapas_atencion/
# Deben existir 3 archivos PNG
find /workspace/nlp_banking_assistant/outputs/mapas_atencion -name "*.png" | wc -l
```

---

## Paso 5: Integración del Asistente y Evaluación Final (20 min)

### Objetivo

Integrar los módulos de resumen y QA en una clase unificada `BankingAssistant`, construir una interfaz interactiva con ipywidgets y realizar la evaluación final del sistema.

### Instrucciones

1. Crea el módulo integrador `src/banking_assistant.py`:

```python
# /workspace/nlp_banking_assistant/src/banking_assistant.py
"""
Asistente bancario integrado: resumen + QA + interfaz interactiva.
"""
import os
import json
import sys
sys.path.insert(0, "/workspace/nlp_banking_assistant/src")

from summarizer import AbstractiveSummarizer, ExtractiveSummarizer, SummaryEvaluator
from qa_module import QAModule

os.environ["TRANSFORMERS_CACHE"] = "/workspace/nlp_banking_assistant/models/"


class BankingAssistant:
    """
    Asistente conversacional para documentos bancarios.
    Combina resumen automático (T5-base) y respuesta a preguntas (BERT-squad).
    """
    
    def __init__(self):
        print("=" * 60)
        print("INICIALIZANDO ASISTENTE BANCARIO")
        print("=" * 60)
        
        # Cargar módulos
        self.extractive_summarizer = ExtractiveSummarizer()
        self.abstractive_summarizer = AbstractiveSummarizer()
        self.qa_module = QAModule()
        self.evaluator = SummaryEvaluator()
        
        # Cargar documentos procesados
        self.documents = self._load_documents()
        
        print("\n✓ Asistente bancario listo.")
        print(f"  Documentos cargados: {list(self.documents.keys())}")
    
    def _load_documents(self):
        """Carga todos los documentos procesados."""
        processed_dir = "/workspace/nlp_banking_assistant/data/processed"
        documents = {}
        
        for filename in os.listdir(processed_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(processed_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    documents[data["document_id"]] = data
        
        return documents
    
    def summarize(self, document_id, method="abstractive", num_sentences=3):
        """
        Genera un resumen del documento especificado.
        
        Args:
            document_id: ID del documento ('terminos', 'contrato', 'reglamento').
            method: 'abstractive' (T5) o 'extractive' (TF-IDF).
            num_sentences: Número de oraciones para resumen extractivo.
        
        Returns:
            Diccionario con resumen y metadatos.
        """
        if document_id not in self.documents:
            return {"error": f"Documento '{document_id}' no encontrado. "
                    f"Disponibles: {list(self.documents.keys())}"}
        
        doc = self.documents[document_id]
        text = doc["chunks"][0]["text"]  # Usar primer chunk
        
        if method == "abstractive":
            summary = self.abstractive_summarizer.summarize(text)
        elif method == "extractive":
            summary = self.extractive_summarizer.summarize(text, num_sentences)
        else:
            return {"error": f"Método '{method}' no soportado. Use 'abstractive' o 'extractive'."}
        
        return {
            "document_id": document_id,
            "method": method,
            "summary": summary,
            "source_length": len(text.split()),
            "summary_length": len(summary.split())
        }
    
    def ask(self, question, document_id):
        """
        Responde una pregunta sobre un documento específico.
        
        Args:
            question: Pregunta en lenguaje natural.
            document_id: ID del documento a consultar.
        
        Returns:
            Diccionario con respuesta, confianza y metadatos.
        """
        if document_id not in self.documents:
            return {"error": f"Documento '{document_id}' no encontrado. "
                    f"Disponibles: {list(self.documents.keys())}"}
        
        doc = self.documents[document_id]
        result = self.qa_module.answer_from_document(question, doc)
        
        return result
    
    def evaluate_summaries(self, document_id):
        """
        Evalúa ambos tipos de resumen con métricas ROUGE.
        
        Args:
            document_id: ID del documento.
        
        Returns:
            Diccionario con scores ROUGE.
        """
        extractive = self.summarize(document_id, method="extractive")
        abstractive = self.summarize(document_id, method="abstractive")
        
        scores = self.evaluator.evaluate_summary(
            abstractive["summary"], 
            extractive["summary"]
        )
        
        return {
            "document_id": document_id,
            "extractive_summary": extractive["summary"],
            "abstractive_summary": abstractive["summary"],
            "rouge_scores": scores
        }
    
    def interactive_session(self):
        """Ejecuta una sesión interactiva en terminal."""
        print("\n" + "=" * 60)
        print("🏦 ASISTENTE BANCARIO INTERACTIVO")
        print("=" * 60)
        print("Comandos disponibles:")
        print("  resumir <doc_id>     - Genera resumen del documento")
        print("  preguntar <doc_id>   - Hace una pregunta sobre el documento")
        print("  evaluar <doc_id>     - Evalúa resúmenes con ROUGE")
        print("  docs                 - Lista documentos disponibles")
        print("  salir                - Termina la sesión")
        print("─" * 60)
        
        while True:
            try:
                user_input = input("\n🏦 > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            
            if not user_input:
                continue
            
            if user_input == "salir":
                print("👋 Sesión terminada.")
                break
            
            elif user_input == "docs":
                print(f"📚 Documentos disponibles: {list(self.documents.keys())}")
            
            elif user_input.startswith("resumir "):
                doc_id = user_input.split(" ", 1)[1]
                result = self.summarize(doc_id)
                if "error" in result:
                    print(f"❌ {result['error']}")
                else:
                    print(f"\n📝 Resumen ({result['method']}) de '{doc_id}':")
                    print(f"   {result['summary']}")
                    print(f"   [Fuente: {result['source_length']} palabras → "
                          f"Resumen: {result['summary_length']} palabras]")
            
            elif user_input.startswith("preguntar "):
                doc_id = user_input.split(" ", 1)[1]
                question = input("   ❓ Escribe tu pregunta: ").strip()
                if question:
                    result = self.ask(question, doc_id)
                    if "error" in result:
                        print(f"❌ {result['error']}")
                    else:
                        print(f"\n   ✅ Respuesta: {result['answer']}")
                        print(f"   📊 Confianza: {result['confidence']}")
            
            elif user_input.startswith("evaluar "):
                doc_id = user_input.split(" ", 1)[1]
                result = self.evaluate_summaries(doc_id)
                if "error" in result:
                    print(f"❌ {result['error']}")
                else:
                    print(f"\n📊 Evaluación ROUGE para '{doc_id}':")
                    for metric, score in result["rouge_scores"].items():
                        print(f"   {metric}: {score}")
            
            else:
                print("❓ Comando no reconocido. Use: resumir, preguntar, evaluar, docs, salir")


def run_full_evaluation():
    """Ejecuta evaluación completa del asistente sin interacción."""
    assistant = BankingAssistant()
    
    print("\n" + "=" * 60)
    print("EVALUACIÓN COMPLETA DEL ASISTENTE")
    print("=" * 60)
    
    # Evaluación de resúmenes
    print("\n📝 EVALUACIÓN DE RESÚMENES")
    print("─" * 40)
    
    all_rouge_scores = []
    for doc_id in assistant.documents.keys():
        result = assistant.evaluate_summaries(doc_id)
        print(f"\n  📄 {doc_id}:")
        print(f"     Extractivo: {result['extractive_summary'][:100]}...")
        print(f"     Abstractivo: {result['abstractive_summary'][:100]}...")
        print(f"     ROUGE-1: {result['rouge_scores']['rouge1']} | "
              f"ROUGE-2: {result['rouge_scores']['rouge2']} | "
              f"ROUGE-L: {result['rouge_scores']['rougeL']}")
        all_rouge_scores.append(result["rouge_scores"])
    
    # Evaluación de QA
    print("\n\n❓ EVALUACIÓN DE QA (10 preguntas)")
    print("─" * 40)
    
    test_questions = [
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
    
    qa_results = []
    for i, (doc_id, question) in enumerate(test_questions, 1):
        result = assistant.ask(question, doc_id)
        print(f"\n  {i}. [{doc_id}] {question}")
        print(f"     → {result['answer']} (confianza: {result['confidence']})")
        qa_results.append(result)
    
    # Análisis de confianza
    confidences = [r["confidence"] for r in qa_results]
    avg_confidence = sum(confidences) / len(confidences)
    high_conf = sum(1 for c in confidences if c > 0.5)
    
    print(f"\n\n📊 RESUMEN DE EVALUACIÓN")
    print("─" * 40)
    print(f"  Confianza promedio QA: {avg_confidence:.4f}")
    print(f"  Respuestas con alta confianza (>0.5): {high_conf}/{len(confidences)}")
    
    avg_rouge1 = sum(s["rouge1"] for s in all_rouge_scores) / len(all_rouge_scores)
    print(f"  ROUGE-1 promedio: {avg_rouge1:.4f}")
    
    # Análisis de limitaciones
    print(f"\n\n⚠️  LIMITACIONES IDENTIFICADAS")
    print("─" * 40)
    print("  1. T5-base genera resúmenes en inglés (preentrenado en corpus inglés)")
    print("  2. BERT-squad tiene rendimiento reducido en texto español")
    print("  3. La búsqueda por TF-IDF no captura semántica profunda")
    print("  4. Chunks de 512 tokens pueden cortar información contextual")
    
    print(f"\n\n💡 ESTRATEGIAS DE MEJORA PROPUESTAS")
    print("─" * 40)
    print("  1. Fine-tuning de T5 con corpus bancario en español")
    print("  2. Usar modelos multilingües: mT5-base, BETO para QA en español")
    print("  3. Reemplazar TF-IDF por embeddings densos (sentence-transformers)")
    print("  4. Implementar RAG con overlap entre chunks")
    print("  5. Prompt engineering: agregar instrucciones en español al input de T5")
    
    print("\n✓ Evaluación completa finalizada.")
    
    return assistant


if __name__ == "__main__":
    assistant = run_full_evaluation()
```

2. Ejecuta la evaluación completa:

```bash
cd /workspace/nlp_banking_assistant
python src/banking_assistant.py
```

3. Ahora crea el notebook con la interfaz interactiva. Crea `notebooks/01-00-01_asistente_bancario.ipynb` ejecutando:

```bash
cd /workspace/nlp_banking_assistant
python -c "
import json

notebook = {
    'cells': [
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': ['# 🏦 Asistente Bancario Inteligente\n',
                      '## Laboratorio 01-00-01: Resumen y Consulta de Documentos\n',
                      '\n',
                      'Este notebook integra los módulos de resumen (T5-base) y QA (BERT-squad) ',
                      'en una interfaz interactiva.']
        },
        {
            'cell_type': 'code',
            'metadata': {},
            'source': [
                'import sys\\n',
                'import os\\n',
                'sys.path.insert(0, \"/workspace/nlp_banking_assistant/src\")\\n',
                'os.environ[\"TRANSFORMERS_CACHE\"] = \"/workspace/nlp_banking_assistant/models/\"\\n',
                '\\n',
                'from banking_assistant import BankingAssistant\\n',
                '\\n',
                '# Inicializar asistente\\n',
                'assistant = BankingAssistant()'
            ],
            'outputs': [],
            'execution_count': None
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': ['## Interfaz Interactiva con ipywidgets']
        },
        {
            'cell_type': 'code',
            'metadata': {},
            'source': [
                'import ipywidgets as widgets\\n',
                'from IPython.display import display, HTML\\n',
                '\\n',
                '# Widgets\\n',
                'doc_dropdown = widgets.Dropdown(\\n',
                '    options=list(assistant.documents.keys()),\\n',
                '    value=list(assistant.documents.keys())[0],\\n',
                '    description=\"Documento:\",\\n',
                '    style={\"description_width\": \"initial\"}\\n',
                ')\\n',
                '\\n',
                'question_input = widgets.Text(\\n',
                '    placeholder=\"Escribe tu pregunta aquí...\",\\n',
                '    description=\"Pregunta:\",\\n',
                '    style={\"description_width\": \"initial\"},\\n',
                '    layout=widgets.Layout(width=\"80%\")\\n',
                ')\\n',
                '\\n',
                'btn_summarize = widgets.Button(\\n',
                '    description=\"📝 Resumir\",\\n',
                '    button_style=\"info\",\\n',
                '    layout=widgets.Layout(width=\"150px\")\\n',
                ')\\n',
                '\\n',
                'btn_ask = widgets.Button(\\n',
                '    description=\"❓ Preguntar\",\\n',
                '    button_style=\"success\",\\n',
                '    layout=widgets.Layout(width=\"150px\")\\n',
                ')\\n',
                '\\n',
                'output_area = widgets.Output(\\n',
                '    layout=widgets.Layout(border=\"1px solid #ccc\", padding=\"10px\", min_height=\"200px\")\\n',
                ')\\n',
                '\\n',
                'def on_summarize(btn):\\n',
                '    output_area.clear_output()\\n',
                '    with output_area:\\n',
                '        doc_id = doc_dropdown.value\\n',
                '        print(f\"Generando resumen de: {doc_id}...\")\\n',
                '        result = assistant.summarize(doc_id)\\n',
                '        print(f\"\\\\n📝 Resumen ({result[\\'method\\']}):\")\\n',
                '        print(f\"   {result[\\'summary\\']}\")\\n',
                '        print(f\"\\\\n   [{result[\\'source_length\\']} palabras → {result[\\'summary_length\\']} palabras]\")\\n',
                '\\n',
                'def on_ask(btn):\\n',
                '    output_area.clear_output()\\n',
                '    with output_area:\\n',
                '        doc_id = doc_dropdown.value\\n',
                '        question = question_input.value\\n',
                '        if not question:\\n',
                '            print(\"⚠️ Escribe una pregunta primero.\")\\n',
                '            return\\n',
                '        print(f\"Buscando respuesta en: {doc_id}...\")\\n',
                '        result = assistant.ask(question, doc_id)\\n',
                '        print(f\"\\\\n❓ {question}\")\\n',
                '        print(f\"✅ Respuesta: {result[\\'answer\\']}\")\\n',
                '        print(f\"📊 Confianza: {result[\\'confidence\\']}\")\\n',
                '\\n',
                'btn_summarize.on_click(on_summarize)\\n',
                'btn_ask.on_click(on_ask)\\n',
                '\\n',
                '# Layout\\n',
                'ui = widgets.VBox([\\n',
                '    widgets.HTML(\"<h3>🏦 Asistente Bancario</h3>\"),\\n',
                '    doc_dropdown,\\n',
                '    question_input,\\n',
                '    widgets.HBox([btn_summarize, btn_ask]),\\n',
                '    widgets.HTML(\"<hr>\"),\\n',
                '    output_area\\n',
                '])\\n',
                '\\n',
                'display(ui)'
            ],
            'outputs': [],
            'execution_count': None
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': ['## Evaluación con Preguntas Predefinidas']
        },
        {
            'cell_type': 'code',
            'metadata': {},
            'source': [
                '# Ejecutar evaluación completa\\n',
                'from banking_assistant import run_full_evaluation\\n',
                'assistant = run_full_evaluation()'
            ],
            'outputs': [],
            'execution_count': None
        }
    ],
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.10.12'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 5
}

with open('notebooks/01-00-01_asistente_bancario.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print('✓ Notebook creado: notebooks/01-00-01_asistente_bancario.ipynb')
"
```

4. Exporta el reporte final en HTML:

```bash
cd /workspace/nlp_banking_assistant
# Si jupyter está instalado, exportar notebook como HTML
jupyter nbconvert --to html notebooks/01-00-01_asistente_bancario.ipynb \
    --output-dir outputs/ \
    --output reporte_final.html 2>/dev/null || \
    echo "Nota: nbconvert requiere ejecutar el notebook primero. El reporte se generará al ejecutar en JupyterLab."
```

### Resultado Esperado

La ejecución de `banking_assistant.py` produce una evaluación completa:

```
============================================================
INICIALIZANDO ASISTENTE BANCARIO
============================================================
Cargando modelo T5-base para resumen...
✓ T5-base cargado.
Cargando modelo QA: bert-large-uncased-whole-word-masking-finetuned-squad...
✓ Modelo QA cargado.

✓ Asistente bancario listo.
  Documentos cargados: ['terminos', 'contrato', 'reglamento']

============================================================
EVALUACIÓN COMPLETA DEL ASISTENTE
============================================================

📝 EVALUACIÓN DE RESÚMENES
────────────────────────────────────────
  📄 terminos:
     ROUGE-1: 0.32 | ROUGE-2: 0.11 | ROUGE-L: 0.28

  📄 contrato:
     ROUGE-1: 0.28 | ROUGE-2: 0.09 | ROUGE-L: 0.25

  📄 reglamento:
     ROUGE-1: 0.30 | ROUGE-2: 0.10 | ROUGE-L: 0.27

❓ EVALUACIÓN DE QA (10 preguntas)
────────────────────────────────────────
  1. [terminos] ¿Cuál es la tasa de interés anual?
     → 24.5% (confianza: 0.78)
  [... más respuestas ...]

📊 RESUMEN DE EVALUACIÓN
────────────────────────────────────────
  Confianza promedio QA: 0.5432
  Respuestas con alta confianza (>0.5): 7/10
  ROUGE-1 promedio: 0.3000

⚠️  LIMITACIONES IDENTIFICADAS
────────────────────────────────────────
  1. T5-base genera resúmenes en inglés
  2.
