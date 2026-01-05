# Documentación P4Tutor - Tutor de Algoritmos con Llama-3

## Descripción General

Este proyecto implementa un **Tutor Inteligente especializado en algoritmos** utilizando el modelo de lenguaje **Llama-3**  y fue optimizado con técnicas de finetuning. El modelo está diseñado para ejecutarse en **Google Colab** ya que la capacid del equipo fisico es insuficiente. 
Entrena un modelo capaz de enseñar y explicar conceptos de algoritmos de forma pedagógica.

---

## Tabla de Contenidos

1. [Requisitos e Instalación](#requisitos-e-instalación)
2. [Configuración del Modelo](#configuración-del-modelo)
3. [Carga y Verificación de Datos](#carga-y-verificación-de-datos)
4. [Procesamiento de Dataset](#procesamiento-de-dataset)
5. [Entrenamiento del Modelo](#entrenamiento-del-modelo)
6. [Evaluación y Pruebas](#evaluación-y-pruebas)
7. [Descarga de Resultados](#descarga-de-resultados)

---

## Requisitos e Instalación


### Instalación de Dependencias Principal

**Objetivo**: Instalar las librerías necesarias compatibles con la GPU del entorno.

```python
import torch
major_version, minor_version = torch.cuda.get_device_capability()
```

- **`torch.cuda.get_device_capability()`**: Obtiene la versión de la arquitectura GPU
- Esto determina qué versión de CUDA es compatible

**Instalaciones realizadas**:
- **Unsloth**: Framework de optimización para modelos de lenguaje
- **Xformers**: Optimización de transformers para mayor eficiencia
- **TRL**: Trainer para Reinforcement Learning
- **PEFT**: Parameter efficient Finetuning (ajuste fino eficiente)
- **Accelerate**: Aceleración de entrenamiento distribuido
- **Bitsandbytes**: Cuantización de 8 bits

### Instalación de Dependencias Adicionales

```python
!pip install --no-deps unsloth_zoo
!pip install trl==0.12.0
```

- **unsloth_zoo**: Catálogo de modelos optimizados
- **trl==0.12.0**: Versión específica compatible con el zoo

---

## Configuración del Modelo

### Inicialización de Llama-3 con Unsloth

**Objetivo**: Cargar y configurar el modelo base Llama-3 con optimizaciones.

```python
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    load_in_4bit = True,
)
```

**Parámetros explicados**:

| Parámetro | Valor | Explicación |
|-----------|-------|-------------|
| `model_name` | `unsloth/llama-3-8b-bnb-4bit` | Modelo Llama-3 de 8B parámetros cuantizado a 4 bits |
| `max_seq_length` | 2048 | Longitud máxima de tokens que puede procesar (aprox. 1500 palabras) |
| `load_in_4bit` | True | Cargar en formato 4-bit para reducir uso de memoria |

### LoRA: Configuración del Finetuning Eficiente

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 64,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
)
```

**¿Qué es LoRA?**

LoRA (Low-Rank Adaptation) es una técnica que:
- Solo entrena una pequeña fracción de parámetros
- Reduce memoria necesaria en ~90%
- Mantiene precisión comparable al fine-tuning completo

**Parámetros de LoRA**:

| Parámetro | Valor | Significado |
|-----------|-------|------------|
| `r` | 32 | Rango de la matriz de adaptación (balance entre eficiencia y capacidad) |
| `target_modules` | 7 módulos | Capas del transformer a adaptar (queries, keys, values, etc.) |
| `lora_alpha` | 64 | Factor de escala del aprendizaje |
| `lora_dropout` | 0 | Sin dropout regularizador en LoRA |
| `bias` | "none" | No adaptar bias, solo pesos principales |
| `use_gradient_checkpointing` | "unsloth" | Optimización de memoria durante backprop |

---

## Carga y Verificación de Datos

### Verificación del Dataset Principal

**Objetivo**: Confirmar que el archivo del dataset existe antes de procesar.

```python
import os
archivo = "dataset_algoritmos.jsonl"
if os.path.exists(archivo):
    print(f"Archivo '{archivo}' encontrado. ¡Puedes continuar!")
else:
    print(f"El archivo '{archivo}' NO está en /content/. Por favor, súbelo de nuevo.")
```

**Formato JSONL**:
- **JSONL** (JSON Lines) = Archivo con un JSON por línea
- Ideal para datasets de chat/conversación
- Cada línea contiene: `{"messages": [{"role": "user", "content": "..."}, {...}]}`

---

## Procesamiento de Dataset

### Aplicar Formato de Chat de Llama-3 (Primera Versión)

**Objetivo**: Convertir mensajes a formato de chat siguiendo la plantilla oficial de Llama-3.

```python
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3",
)

def format_prompt(examples):
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        texts.append(text)
    return { "text" : texts }

dataset = dataset.map(format_prompt, batched=True)
```

**¿Por qué esto es importante?**

- Llama-3 tiene un formato específico para distinguir roles (user, assistant)
- `apply_chat_template()` convierte lista de mensajes al formato correcto
- `tokenize=False`: Devuelve texto, no tokens aún
- `add_generation_prompt=False`: No añade tokens especiales de generación

---

### Carga de Múltiples Datasets

**Objetivo**: Cargar dos archivos de training para ampliar el conocimiento del tutor.

```python
from datasets import load_dataset

data_files = {
    "train": ["dataset_algoritmos.jsonl", 
              "dataset_algoritmos_basico_400.jsonl"]
}

dataset = load_dataset("json", data_files=data_files, split="train")
```

**Dos datasets**:
1. **dataset_algoritmos.jsonl**: Dataset general completo
2. **dataset_algoritmos_basico_400.jsonl**: 400 ejemplos básicos enfocados

**Ventaja**: Combinar ambos permite:
- Cobertura amplia de temas
- Énfasis en fundamentos básicos
- Mejor generalización

**Formato del chat vectorizado**:
```python
def format_prompt(examples):
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        texts.append(text)
    return { "text" : texts }

dataset = dataset.map(format_prompt, batched=True)
```

- `batched=True`: Procesa múltiples ejemplos simultáneamente para eficiencia
- Cada ejemplo se convierte a formato de chat de Llama-3

---

## Entrenamiento del Modelo

### Configuración y Ejecución del Entrenador

**Objetivo**: Configurar los parámetros de entrenamiento y ejecutar el fine-tuning.

```python
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(...)
)
```

**Parámetros del SFTTrainer**:

| Parámetro | Valor | Explicación |
|-----------|-------|------------|
| `model` | modelo Llama-3 | Modelo a entrenar |
| `tokenizer` | tokenizer Llama-3 | Convierte texto a tokens |
| `train_dataset` | dataset procesado | Datos de entrenamiento |
| `dataset_text_field` | "text" | Campo del dataset con el texto |
| `max_seq_length` | 2048 | Truncar o rellenar a esta longitud |
| `dataset_num_proc` | 2 | Procesos paralelos para cargar datos |

#### Argumentos de Entrenamiento (TrainingArguments)

```python
TrainingArguments(
    num_train_epochs = 3,              # Pases completos sobre el dataset
    per_device_train_batch_size = 2,   # Ejemplos por lote en GPU
    gradient_accumulation_steps = 4,   # Acumular gradientes 4 pasos
    warmup_steps = 5,                  # Calentar tasa de aprendizaje
    max_steps = 300,                   # Total de pasos de entrenamiento
    learning_rate = 2e-4,              # Tasa de aprendizaje (0.0002)
    fp16 = not is_bfloat16_supported(), # Precisión mixta 16-bit float
    bf16 = is_bfloat16_supported(),    # O 16-bit brain float si disponible
    logging_steps = 1,                 # Registrar métricas cada paso
    optim = "adamw_8bit",              # Optimizador AdamW en 8-bit
    weight_decay = 0.01,               # Regularización L2
    lr_scheduler_type = "linear",      # Programar tasa de aprendizaje lineal
    seed = 3407,                       # Reproducibilidad
    output_dir = "outputs",            # Guardar checkpoints aquí
)
```

**Explicación de parámetros clave**:

- **num_train_epochs**: 3 pasadas completas sobre los datos
- **per_device_train_batch_size**: Solo 2 ejemplos por batch (limitación GPU)
- **gradient_accumulation_steps**: Simular batch de 8 (2 × 4) sin más memoria
- **max_steps**: Entrenar máximo 300 actualizaciones de pesos
- **learning_rate**: 0.0002 es pequeña para evitar sobreajuste
- **fp16/bf16**: Precisión mixta para ahorrar memoria
- **adamw_8bit**: Optimizador con pesos de 8-bits

#### Entrenamiento y Guardado

```python
trainer_stats = trainer.train()

model.save_pretrained("tutor_llama3_final")
tokenizer.save_pretrained("tutor_llama3_final")
```

- **trainer.train()**: Ejecuta el ciclo completo de entrenamiento
- **save_pretrained()**: Guarda los pesos entrenados y configuración

---

## Evaluación y Pruebas

### Función de Evaluación del Tutor

**Objetivo**: Hacer preguntas al tutor entrenado y obtener respuestas pedagógicas.

```python
def evaluar_tutor(pregunta):
    # 1. Preparar mensaje en formato Llama-3
    messages = [
        {"role": "user", "content": pregunta},
    ]

    # Convertir a tokens con prompt de generación
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    # 2. Generar respuesta
    outputs = model.generate(
        input_ids = inputs,
        max_new_tokens = 500,
        temperature = 0.1,
        repetition_penalty = 1.2,
        eos_token_id = tokenizer.eos_token_id,
        pad_token_id = tokenizer.pad_token_id
    )

    # 3. Limpiar y devolver respuesta
    decoded = tokenizer.decode(
        outputs[0][len(inputs[0]):], 
        skip_special_tokens=True
    )
    respuesta_limpia = decoded.split("user")[0].split("assistant")[0].strip()
    return respuesta_limpia
```

#### Parámetros de Generación

| Parámetro | Valor | Explicación |
|-----------|-------|------------|
| `max_new_tokens` | 500 | Máximo de tokens nuevos a generar |
| `temperature` | 0.1 | Baja aleatoriedad = respuestas coherentes y pedagógicas |
| `repetition_penalty` | 1.2 | Penalizar repetición de palabras |
| `eos_token_id` | ID especial | Detener al final de turno |
| `pad_token_id` | ID especial | Token de relleno |

**¿Por qué temperature = 0.1?**

- **temperature alta (>1)**: Genera más creativo pero menos confiable
- **temperature baja (<0.5)**: Genera determinista y confiable
- 0.1 es muy bajo → respuestas consistentes para un tutor

#### Limpieza de Respuesta

```python
decoded = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
respuesta_limpia = decoded.split("user")[0].split("assistant")[0].strip()
```

- `outputs[0][len(inputs[0]):]`: Solo la parte generada (sin el input)
- `.split("user")[0]`: Corta si aparece "user" (para detener)
- `.split("assistant")[0]`: Corta si aparece "assistant"

#### Prueba Ejemplo

```python
print(evaluar_tutor("¿Qué es la complejidad temporal y por qué es importante en algoritmos?"))
```

Pregunta sobre complejidad temporal → el tutor proporciona explicación.



## Características Técnicas Clave

### 1. **Optimizaciones de Memoria**

- **Cuantización 4-bit**: Reduce modelo de ~32 GB a ~8 GB
- **LoRA**: Solo entrena ~1-2% de parámetros
- **Gradient Accumulation**: Simula batches más grandes
- **bfloat16**: Precisión mixta para eficiencia

### 2. **Fine-tuning Pedagógico**

- Entrenamiento con datos específicos de algoritmos
- Temperatura baja para respuestas consistentes
- Límite de tokens para respuestas concisas

### 3. **Reproducibilidad**

- `seed = 3407`: Permite recrear resultados
- Versiones específicas de librerías fijas

---

## Requisitos del Sistema

- **GPU**: NVIDIA con CUDA (recomendado 16GB VRAM mínimo)
- **Software**: Google Colab (recomendado) o ambiente local con CUDA
- **Datos**: 2 archivos JSONL con pares pregunta-respuesta o dataset en su caso

---

## Casos de Uso

Este tutor puede:

1. **Explicar algoritmos**: Proporcionar definiciones claras
2. **Enseñar complejidad**: O(n), O(n²), recursión, etc.
3. **Resolver problemas**: Ayudar con pseudocódigo
4. **Preguntas pedagógicas**: Adaptar respuestas al nivel del estudiante
5. **Generar ejemplos**: Crear instancias de problemas algorítmicos

---

## Referencias de Parámetros Importantes

### Ajustes para Mayor Precisión
tutor más preciso:

```python
max_steps = 600,              # Entrenar más pasos
learning_rate = 1e-4,         # Tasa de aprendizaje más pequeña
per_device_train_batch_size = 4,  # Batches más grandes (si hay GPU suficiente)
```

### Ajustes para Menor Tiempo/Memoria

```python
max_steps = 100,              # Entrenar menos pasos
lora_dropout = 0.05,          # Añadir regularización
per_device_train_batch_size = 1,  # Batches mínimos
```
