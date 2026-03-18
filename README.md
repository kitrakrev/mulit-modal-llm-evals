# LLM Evaluation Framework

<div align="center">

```
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║     ██╗     ██╗     ███╗   ███╗    ███████╗██╗   ██╗ █████╗ ██╗      ║
║     ██║     ██║     ████╗ ████║    ██╔════╝██║   ██║██╔══██╗██║      ║
║     ██║     ██║     ██╔████╔██║    █████╗  ██║   ██║███████║██║      ║
║     ██║     ██║     ██║╚██╔╝██║    ██╔══╝  ╚██╗ ██╔╝██╔══██║██║      ║
║     ███████╗███████╗██║ ╚═╝ ██║    ███████╗ ╚████╔╝ ██║  ██║███████╗ ║
║     ╚══════╝╚══════╝╚═╝     ╚═╝    ╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝ ║
║                                                                       ║
║     Multi-model evaluation framework with RAG, voice, tracing,       ║
║     regression detection, and OpenTelemetry observability             ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
```

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![MLX](https://img.shields.io/badge/MLX-Local_Inference-000000?style=flat-square&logo=apple)](https://ml-explore.github.io/mlx/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6F00?style=flat-square)](https://www.trychroma.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker)](https://docker.com)

</div>

---

## 🎯 What is this? (Beginner Friendly)
How do we know if an AI like ChatGPT or Claude is actually giving good answers? This project is an **LLM Evaluation Framework**! It runs automated tests on multiple AI models using RAG (Retrieval-Augmented Generation) and voice inputs, scoring them on faithfulness, relevance, and hallucinations so we always pick the best model for the job.

![Architecture Animation](https://raw.githubusercontent.com/karthikraja/llm-eval/main/docs/assets/architecture.gif)
*Note: The GIF above illustrates how the evaluation engine coordinates RAG pipelines, voice models, and scoring metrics to produce a detailed performance report.*

---

## Architecture

```mermaid
flowchart TB
    subgraph EvalEngine["🧪 Evaluation Engine"]
        direction TB
        ER[Eval Runner<br/>Orchestrates eval runs]
        CE[Comparison Engine<br/>Multi-model ranking]
        RD[Regression Detector<br/>Baseline comparison]
    end

    subgraph Metrics["📏 Metric Suite"]
        direction LR
        subgraph TextMetrics["Text Metrics"]
            F[Faithfulness<br/>F1 Score]
            R[Relevance<br/>Token Overlap]
            H[Hallucination<br/>Sentence NLI]
            L[Latency]
            C[Cost]
        end
        subgraph VoiceMetrics["Voice Metrics"]
            WER[STT Accuracy<br/>WER · CER]
            MOS[TTS Quality<br/>MOS · Naturalness]
            VL[Voice Latency]
        end
    end

    subgraph ModelProviders["🤖 Model Providers"]
        direction LR
        OAI[OpenAI<br/>GPT-4o]
        ANT[Anthropic<br/>Claude Sonnet]
        MLX[MLX Local<br/>Llama 3.2]
        CUS[Custom<br/>OpenAI-compat]
    end

    subgraph RAG["📚 RAG Pipeline"]
        direction LR
        CH[Chunker<br/>Token/Separator]
        EMB[Embeddings<br/>Hash/Custom]
        VR[Vector Retriever<br/>Cosine Similarity]
        CTX[Context Builder]
    end

    subgraph Voice["🎤 Voice Pipeline"]
        direction LR
        STT[STT Runner<br/>MLX Whisper / OpenAI]
        TTS[TTS Runner<br/>Local / OpenAI]
    end

    subgraph Tracing["📊 Distributed Tracing"]
        direction LR
        OT[OTel SDK<br/>@trace_eval]
        KE[Kafka Exporter]
        LP[Log Pipeline<br/>traces.spans topic]
    end

    subgraph Storage["💾 Storage"]
        direction LR
        PG[(PostgreSQL<br/>Eval runs + results)]
        CR[(ChromaDB<br/>Document vectors)]
    end

    subgraph API["🔍 FastAPI"]
        direction LR
        A1[POST /api/evals/run]
        A2[GET /api/evals/compare]
        A3[GET /api/regressions]
        A4[GET /api/models]
    end

    ER --> Metrics
    ER --> ModelProviders
    ER --> RAG
    ER --> Voice
    CE --> ER
    RD --> ER

    RAG --> CR
    ER --> PG
    ER --> Tracing
    KE --> LP

    API --> EvalEngine
    API --> Storage

    style EvalEngine fill:#e1f5fe,stroke:#0288d1
    style Metrics fill:#fff3e0,stroke:#f57c00
    style ModelProviders fill:#e8f5e9,stroke:#388e3c
    style RAG fill:#f3e5f5,stroke:#7b1fa2
    style Voice fill:#fce4ec,stroke:#c62828
    style Tracing fill:#e0f2f1,stroke:#00695c
    style Storage fill:#f5f5f5,stroke:#616161
    style API fill:#e0f7fa,stroke:#00838f
```

## Evaluation Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant API as FastAPI
    participant ER as EvalRunner
    participant MP as ModelProvider
    participant RAG as RAG Pipeline
    participant M as Metrics
    participant DB as PostgreSQL
    participant OT as OTel/Kafka

    C->>API: POST /api/evals/run
    API->>ER: Start eval run

    loop Each sample in dataset
        ER->>RAG: Retrieve context (if RAG mode)
        RAG-->>ER: Relevant chunks

        ER->>MP: Generate response
        MP-->>ER: Model output + latency

        loop Each metric
            ER->>M: Compute metric
            M-->>ER: MetricResult (score, metadata)
        end

        ER->>OT: Export trace span
    end

    ER->>DB: Store eval run results
    ER-->>API: EvalRunResult
    API-->>C: JSON response

    Note over C,OT: Comparison Mode
    C->>API: GET /api/evals/compare
    API->>ER: Run N models on same dataset
    ER-->>API: Ranked comparison
    API-->>C: Model rankings + deltas
```

## Evaluation Modes

```mermaid
flowchart LR
    subgraph Modes
        direction TB
        T[text<br/>Standard LLM eval]
        RAG_M[rag<br/>Retrieval + generation]
        STT[voice-stt<br/>Speech-to-text accuracy]
        TTS_M[voice-tts<br/>Text-to-speech quality]
        E2E[voice-e2e<br/>End-to-end voice]
    end

    subgraph TextPipeline
        G[Generate] --> ME[Measure<br/>Faithfulness · Relevance<br/>Hallucination · Latency · Cost]
    end

    subgraph RAGPipeline
        R[Retrieve] --> B[Build Context] --> G2[Generate] --> ME2[Measure<br/>+ Retrieval Precision]
    end

    subgraph VoicePipeline
        A[Audio In] --> TR[Transcribe] --> ME3[Measure<br/>WER · CER · MOS<br/>Naturalness · Latency]
    end

    T --> TextPipeline
    RAG_M --> RAGPipeline
    STT & TTS_M & E2E --> VoicePipeline

    style Modes fill:#e1f5fe
    style TextPipeline fill:#e8f5e9
    style RAGPipeline fill:#f3e5f5
    style VoicePipeline fill:#fff3e0
```

## Project Structure

```
llm-eval/
├── eval/
│   ├── runner.py                  # EvalRunner orchestrator
│   ├── comparison.py              # Multi-model comparison engine
│   ├── regression.py              # Regression detection + alerts
│   ├── metrics/
│   │   ├── base.py                # BaseMetric, MetricResult, EvalMode
│   │   ├── faithfulness.py        # F1-based faithfulness scoring
│   │   ├── relevance.py           # Token overlap relevance
│   │   ├── hallucination.py       # Sentence-level NLI detection
│   │   ├── latency.py             # Response latency tracking
│   │   ├── cost.py                # Token cost calculation
│   │   └── voice/
│   │       ├── stt_accuracy.py    # WER, CER metrics
│   │       ├── tts_quality.py     # MOS, naturalness scoring
│   │       └── voice_latency.py   # Voice pipeline latency
│   └── tests/test_metrics.py      # Metric tests
├── models/
│   ├── registry.py                # ModelRegistry + YAML config loader
│   ├── providers/
│   │   ├── base.py                # BaseProvider abstract class
│   │   ├── openai_provider.py     # OpenAI GPT provider
│   │   ├── anthropic_provider.py  # Anthropic Claude provider
│   │   ├── mlx_provider.py        # MLX local inference
│   │   └── custom_provider.py     # OpenAI-compatible endpoints
│   └── tests/test_registry.py     # Registry tests
├── rag/
│   ├── pipeline.py                # RAGPipeline (index → retrieve → generate)
│   ├── retriever.py               # Vector retriever + cosine similarity
│   ├── embeddings.py              # Embedding providers
│   ├── chunker.py                 # Token/separator-based chunking
│   └── tests/test_pipeline.py     # RAG pipeline tests
├── tracing/
│   ├── otel_setup.py              # OTel SDK initialization
│   ├── span_decorators.py         # @trace_eval decorator (sync + async)
│   ├── kafka_exporter.py          # Trace export to Kafka
│   └── tests/test_tracing.py      # Tracing tests
├── voice/
│   ├── stt_runner.py              # Speech-to-text orchestrator
│   ├── tts_runner.py              # Text-to-speech orchestrator
│   ├── audio_utils.py             # Audio loading, resampling, conversion
│   ├── providers/
│   │   ├── mlx_whisper.py         # Local MLX Whisper STT
│   │   ├── openai_whisper.py      # OpenAI Whisper API STT
│   │   ├── openai_tts.py          # OpenAI TTS API
│   │   └── local_tts.py           # Local TTS engine
│   └── tests/test_voice.py        # Voice pipeline tests
├── datasets/
│   ├── loader.py                  # JSON dataset loader
│   ├── voice_loader.py            # Audio dataset loader
│   └── samples/qa_sample.json     # Sample Q&A dataset
├── api/
│   ├── main.py                    # FastAPI application
│   ├── models.py                  # Pydantic request/response models
│   ├── routes/
│   │   ├── evals.py               # POST /api/evals/run, GET /api/evals/{id}
│   │   ├── models.py              # POST /api/models/register, GET /api/models
│   │   ├── compare.py             # GET /api/evals/compare
│   │   ├── regressions.py         # GET /api/regressions
│   │   └── health.py              # GET /health
│   └── tests/test_api.py          # API route tests
├── db/
│   ├── models.py                  # SQLAlchemy ORM models
│   ├── init.sql                   # Database schema
│   └── migrations/                # Alembic versioning
├── tests/
│   ├── test_integration.py        # End-to-end eval pipeline test
│   ├── test_load.py               # Locust: concurrent eval runs
│   └── test_stress.py             # Locust: large-scale eval stress
├── docker-compose.yml             # Eval + ChromaDB + PostgreSQL
├── docker-compose.full.yml        # + Log Pipeline stack
├── docker-compose.monitoring.yml  # + OTel + Prometheus + Grafana
├── Dockerfile
├── Makefile
├── model-registry.yaml            # Model configuration
└── pyproject.toml
```

## Quick Start

```bash
# Start eval stack: service + ChromaDB + PostgreSQL
make up

# Run evaluation
make eval

# Multi-model comparison
make compare

# Start with log-pipeline integration
make up-full

# Full monitoring stack
make up-monitoring
```

## Model Registry

Configure models in `model-registry.yaml`:

```yaml
models:
  gpt-4o:
    provider: openai
    model_id: gpt-4o
    type: [text, vision]
  claude-sonnet:
    provider: anthropic
    model_id: claude-sonnet-4-20250514
    type: [text]
  llama-3.2-1b-local:
    provider: mlx
    model_id: mlx-community/Llama-3.2-1B-Instruct
    type: [text]
  whisper-local:
    provider: mlx
    model_id: mlx-community/whisper-large-v3
    type: [stt]
  custom-endpoint:
    provider: custom
    base_url: http://localhost:11434/v1
    model_id: my-model
    type: [text]
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/evals/run` | Trigger eval run `{models, dataset, metrics, mode}` |
| `GET` | `/api/evals/{run_id}` | Get eval run results |
| `GET` | `/api/evals/compare` | Multi-model comparison |
| `GET` | `/api/regressions` | Regression alerts |
| `POST` | `/api/models/register` | Register a new model |
| `GET` | `/api/models` | List registered models |
| `GET` | `/health` | Service health check |
| `GET` | `/metrics` | Prometheus scrape endpoint |

## Tracing Integration with Log Pipeline

```mermaid
flowchart LR
    subgraph LLMEval["LLM Eval Service"]
        TD["@trace_eval<br/>decorated functions"]
        OS[OTel SDK]
    end

    subgraph Transport
        OC[OTel Collector]
        KF[Kafka<br/>traces.spans]
    end

    subgraph LogPipeline["Log Pipeline"]
        FA[Faust Agent]
        PG[(PostgreSQL<br/>trace_spans)]
        API[Query API]
    end

    TD --> OS --> OC --> KF --> FA --> PG --> API

    style LLMEval fill:#e1f5fe
    style Transport fill:#fff3e0
    style LogPipeline fill:#e8f5e9
```

Eval functions decorated with `@trace_eval` automatically export spans through the OTel Collector to Kafka, where the Log Pipeline's Faust worker processes and stores them in PostgreSQL. Query traces via: `GET /api/traces?service=llm-eval`

## Regression Detection

```mermaid
flowchart TD
    subgraph Current["Current Run"]
        CR[Eval Run<br/>Model X, Dataset Y]
    end

    subgraph Baseline["Baseline"]
        BL[Previous Best<br/>or Pinned Baseline]
    end

    subgraph Detection
        CMP{Compare<br/>metric-by-metric}
        TH{Degradation ><br/>threshold?}
    end

    subgraph Output
        OK[✅ Pass<br/>No regression]
        ALERT[🚨 Regression Alert<br/>metric, delta, severity]
    end

    CR --> CMP
    BL --> CMP
    CMP --> TH
    TH -->|No| OK
    TH -->|Yes| ALERT

    style Current fill:#e1f5fe
    style Baseline fill:#e8f5e9
    style Detection fill:#fff3e0
    style Output fill:#fce4ec
```

## Makefile Targets

```bash
make up               # Eval + ChromaDB + PostgreSQL
make up-full          # + Log Pipeline stack (Kafka, Faust, API)
make up-monitoring    # + OTel Collector + Prometheus + Grafana
make down             # Tear down everything
make eval             # Run evaluation suite
make eval-voice       # Run voice evaluation suite
make compare          # Multi-model comparison
make register-model   # Add model to registry
make test             # Unit tests (pytest)
make test-load        # Locust: concurrent evals
make test-stress      # Locust: large-scale eval
make dev              # API with auto-reload
make seed             # Load sample eval datasets
make migrate          # Alembic migrations
make lint             # ruff + mypy
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection | `postgresql://...` |
| `CHROMA_HOST` | ChromaDB host | `localhost` |
| `CHROMA_PORT` | ChromaDB port | `8000` |
| `OPENAI_API_KEY` | OpenAI API key | — |
| `ANTHROPIC_API_KEY` | Anthropic API key | — |
| `VOICE_PROVIDER` | Voice engine: `local\|api\|auto` | `auto` |
| `OTEL_ENDPOINT` | OTel Collector endpoint | `http://localhost:4317` |
| `KAFKA_BROKER` | Kafka broker (for tracing) | `localhost:9092` |
| `LOG_PIPELINE_API` | Log Pipeline API URL | `http://localhost:8000` |

## License

MIT
