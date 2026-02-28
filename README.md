# RAGKit (`rnl-rag-kit`)

On-device RAG, embeddings, chunking, and retrieval for Apple apps.

## Scope

This package contains the reusable local RAG core:

- `RAGManager`
- `LocalRAGStore`
- `AppleEmbeddingService`
- `RAGChunker`
- `SentenceChunker`
- `AggressiveChunker`
- `RAGContextProvider`
- `RAGSourceType`
- `RAGIndexable`

## What Moved

- Local chunking, embedding, indexing, search, re-ranking, and retrieval orchestration.
- RAG context formatting and retrieval support for generation pipelines.
- Chunking tests (`RAGChunker`, `SentenceChunker`, config behavior).

## What Stayed In App / Adapters

- CloudKit RAG sync implementation (`RAGCloudKitStore`) remains adapter-side.
- App lifecycle/background bootstrapping (`RAGMigrationManager`, `BackgroundRAGIndexer`) remains host-side.
- Domain-model-specific `RAGIndexable` conformances remain with domain models.

## Integration Notes

- Add package dependency: `https://github.com/mosif16/rnl-rag-kit`
- Import module: `import RAGKit`
- Configure runtime flags via `RAGKitConfiguration` as needed.
- Optional cloud sync bridge:
  - Set `RAGKitConfiguration.cloudSyncEnabled = true`
  - Provide callback through `LocalRAGStore.shared.setCloudSyncHandler { ... }`

## Source Compatibility Notes

- `RAGIndexable` and `RAGSourceType` are public and intended for host model conformance.
- Local store persistence format is preserved from the app extraction slice (documents + metadata JSON, binary quantized embeddings).
