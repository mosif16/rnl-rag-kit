import Foundation

/// RAGManager: Handles indexing and retrieval for RAG (Retrieval-Augmented Generation).
///
/// Local-only pipeline backed by Apple on-device embeddings and LocalRAGStore:
/// - Works fully offline
/// - Uses AppleEmbeddingService (contextual → sentence → word fallback)
/// - Stores chunks and embeddings on-device (optionally synced via CloudKit)
///
public actor RAGManager {
    public static let shared = RAGManager()

    private struct CanonicalDocumentId: Equatable {
        let value: String
        let strategy: Strategy

        enum Strategy: Equatable {
            case explicit
            case sourcePathHash
        }
    }

    public struct SearchResult: Codable, Hashable {
        public let score: Double
        public let chunk_id: String
        public let document_id: String
        public let title: String?
        public let source_type: String
        public let source_path: String?
        public let content: String
        public let start_offset: Int?
        public let end_offset: Int?

        public init(
            score: Double,
            chunk_id: String,
            document_id: String,
            title: String?,
            source_type: String,
            source_path: String?,
            content: String,
            start_offset: Int?,
            end_offset: Int?
        ) {
            self.score = score
            self.chunk_id = chunk_id
            self.document_id = document_id
            self.title = title
            self.source_type = source_type
            self.source_path = source_path
            self.content = content
            self.start_offset = start_offset
            self.end_offset = end_offset
        }
    }

    // MARK: - Public API

    /// Index any RAGIndexable content.
    public func index<T: RAGIndexable>(_ content: T) async -> String? {
        let text = content.ragTextContent()
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }

        return await indexDocument(
            documentID: content.ragDocumentID,
            title: content.ragTitle,
            sourceType: content.ragSourceType.rawValue,
            sourcePath: content.ragSourcePath,
            text: text
        )
    }

    /// Delete indexed content by its document identifier.
    public func delete<T: RAGIndexable>(_ content: T) async -> Bool {
        return await deleteDocument(documentID: content.ragDocumentID)
    }

    /// Search with optional source type filtering.
    public func search(query: String, sourceTypes: [RAGSourceType]?) async -> [SearchResult] {
        let filters: [String: Any]? = {
            guard let sourceTypes, !sourceTypes.isEmpty else { return nil }
            if sourceTypes.count == 1, let first = sourceTypes.first {
                return ["source_type": first.rawValue]
            }
            return nil
        }()

        let results = await search(query: query, filters: filters, minScore: nil)
        guard let sourceTypes, sourceTypes.count > 1 else { return results }

        let allowed = Set(sourceTypes.map { $0.rawValue })
        return results.filter { allowed.contains($0.source_type) }
    }

    /// Search top-k chunks for a query. Returns [] on failure for graceful fallback.
    /// Conformance to AINotesSearching protocol (exact signature match required).
    public func search(query: String, topK: Int, filters: [String: Any]?) async -> [SearchResult] {
        return await search(query: query, topK: topK, filters: filters, minScore: nil)
    }

    /// Search top-k chunks for a query with optional minimum score threshold.
    /// - Parameters:
    ///   - query: Search query text
    ///   - topK: Maximum number of results to return
    ///   - filters: Optional filters (source_type, document_id)
    ///   - minScore: Minimum similarity score threshold (local RAG only, default: 0.25)
    public func search(query: String, topK: Int = 6, filters: [String: Any]? = nil, minScore: Double?) async -> [SearchResult] {
        return await search(
            query: query,
            topK: topK,
            filters: filters,
            minScore: minScore,
            enableQueryExpansion: true,
            enableHybridSearch: true
        )
    }

    /// Search with full control over advanced features
    /// - Parameters:
    ///   - query: Search query text
    ///   - topK: Maximum number of results to return
    ///   - filters: Optional filters (source_type, document_id)
    ///   - minScore: Minimum similarity score threshold (local RAG only, default: 0.25)
    ///   - enableQueryExpansion: Enable synonym/related term expansion for better recall
    ///   - enableHybridSearch: Combine embedding with keyword matching (BM25-like)
    public func search(
        query: String,
        topK: Int = 6,
        filters: [String: Any]? = nil,
        minScore: Double?,
        enableQueryExpansion: Bool,
        enableHybridSearch: Bool
    ) async -> [SearchResult] {
        guard FeatureFlags.RAGEnabled else { return [] }

        return await searchLocal(
            query: query,
            topK: topK,
            filters: filters,
            minScore: minScore,
            enableQueryExpansion: enableQueryExpansion,
            enableHybridSearch: enableHybridSearch
        )
    }

    /// Index a document by chunking and embedding.
    /// Returns document_id or nil on failure.
    public func indexDocument(
        documentID: String? = nil,
        title: String,
        sourceType: String,
        sourcePath: String?,
        text: String
    ) async -> String? {
        guard FeatureFlags.RAGEnabled else { return nil }

        let canonical = canonicalLocalDocumentID(
            documentID: documentID,
            sourceType: sourceType,
            sourcePath: sourcePath
        )

        let indexedId = await indexLocal(
            documentID: canonical?.value,
            title: title,
            sourceType: sourceType,
            sourcePath: sourcePath,
            text: text
        )

        if let canonical, canonical.strategy == .sourcePathHash,
           let sourcePath,
           let indexedId {
            await LocalRAGStore.shared.deleteDocuments(
                sourceType: sourceType,
                sourcePath: sourcePath,
                excludingDocumentId: indexedId
            )
        }

        return indexedId
    }

    /// Delete a document and its chunks.
    public func deleteDocument(documentID: String) async -> Bool {
        guard FeatureFlags.RAGEnabled else { return false }

        return await LocalRAGStore.shared.deleteDocument(documentId: documentID)
    }

    // MARK: - Local RAG Implementation

    private func searchLocal(
        query: String,
        topK: Int,
        filters: [String: Any]?,
        minScore: Double?,
        enableQueryExpansion: Bool = true,
        enableHybridSearch: Bool = true
    ) async -> [SearchResult] {
        // Convert filters to string-based format for LocalRAGStore
        var stringFilters: [String: String]? = nil
        if let filters = filters {
            stringFilters = [:]
            for (key, value) in filters {
                if let stringValue = value as? String {
                    stringFilters?[key] = stringValue
                }
            }
        }

        let results = await LocalRAGStore.shared.searchCompatible(
            query: query,
            topK: topK,
            filters: stringFilters,
            minScore: minScore,
            enableQueryExpansion: enableQueryExpansion,
            enableHybridSearch: enableHybridSearch
        )

        if FeatureFlags.RAGDebugLogging {
            if let first = results.first {
                DiagnosticsLogger.log("[RAG-Local] search ok count=\(results.count) top1(score=\(String(format: "%.4f", first.score))) doc=\(first.document_id) queryExpansion=\(enableQueryExpansion) hybrid=\(enableHybridSearch)")
            } else {
                DiagnosticsLogger.log("[RAG-Local] search ok count=0")
            }
        }

        return results
    }

    private func indexLocal(
        documentID: String?,
        title: String,
        sourceType: String,
        sourcePath: String?,
        text: String
    ) async -> String? {
        if FeatureFlags.RAGDebugLogging {
            let idHint = documentID ?? "<auto>"
            DiagnosticsLogger.log("[RAG-Local] index start docId=\(idHint) title=\(title) type=\(sourceType) textLen=\(text.count)")
        }

        let documentId = await LocalRAGStore.shared.indexDocument(
            documentId: documentID,
            title: title,
            sourceType: sourceType,
            sourcePath: sourcePath,
            text: text
        )

        if FeatureFlags.RAGDebugLogging {
            if let docId = documentId {
                DiagnosticsLogger.log("[RAG-Local] index success docId=\(docId)")
            } else {
                DiagnosticsLogger.log("[RAG-Local] index failed")
            }
        }

        return documentId
    }

    private func canonicalLocalDocumentID(
        documentID: String?,
        sourceType: String,
        sourcePath: String?
    ) -> CanonicalDocumentId? {
        if let explicit = documentID?.trimmingCharacters(in: .whitespacesAndNewlines), !explicit.isEmpty {
            return CanonicalDocumentId(value: explicit, strategy: .explicit)
        }

        guard let sourcePath = sourcePath?.trimmingCharacters(in: .whitespacesAndNewlines), !sourcePath.isEmpty else {
            return nil
        }

        let hash = DeduplicationService.hash("\(sourceType)|\(sourcePath)")
        return CanonicalDocumentId(value: "\(sourceType)_\(hash)", strategy: .sourcePathHash)
    }

    // MARK: - Diagnostics

    /// Get diagnostic info about RAG and embedding configuration
    public func getDiagnostics() async -> [String: Any] {
        let embeddingService = AppleEmbeddingService.shared
        let embeddingDiagnostics = await embeddingService.getDiagnostics()

        var diagnostics: [String: Any] = [
            "ragEnabled": FeatureFlags.RAGEnabled,
            "appleEmbedding": embeddingDiagnostics,
            "mode": "local"
        ]

        let localStats = await LocalRAGStore.shared.getStatistics()
        diagnostics["localRAGStats"] = localStats

        return diagnostics
    }
}
