import Accelerate
import Foundation

/// LocalRAGStore: Fully on-device RAG storage and retrieval.
/// Replaces server-side vector databases with local JSON storage and in-memory similarity search.
///
/// Features:
/// - **Binary storage with Int8 quantization** - 75% smaller than JSON float storage
/// - **Lazy loading** - Only metadata loaded on init, embeddings loaded on-demand
/// - **Incremental saves** - Only dirty chunks are written to disk
/// - **Score threshold filtering** - Filter out low-similarity results for better quality
/// - In-memory ANN index for fast similarity search
/// - Cosine similarity using Apple NLEmbedding vectors
/// - No network dependency - works fully offline
///
/// Storage Format:
/// - Documents: JSON file (small, always loaded)
/// - Chunk metadata: JSON file (no embeddings, always loaded)
/// - Embeddings: Binary files per document using Int8 quantization
///
/// Usage:
/// ```swift
/// // Index a document
/// let docId = await LocalRAGStore.shared.indexDocument(
///     title: "My Notes",
///     sourceType: "transcript",
///     sourcePath: "/path/to/file.txt",
///     text: "Full text content..."
/// )
///
/// // Search with minimum score threshold
/// let results = await LocalRAGStore.shared.search(query: "search term", topK: 5, minScore: 0.3)
/// ```
public actor LocalRAGStore {
    public static let shared = LocalRAGStore()

    /// Optional host-provided cloud sync callback, kept out of core storage logic.
    private var cloudSyncHandler: (@Sendable (String) async throws -> Void)?

    // MARK: - Data Models

    public struct Document: Codable, Identifiable {
        public let id: String
        public let title: String
        public let sourceType: String
        public let sourcePath: String
        public let createdAt: Date
        public var updatedAt: Date
        public var embeddingMetadata: EmbeddingMetadata?
    }

    /// Metadata describing how embeddings were generated for a document.
    public struct EmbeddingMetadata: Codable {
        public let version: String
        public let dimension: Int
        public let language: String
        public let strategy: String?
        public let modelIdentifier: String?
        public let createdAt: Date?
    }

    /// Chunk metadata stored in JSON (no embedding - stored separately in binary)
    public struct ChunkMetadata: Codable, Identifiable {
        public let id: String
        public let documentId: String
        public let index: Int
        public let content: String
        public let startOffset: Int
        public let endOffset: Int
        public let createdAt: Date
    }

    /// Full chunk with embedding for in-memory use
    public struct Chunk: Identifiable {
        public let id: String
        public let documentId: String
        public let index: Int
        public let content: String
        public let startOffset: Int
        public let endOffset: Int
        public let embedding: [Float]
        public let createdAt: Date

        public init(metadata: ChunkMetadata, embedding: [Float]) {
            self.id = metadata.id
            self.documentId = metadata.documentId
            self.index = metadata.index
            self.content = metadata.content
            self.startOffset = metadata.startOffset
            self.endOffset = metadata.endOffset
            self.embedding = embedding
            self.createdAt = metadata.createdAt
        }
    }

    public struct SearchResult: Hashable {
        public let score: Double
        public let chunkId: String
        public let documentId: String
        public let title: String?
        public let sourceType: String
        public let sourcePath: String?
        public let content: String
        public let startOffset: Int?
        public let endOffset: Int?
    }

    // MARK: - Quantization

    /// Quantization parameters for Int8 encoding
    public struct QuantizationParams: Codable {
        public let minValue: Float
        public let maxValue: Float
        public let dimension: Int

        /// Scale factor: (max - min) / 255
        public var scale: Float { (maxValue - minValue) / 255.0 }
    }

    // MARK: - Storage

    private var documents: [String: Document] = [:]
    private var chunkMetadata: [String: ChunkMetadata] = [:]
    private var documentChunks: [String: [String]] = [:] // documentId -> [chunkId]

    // Lazy-loaded embeddings (loaded on-demand per document)
    private var loadedEmbeddings: [String: [Float]] = [:] // chunkId -> embedding
    private var loadedDocumentEmbeddings: Set<String> = [] // documentIds with loaded embeddings
    private var loadedDocumentOrder: [String] = [] // LRU order for eviction
    private let maxLoadedDocuments = 50 // Evict oldest when exceeded
    private var quantizationParams: [String: QuantizationParams] = [:] // documentId -> params

    // Dirty tracking for incremental saves
    private var dirtyDocumentIds: Set<String> = []
    private var dirtyMetadata: Bool = false
    private var didValidateEmbeddingVersions = false

    // Index deduplication (prevents wasteful repeat indexing triggered by UI/lifecycle)
    private var inFlightIndexTasks: [String: Task<String?, Never>] = [:]
    private var recentIndexCompletions: [String: Date] = [:]
    private let duplicateIndexWindow: TimeInterval = 10.0
    private let recentIndexRetention: TimeInterval = 120.0
    private let recentIndexMaxEntries: Int = 200

    // Pre-computed normalized embeddings cache
    private var normalizedEmbeddings: [String: [Float]] = [:]

    // ANN index (coarse centroid clusters)
    private var clusterCentroids: [[Float]] = []
    private var annClusters: [[String]] = []
    private var clusterSizes: [Int] = []
    private var embeddingDimension: Int?
    private let maxAnnClusters = 8
    private let clusterSelectionCount = 3

    // MARK: - Search Cache

    private struct SearchCacheEntry {
        let results: [SearchResult]
        let createdAt: Date
    }

    private var searchCache: [String: SearchCacheEntry] = [:]
    private var searchCacheOrder: [String] = []
    private let searchCacheMaxEntries = 96
    private let searchCacheTTL: TimeInterval = 90

    // Default minimum similarity score threshold
    private let defaultMinScore: Double = 0.25

    // Re-ranking configuration
    private let coarseTopK: Int = 50 // First-pass retrieval count
    private let reRankEnabled: Bool = true

    private let storageURL: URL
    private let documentsFileName = "rag_documents.json"
    private let metadataFileName = "rag_metadata.json"
    private let embeddingsDir = "embeddings"
    private let legacyChunksFileName = "rag_chunks.json" // For migration

    // MARK: - Embedding Metadata Management

    /// Build a descriptor for the currently loaded embedding model.
    private func currentEmbeddingMetadata() async -> EmbeddingMetadata? {
        let embeddingService = AppleEmbeddingService.shared
        guard await embeddingService.isAvailable else { return nil }
        let info = await embeddingService.currentModelInfo

        return EmbeddingMetadata(
            version: info.version,
            dimension: info.dimension,
            language: info.languageCode,
            strategy: info.strategy.rawValue,
            modelIdentifier: info.modelIdentifier,
            createdAt: Date()
        )
    }

    /// Determine whether persisted embeddings need to be regenerated.
    private func shouldReindex(document: Document, currentMetadata: EmbeddingMetadata) -> Bool {
        guard let existing = document.embeddingMetadata else { return true }
        if existing.version != currentMetadata.version { return true }
        if existing.dimension != currentMetadata.dimension { return true }
        if existing.language != currentMetadata.language { return true }
        if (existing.strategy ?? "") != currentMetadata.strategy { return true }
        return false
    }

    private func validateEmbeddingVersions() async {
        guard !didValidateEmbeddingVersions else { return }
        guard await currentEmbeddingMetadata() != nil else { return }
        didValidateEmbeddingVersions = true
        await reconcileEmbeddingMetadataIfNeeded()
    }

    // MARK: - Initialization

    private init() {
        // Use app's documents directory for persistent storage
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        self.storageURL = documentsPath.appendingPathComponent("LocalRAG", isDirectory: true)

        // Create directories if needed
        try? FileManager.default.createDirectory(at: storageURL, withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(
            at: storageURL.appendingPathComponent(embeddingsDir),
            withIntermediateDirectories: true
        )

        // Load existing data (metadata only - lazy load embeddings)
        Task {
            await loadMetadataFromDisk()
            await validateEmbeddingVersions()
        }
    }

    // MARK: - Public API

    /// Index a document by chunking and embedding locally
    /// - Returns: Document ID or nil on failure
    public func indexDocument(documentId: String? = nil, title: String, sourceType: String, sourcePath: String?, text: String) async -> String? {
        await validateEmbeddingVersions()
        clearSearchCache()

        let trimmedExplicitId = documentId?.trimmingCharacters(in: .whitespacesAndNewlines)
        let explicitId = (trimmedExplicitId?.isEmpty == false) ? trimmedExplicitId : nil

        if let explicitId {
            let contentHash = DeduplicationService.hash(text)
            let dedupeKey = "\(explicitId):\(contentHash)"
            let now = Date()

            if let completedAt = recentIndexCompletions[dedupeKey],
               now.timeIntervalSince(completedAt) < duplicateIndexWindow {
                if FeatureFlags.RAGDebugLogging {
                    DiagnosticsLogger.log("[LocalRAG] Skip duplicate index docId=\(explicitId) type=\(sourceType) hash=\(contentHash)")
                }
                return explicitId
            }

            if let existingTask = inFlightIndexTasks[dedupeKey] {
                return await existingTask.value
            }

            let task = Task { [title, sourceType, sourcePath, text] in
                await self.indexDocumentUncached(
                    documentId: explicitId,
                    title: title,
                    sourceType: sourceType,
                    sourcePath: sourcePath,
                    text: text
                )
            }

            inFlightIndexTasks[dedupeKey] = task
            defer { inFlightIndexTasks[dedupeKey] = nil }

            let result = await task.value
            if result != nil {
                recentIndexCompletions[dedupeKey] = Date()
                pruneRecentIndexCompletions(now: Date())
            }
            return result
        }

        return await indexDocumentUncached(
            documentId: documentId,
            title: title,
            sourceType: sourceType,
            sourcePath: sourcePath,
            text: text
        )
    }

    private func pruneRecentIndexCompletions(now: Date) {
        if recentIndexCompletions.count > recentIndexMaxEntries {
            let sorted = recentIndexCompletions.sorted { $0.value < $1.value }
            let overflow = sorted.prefix(recentIndexCompletions.count - recentIndexMaxEntries)
            for (key, _) in overflow {
                recentIndexCompletions[key] = nil
            }
        }

        for (key, timestamp) in recentIndexCompletions {
            if now.timeIntervalSince(timestamp) > recentIndexRetention {
                recentIndexCompletions[key] = nil
            }
        }
    }

    private func indexDocumentUncached(documentId: String? = nil, title: String, sourceType: String, sourcePath: String?, text: String) async -> String? {
        let startTime = Date()

        // Chunk the text
        let textChunks = RAGChunker.chunk(text: text)
        guard !textChunks.isEmpty else {
            if FeatureFlags.AppleEmbeddingDebugLogging {
                DiagnosticsLogger.log("[LocalRAG] No chunks generated for document: \(title)")
            }
            return nil
        }

        // Generate embeddings for all chunks
        let embeddingService = AppleEmbeddingService.shared
        guard await embeddingService.isAvailable else {
            DiagnosticsLogger.log("[LocalRAG] Apple embedding service not available")
            return nil
        }

        let chunkTexts = textChunks.map { $0.text }
        let embeddings = await embeddingService.embedBatch(texts: chunkTexts)

        let resolvedDocumentId: String = {
            if let explicit = documentId?.trimmingCharacters(in: .whitespacesAndNewlines),
               !explicit.isEmpty {
                return explicit
            }
            return UUID().uuidString
        }()

        let createdAt = documents[resolvedDocumentId]?.createdAt ?? Date()
        let updatedAt = Date()

        // Create chunks with embeddings (staged; committed after we know we have embeddings)
        var stagedChunkMetadata: [String: ChunkMetadata] = [:]
        var stagedChunkIds: [String] = []
        var stagedEmbeddings: [String: [Float]] = [:]
        var stagedNormalized: [String: [Float]] = [:]
        var successCount = 0

        for (i, textChunk) in textChunks.enumerated() {
            guard let embedding = embeddings[i] else {
                if FeatureFlags.AppleEmbeddingDebugLogging {
                    DiagnosticsLogger.log("[LocalRAG] Failed to generate embedding for chunk \(i)")
                }
                continue
            }

            let chunkId = chunkIdForDocument(documentId: resolvedDocumentId, chunkIndex: textChunk.index)
            let metadata = ChunkMetadata(
                id: chunkId,
                documentId: resolvedDocumentId,
                index: textChunk.index,
                content: textChunk.text,
                startOffset: textChunk.startOffset,
                endOffset: textChunk.endOffset,
                createdAt: updatedAt
            )

            stagedChunkMetadata[chunkId] = metadata
            stagedEmbeddings[chunkId] = embedding
            if let normalized = normalizeVector(embedding) {
                stagedNormalized[chunkId] = normalized
            }

            stagedChunkIds.append(chunkId)
            successCount += 1
        }

        // Only save if we got at least some embeddings
        guard successCount > 0 else {
            DiagnosticsLogger.log("[LocalRAG] No embeddings generated, aborting index")
            return nil
        }

        let modelInfo = await embeddingService.currentModelInfo
        let embeddingMetadata = EmbeddingMetadata(
            version: modelInfo.version,
            dimension: modelInfo.dimension,
            language: modelInfo.languageCode,
            strategy: modelInfo.strategy.rawValue,
            modelIdentifier: modelInfo.modelIdentifier,
            createdAt: updatedAt
        )

        let didReplaceExisting = documents[resolvedDocumentId] != nil
        if didReplaceExisting {
            purgeDocumentInMemory(documentId: resolvedDocumentId, shouldDeleteEmbeddingFile: false)
        }

        documents[resolvedDocumentId] = Document(
            id: resolvedDocumentId,
            title: title,
            sourceType: sourceType,
            sourcePath: sourcePath ?? "",
            createdAt: createdAt,
            updatedAt: updatedAt,
            embeddingMetadata: embeddingMetadata
        )
        documentChunks[resolvedDocumentId] = stagedChunkIds
        loadedDocumentEmbeddings.insert(resolvedDocumentId)
        loadedDocumentOrder.removeAll { $0 == resolvedDocumentId }
        loadedDocumentOrder.append(resolvedDocumentId)

        for (chunkId, metadata) in stagedChunkMetadata {
            chunkMetadata[chunkId] = metadata
        }
        for chunkId in stagedChunkIds {
            guard let embedding = stagedEmbeddings[chunkId] else { continue }
            loadedEmbeddings[chunkId] = embedding
            if let normalized = stagedNormalized[chunkId] {
                normalizedEmbeddings[chunkId] = normalized
                addToANNIndex(chunkId: chunkId, normalizedEmbedding: normalized)
            }
        }

        // Mark as dirty for incremental save
        dirtyDocumentIds.insert(resolvedDocumentId)
        dirtyMetadata = true

        // Persist to disk (incremental)
        await saveIncrementally()

        if didReplaceExisting {
            await rebuildANNIndex()
        }

        let duration = Date().timeIntervalSince(startTime)
        if FeatureFlags.AppleEmbeddingDebugLogging {
            DiagnosticsLogger.log("[LocalRAG] Indexed document '\(title)' with \(successCount)/\(textChunks.count) chunks in \(String(format: "%.2f", duration))s")
        }

        triggerCloudKitSync(for: resolvedDocumentId)

        return resolvedDocumentId
    }

    /// Search for similar chunks using cosine similarity with two-stage re-ranking
    /// - Parameters:
    ///   - query: Search query text
    ///   - topK: Number of results to return
    ///   - filters: Optional filters (sourceType, documentId)
    ///   - minScore: Minimum similarity score threshold (default: 0.25)
    ///   - enableQueryExpansion: Enable synonym/related term expansion for better recall
    ///   - enableHybridSearch: Combine embedding with keyword matching (BM25-like)
    /// - Returns: Array of search results sorted by similarity score, filtered by minScore
    public func search(
        query: String,
        topK: Int = 6,
        filters: [String: String]? = nil,
        minScore: Double? = nil,
        enableQueryExpansion: Bool = true,
        enableHybridSearch: Bool = true
    ) async -> [SearchResult] {
        let startTime = Date()
        let scoreThreshold = minScore ?? defaultMinScore

        await validateEmbeddingVersions()

        let cacheKey = makeSearchCacheKey(
            query: query,
            topK: topK,
            filters: filters,
            minScore: minScore,
            enableQueryExpansion: enableQueryExpansion,
            enableHybridSearch: enableHybridSearch
        )
        pruneSearchCache(now: startTime)
        if let cached = searchCache[cacheKey], startTime.timeIntervalSince(cached.createdAt) <= searchCacheTTL {
            return cached.results
        }

        // Generate query embedding
        let embeddingService = AppleEmbeddingService.shared
        guard await embeddingService.isAvailable else {
            if FeatureFlags.AppleEmbeddingDebugLogging {
                DiagnosticsLogger.log("[LocalRAG] Embedding service not available")
            }
            return []
        }

        // Query expansion: generate embeddings for expanded queries.
        // Use a batch call to avoid repeated language detection / model switching overhead.
        let expandedQueries = enableQueryExpansion ? expandQuery(query) : [query]
        let rawQueryEmbeddings = await embeddingService.embedBatch(texts: expandedQueries)
        let queryEmbeddings = rawQueryEmbeddings.compactMap { $0 }

        guard !queryEmbeddings.isEmpty else {
            if FeatureFlags.AppleEmbeddingDebugLogging {
                DiagnosticsLogger.log("[LocalRAG] Failed to generate query embeddings")
            }
            return []
        }

        // Normalize all query embeddings
        let normalizedQueries = queryEmbeddings.compactMap { normalizeVector($0) }
        guard let primaryNormalizedQuery = normalizedQueries.first else {
            return []
        }

        // Stage 1: Coarse retrieval - get more candidates than needed
        let coarseK = reRankEnabled ? max(coarseTopK, topK * 3) : topK
        let candidateIds = candidateChunkIds(for: primaryNormalizedQuery)
        await ensureEmbeddingsLoaded(forChunkIds: candidateIds)

        // Calculate embedding-only scores with all query variations.
        // Keyword/BM25 scoring is computed later for a small top slice to keep latency low.
        var scoredChunks: [(metadata: ChunkMetadata, embeddingScore: Double)] = []

        for chunkId in candidateIds {
            guard let metadata = chunkMetadata[chunkId] else { continue }

            // Apply filters
            if let filters = filters {
                if let sourceTypeFilter = filters["source_type"] {
                    guard let doc = documents[metadata.documentId],
                          doc.sourceType == sourceTypeFilter else { continue }
                }
                if let documentIdFilter = filters["document_id"],
                   metadata.documentId != documentIdFilter { continue }
            }

            guard let embedding = loadedEmbeddings[chunkId] else { continue }

            // Calculate max similarity across all expanded queries (improves recall)
            var maxEmbeddingSim: Double = 0
            for normalizedQuery in normalizedQueries {
                if let similarity = fastCosine(queryEmbedding: normalizedQuery, chunkId: chunkId, chunkEmbedding: embedding) {
                    maxEmbeddingSim = max(maxEmbeddingSim, similarity)
                }
            }

            // Hybrid search: combine with keyword/BM25-like score
            scoredChunks.append((metadata: metadata, embeddingScore: maxEmbeddingSim))
        }

        // Sort by embedding score and compute optional keyword score only for top candidates.
        scoredChunks.sort { $0.embeddingScore > $1.embeddingScore }

        // Combine scores: weighted fusion of embedding and keyword scores.
        // Do not allow keyword fusion to *reduce* the embedding score (quality-preserving).
        let hybridWeight: Double = enableHybridSearch ? 0.3 : 0.0 // 30% weight for keywords
        let embeddingWeight: Double = 1.0 - hybridWeight

        let keywordCandidateCount: Int = {
            guard enableHybridSearch else { return 0 }
            // Compute BM25 only for the same slice we'd consider for coarse rerank anyway.
            let baseline = reRankEnabled ? max(coarseTopK, topK * 3) : topK
            return min(scoredChunks.count, max(baseline, topK * 6))
        }()

        var combinedScores: [(metadata: ChunkMetadata, score: Double)] = []
        combinedScores.reserveCapacity(scoredChunks.count)

        for (index, item) in scoredChunks.enumerated() {
            let combined: Double
            if enableHybridSearch, index < keywordCandidateCount {
                let keywordScore = computeBM25Score(query: query, content: item.metadata.content)
                let fused = item.embeddingScore * embeddingWeight + keywordScore * hybridWeight
                combined = max(item.embeddingScore, fused)
            } else {
                combined = item.embeddingScore
            }
            combinedScores.append((metadata: item.metadata, score: combined))
        }

        // Filter by threshold and sort
        combinedScores = combinedScores.filter { $0.score >= scoreThreshold }
        combinedScores.sort { $0.score > $1.score }

        // Stage 2: Fine re-ranking on top candidates
        var finalResults: [(metadata: ChunkMetadata, score: Double)]
        if reRankEnabled && combinedScores.count > topK {
            let topCandidates = Array(combinedScores.prefix(coarseK))
            finalResults = await reRankCandidates(
                candidates: topCandidates,
                query: query,
                primaryEmbedding: primaryNormalizedQuery
            )
        } else {
            finalResults = combinedScores
        }

        finalResults = diversifyResults(finalResults, desiredTopK: topK)

        // Take final top K
        let topResults = finalResults.prefix(topK)

        // Convert to SearchResult
        let results: [SearchResult] = topResults.compactMap { item in
            let doc = documents[item.metadata.documentId]
            return SearchResult(
                score: item.score,
                chunkId: item.metadata.id,
                documentId: item.metadata.documentId,
                title: doc?.title,
                sourceType: doc?.sourceType ?? "unknown",
                sourcePath: doc?.sourcePath,
                content: item.metadata.content,
                startOffset: item.metadata.startOffset,
                endOffset: item.metadata.endOffset
            )
        }

        let duration = Date().timeIntervalSince(startTime)
        if FeatureFlags.AppleEmbeddingDebugLogging {
            let topScore = results.first?.score ?? 0
            let expandedCount = expandedQueries.count
            DiagnosticsLogger.log("[LocalRAG] Search completed in \(String(format: "%.3f", duration))s, found \(results.count) results, queries=\(expandedCount), hybrid=\(enableHybridSearch), rerank=\(reRankEnabled), top score: \(String(format: "%.4f", topScore))")
        }

        storeSearchCache(results, forKey: cacheKey)
        return results
    }

    private func makeSearchCacheKey(
        query: String,
        topK: Int,
        filters: [String: String]?,
        minScore: Double?,
        enableQueryExpansion: Bool,
        enableHybridSearch: Bool
    ) -> String {
        let normalizedQuery = query
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .lowercased()

        let filtersKey: String = {
            guard let filters, !filters.isEmpty else { return "" }
            return filters
                .sorted { $0.key < $1.key }
                .map { "\($0.key)=\($0.value)" }
                .joined(separator: "&")
        }()

        let raw = "\(normalizedQuery)|k=\(topK)|min=\(minScore ?? -1)|qe=\(enableQueryExpansion ? 1 : 0)|hy=\(enableHybridSearch ? 1 : 0)|f=\(filtersKey)"
        return DeduplicationService.hash(raw)
    }

    private func pruneSearchCache(now: Date) {
        if !searchCacheOrder.isEmpty {
            let deadline = now.addingTimeInterval(-searchCacheTTL)
            for key in searchCacheOrder {
                if let entry = searchCache[key], entry.createdAt < deadline {
                    searchCache[key] = nil
                }
            }
            searchCacheOrder.removeAll { searchCache[$0] == nil }
        }

        if searchCacheOrder.count > searchCacheMaxEntries {
            let overflow = searchCacheOrder.count - searchCacheMaxEntries
            let evicted = searchCacheOrder.prefix(overflow)
            for key in evicted {
                searchCache[key] = nil
            }
            searchCacheOrder.removeFirst(overflow)
        }
    }

    private func storeSearchCache(_ results: [SearchResult], forKey key: String) {
        searchCache[key] = SearchCacheEntry(results: results, createdAt: Date())
        searchCacheOrder.removeAll { $0 == key }
        searchCacheOrder.append(key)
        pruneSearchCache(now: Date())
    }

    private func clearSearchCache() {
        searchCache.removeAll()
        searchCacheOrder.removeAll()
    }

    private func diversifyResults(
        _ results: [(metadata: ChunkMetadata, score: Double)],
        desiredTopK: Int
    ) -> [(metadata: ChunkMetadata, score: Double)] {
        guard results.count > 1 else { return results }
        let uniqueDocs = Set(results.map { $0.metadata.documentId })
        guard uniqueDocs.count > 1 else { return results }

        let maxPerDocument = max(1, min(3, (desiredTopK + 1) / 2))
        var perDocCount: [String: Int] = [:]
        var seenContentHashes: Set<String> = []
        var out: [(metadata: ChunkMetadata, score: Double)] = []
        out.reserveCapacity(min(desiredTopK, results.count))

        func contentHash(_ text: String) -> String {
            let normalized = text
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
                .lowercased()
            return DeduplicationService.hash(normalized)
        }

        for r in results {
            if out.count >= desiredTopK { break }
            let docId = r.metadata.documentId
            if perDocCount[docId, default: 0] >= maxPerDocument { continue }
            let hash = contentHash(r.metadata.content)
            if !seenContentHashes.insert(hash).inserted { continue }
            perDocCount[docId, default: 0] += 1
            out.append(r)
        }

        if out.count >= min(desiredTopK, results.count) { return out }

        // Backfill (relax per-doc constraint) so we always return up to desiredTopK when possible.
        for r in results {
            if out.count >= desiredTopK { break }
            let hash = contentHash(r.metadata.content)
            if !seenContentHashes.insert(hash).inserted { continue }
            out.append(r)
        }

        return out.isEmpty ? results : out
    }

    // MARK: - Query Expansion

    /// Expand query with synonyms and related terms for better recall
    private func expandQuery(_ query: String) -> [String] {
        var queries = [query]

        // Extract key terms
        let words = query.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count > 2 }

        // Add synonym expansions for common educational terms
        let synonymMap: [String: [String]] = [
            // Learning terms
            "learn": ["study", "understand", "master"],
            "study": ["learn", "review", "practice"],
            "explain": ["describe", "clarify", "define"],
            "example": ["sample", "instance", "illustration"],
            "concept": ["idea", "principle", "theory"],
            "define": ["explain", "describe", "meaning"],
            // Question types
            "what": ["which", "how"],
            "why": ["reason", "cause", "because"],
            "how": ["method", "process", "steps"],
            // Content types
            "summary": ["overview", "recap", "brief"],
            "detail": ["specific", "in-depth", "comprehensive"],
            "key": ["important", "main", "essential"],
            // Academic terms
            "theory": ["concept", "principle", "hypothesis"],
            "method": ["approach", "technique", "process"],
            "result": ["outcome", "finding", "conclusion"],
            "analysis": ["examination", "study", "review"]
        ]

        // Generate expanded queries
        for word in words {
            if let synonyms = synonymMap[word] {
                for synonym in synonyms.prefix(2) { // Limit expansions
                    let expanded = query.replacingOccurrences(of: word, with: synonym, options: .caseInsensitive)
                    if expanded != query && !queries.contains(expanded) {
                        queries.append(expanded)
                    }
                }
            }
        }

        // Limit total expanded queries to avoid performance issues
        return Array(queries.prefix(4))
    }

    // MARK: - Hybrid Search (BM25-like scoring)

    /// Compute BM25-like keyword matching score
    private func computeBM25Score(query: String, content: String) -> Double {
        let queryTerms = tokenize(query)
        let contentTerms = tokenize(content)

        guard !queryTerms.isEmpty, !contentTerms.isEmpty else { return 0 }

        let contentLength = Double(contentTerms.count)
        let avgDocLength: Double = 200.0 // Approximate average chunk size
        let k1: Double = 1.5
        let b: Double = 0.75

        // Count term frequencies
        var termFrequency: [String: Int] = [:]
        for term in contentTerms {
            termFrequency[term, default: 0] += 1
        }

        var score: Double = 0
        for term in Set(queryTerms) {
            guard let tf = termFrequency[term] else { continue }

            // Simplified IDF (assume moderate document frequency)
            let idf: Double = 2.0

            // BM25 term score
            let tfNorm = Double(tf) * (k1 + 1)
            let denominator = Double(tf) + k1 * (1 - b + b * contentLength / avgDocLength)
            let termScore = idf * tfNorm / denominator

            score += termScore
        }

        // Normalize to 0-1 range (approximate)
        return min(1.0, score / (Double(queryTerms.count) * 3.0))
    }

    /// Simple tokenization for BM25
    private func tokenize(_ text: String) -> [String] {
        return text.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count > 2 }
    }

    // MARK: - Two-Stage Re-ranking

    /// Fine re-ranking of candidates using additional signals
    private func reRankCandidates(
        candidates: [(metadata: ChunkMetadata, score: Double)],
        query: String,
        primaryEmbedding: [Float]
    ) async -> [(metadata: ChunkMetadata, score: Double)] {
        guard !candidates.isEmpty else { return [] }

        let queryTerms = Set(tokenize(query))

        var reRankedResults: [(metadata: ChunkMetadata, score: Double)] = []

        for candidate in candidates {
            var finalScore = candidate.score

            // Factor 1: Term overlap boost
            let contentTerms = Set(tokenize(candidate.metadata.content))
            let overlap = Double(queryTerms.intersection(contentTerms).count) / Double(max(1, queryTerms.count))
            finalScore += overlap * 0.1 // 10% boost for term overlap

            // Factor 2: Position boost (earlier chunks in document may be more relevant)
            let positionBoost = 1.0 / (1.0 + Double(candidate.metadata.index) * 0.05)
            finalScore *= positionBoost

            // Factor 3: Length normalization (prefer chunks with substantial content)
            let contentLength = candidate.metadata.content.count
            let lengthBoost: Double
            if contentLength < 100 {
                lengthBoost = 0.9 // Penalize very short chunks
            } else if contentLength > 500 {
                lengthBoost = 0.95 // Slight penalty for very long chunks
            } else {
                lengthBoost = 1.0
            }
            finalScore *= lengthBoost

            // Factor 4: Exact phrase match boost
            if candidate.metadata.content.lowercased().contains(query.lowercased()) {
                finalScore += 0.15 // 15% boost for exact phrase match
            }

            reRankedResults.append((metadata: candidate.metadata, score: finalScore))
        }

        // Sort by final score
        reRankedResults.sort { $0.score > $1.score }

        return reRankedResults
    }

    /// Delete a document and all its chunks
    public func deleteDocument(documentId: String) async -> Bool {
        guard documents[documentId] != nil else { return false }
        clearSearchCache()

        purgeDocumentInMemory(documentId: documentId, shouldDeleteEmbeddingFile: true)

        // Mark metadata as dirty
        dirtyMetadata = true
        dirtyDocumentIds.remove(documentId)

        // Persist changes
        await saveIncrementally()
        await rebuildANNIndex()

        if FeatureFlags.AppleEmbeddingDebugLogging {
            DiagnosticsLogger.log("[LocalRAG] Deleted document: \(documentId)")
        }

        return true
    }

    /// Delete all documents matching a source type + source path pair.
    /// Intended for keeping hashed `sourceType/sourcePath` documents idempotent (e.g., transcripts).
    public func deleteDocuments(sourceType: String, sourcePath: String, excludingDocumentId: String? = nil) async {
        let idsToDelete = documents.values.compactMap { doc -> String? in
            guard doc.sourceType == sourceType, doc.sourcePath == sourcePath else { return nil }
            if let excludingDocumentId, doc.id == excludingDocumentId { return nil }
            return doc.id
        }

        guard !idsToDelete.isEmpty else { return }
        clearSearchCache()

        for id in idsToDelete {
            purgeDocumentInMemory(documentId: id, shouldDeleteEmbeddingFile: true)
            dirtyDocumentIds.remove(id)
        }

        dirtyMetadata = true
        await saveIncrementally()
        await rebuildANNIndex()
    }

    /// Check if a document exists
    public func documentExists(documentId: String) -> Bool {
        return documents[documentId] != nil
    }

    /// Get document by ID
    public func getDocument(documentId: String) -> Document? {
        return documents[documentId]
    }

    /// Get all documents
    public func getAllDocuments() -> [Document] {
        return Array(documents.values).sorted { $0.createdAt > $1.createdAt }
    }

    // MARK: - Coaching/RAG Status API

    /// Check if a document has indexed embeddings
    public func hasEmbeddings(documentId: String) -> Bool {
        guard documents[documentId] != nil else { return false }
        guard let chunkIds = documentChunks[documentId], !chunkIds.isEmpty else { return false }
        return true
    }

    /// Get all indexed document IDs
    public func getAllIndexedDocumentIds() -> Set<String> {
        return Set(documents.keys)
    }

    /// Statistics for a specific document
    public struct DocumentStats {
        public let chunkCount: Int
        public let embeddingMetadata: EmbeddingMetadata?
        public let createdAt: Date
        public let updatedAt: Date
        public let hasLoadedEmbeddings: Bool

        public init(
            chunkCount: Int,
            embeddingMetadata: EmbeddingMetadata?,
            createdAt: Date,
            updatedAt: Date,
            hasLoadedEmbeddings: Bool
        ) {
            self.chunkCount = chunkCount
            self.embeddingMetadata = embeddingMetadata
            self.createdAt = createdAt
            self.updatedAt = updatedAt
            self.hasLoadedEmbeddings = hasLoadedEmbeddings
        }
    }

    /// Get embedding statistics for a specific document
    public func getDocumentStats(documentId: String) -> DocumentStats? {
        guard let doc = documents[documentId] else { return nil }
        let chunkCount = documentChunks[documentId]?.count ?? 0
        let hasLoaded = loadedDocumentEmbeddings.contains(documentId)

        return DocumentStats(
            chunkCount: chunkCount,
            embeddingMetadata: doc.embeddingMetadata,
            createdAt: doc.createdAt,
            updatedAt: doc.updatedAt,
            hasLoadedEmbeddings: hasLoaded
        )
    }

    /// Get chunks for a document (loads embeddings if needed)
    public func getChunks(forDocumentId documentId: String) async -> [Chunk] {
        guard let chunkIds = documentChunks[documentId] else { return [] }

        // Ensure embeddings are loaded
        await ensureEmbeddingsLoaded(forDocumentId: documentId)

        return chunkIds.compactMap { chunkId in
            guard let metadata = chunkMetadata[chunkId],
                  let embedding = loadedEmbeddings[chunkId] else { return nil }
            return Chunk(metadata: metadata, embedding: embedding)
        }.sorted { $0.index < $1.index }
    }

    /// Get statistics about the local RAG store
    public func getStatistics() -> [String: Any] {
        let totalChunks = chunkMetadata.count
        let totalDocuments = documents.count
        let avgChunksPerDoc = totalDocuments > 0 ? Double(totalChunks) / Double(totalDocuments) : 0

        // Calculate storage size
        let docsURL = storageURL.appendingPathComponent(documentsFileName)
        let metadataURL = storageURL.appendingPathComponent(metadataFileName)
        let embeddingsURL = storageURL.appendingPathComponent(embeddingsDir)

        let docsSize = (try? FileManager.default.attributesOfItem(atPath: docsURL.path)[.size] as? Int) ?? 0
        let metadataSize = (try? FileManager.default.attributesOfItem(atPath: metadataURL.path)[.size] as? Int) ?? 0

        var embeddingsSize = 0
        if let enumerator = FileManager.default.enumerator(at: embeddingsURL, includingPropertiesForKeys: [.fileSizeKey]) {
            for case let fileURL as URL in enumerator {
                if let size = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                    embeddingsSize += size
                }
            }
        }

        let totalSize = docsSize + metadataSize + embeddingsSize

        return [
            "totalDocuments": totalDocuments,
            "totalChunks": totalChunks,
            "averageChunksPerDocument": avgChunksPerDoc,
            "storageSizeBytes": totalSize,
            "storageSizeMB": Double(totalSize) / 1_000_000,
            "metadataSizeMB": Double(docsSize + metadataSize) / 1_000_000,
            "embeddingsSizeMB": Double(embeddingsSize) / 1_000_000,
            "loadedEmbeddings": loadedEmbeddings.count,
            "loadedDocuments": loadedDocumentEmbeddings.count
        ]
    }

    /// Clear all local RAG data
    public func clearAll() async {
        documents.removeAll()
        chunkMetadata.removeAll()
        documentChunks.removeAll()
        loadedEmbeddings.removeAll()
        loadedDocumentEmbeddings.removeAll()
        loadedDocumentOrder.removeAll()
        quantizationParams.removeAll()
        normalizedEmbeddings.removeAll()
        clusterCentroids.removeAll()
        annClusters.removeAll()
        clusterSizes.removeAll()
        embeddingDimension = nil
        dirtyDocumentIds.removeAll()
        dirtyMetadata = false

        // Delete all files
        try? FileManager.default.removeItem(at: storageURL.appendingPathComponent(documentsFileName))
        try? FileManager.default.removeItem(at: storageURL.appendingPathComponent(metadataFileName))
        try? FileManager.default.removeItem(at: storageURL.appendingPathComponent(embeddingsDir))

        // Recreate embeddings directory
        try? FileManager.default.createDirectory(
            at: storageURL.appendingPathComponent(embeddingsDir),
            withIntermediateDirectories: true
        )

        if FeatureFlags.AppleEmbeddingDebugLogging {
            DiagnosticsLogger.log("[LocalRAG] Cleared all data")
        }
    }

    /// Configure optional cloud sync integration from host app/adapters.
    public func setCloudSyncHandler(_ handler: (@Sendable (String) async throws -> Void)?) {
        cloudSyncHandler = handler
    }

    /// Force save all data (for app termination)
    public func forceSaveAll() async {
        await saveAllMetadata()
        for documentId in loadedDocumentEmbeddings {
            await saveEmbeddingsBinary(forDocumentId: documentId)
        }
    }

    // MARK: - Private: Lazy Loading

    /// Ensure embeddings are loaded for a specific document
    private func ensureEmbeddingsLoaded(forDocumentId documentId: String) async {
        guard !loadedDocumentEmbeddings.contains(documentId) else {
            // Move to end of LRU order (most recently used)
            if let idx = loadedDocumentOrder.firstIndex(of: documentId) {
                loadedDocumentOrder.remove(at: idx)
                loadedDocumentOrder.append(documentId)
            }
            return
        }
        evictOldDocumentEmbeddingsIfNeeded()
        await loadEmbeddingsBinary(forDocumentId: documentId)
        loadedDocumentOrder.append(documentId)
    }

    /// Evict least-recently-used document embeddings when cache exceeds limit
    private func evictOldDocumentEmbeddingsIfNeeded() {
        while loadedDocumentEmbeddings.count >= maxLoadedDocuments, let oldest = loadedDocumentOrder.first {
            loadedDocumentOrder.removeFirst()
            loadedDocumentEmbeddings.remove(oldest)
            if let chunkIds = documentChunks[oldest] {
                for chunkId in chunkIds {
                    loadedEmbeddings.removeValue(forKey: chunkId)
                    normalizedEmbeddings.removeValue(forKey: chunkId)
                }
            }
            quantizationParams.removeValue(forKey: oldest)
        }
    }

    /// Ensure embeddings are loaded for specific chunk IDs
    private func ensureEmbeddingsLoaded(forChunkIds chunkIds: [String]) async {
        var documentsToLoad: Set<String> = []

        for chunkId in chunkIds {
            if loadedEmbeddings[chunkId] == nil,
               let metadata = chunkMetadata[chunkId] {
                documentsToLoad.insert(metadata.documentId)
            }
        }

        for documentId in documentsToLoad {
            await ensureEmbeddingsLoaded(forDocumentId: documentId)
        }
    }

    // MARK: - Private: Similarity Calculation

    /// Fast cosine similarity using pre-normalized vectors and vDSP dot products
    private func fastCosine(queryEmbedding: [Float], chunkId: String, chunkEmbedding: [Float]) -> Double? {
        if let normalizedChunk = normalizedEmbeddings[chunkId] {
            guard queryEmbedding.count == normalizedChunk.count else { return nil }
            let length = vDSP_Length(queryEmbedding.count)
            var dot: Float = 0
            vDSP_dotpr(queryEmbedding, 1, normalizedChunk, 1, &dot, length)
            return Double(dot)
        }

        guard let normalizedChunk = normalizeVector(chunkEmbedding) else { return nil }
        normalizedEmbeddings[chunkId] = normalizedChunk

        let length = vDSP_Length(queryEmbedding.count)
        var dot: Float = 0
        vDSP_dotpr(queryEmbedding, 1, normalizedChunk, 1, &dot, length)
        return Double(dot)
    }

    /// Normalize a vector to unit length
    private func normalizeVector(_ vector: [Float]) -> [Float]? {
        guard !vector.isEmpty else { return nil }

        if let dimension = embeddingDimension, dimension != vector.count {
            return nil
        }

        var normSquared: Float = 0
        vDSP_svesq(vector, 1, &normSquared, vDSP_Length(vector.count))

        let norm = sqrt(normSquared)
        guard norm > 0 else { return nil }

        var normalized = [Float](repeating: 0, count: vector.count)
        var normCopy = norm
        vDSP_vsdiv(vector, 1, &normCopy, &normalized, 1, vDSP_Length(vector.count))

        if embeddingDimension == nil {
            embeddingDimension = vector.count
        }

        return normalized
    }

    /// Maintain lightweight ANN clusters for fast candidate selection
    private func addToANNIndex(chunkId: String, normalizedEmbedding: [Float]) {
        if clusterCentroids.isEmpty {
            clusterCentroids = [normalizedEmbedding]
            annClusters = [[chunkId]]
            clusterSizes = [1]
            return
        }

        var bestIndex = 0
        var bestScore: Float = -1

        for (index, centroid) in clusterCentroids.enumerated() {
            guard centroid.count == normalizedEmbedding.count else { continue }
            var dot: Float = 0
            vDSP_dotpr(centroid, 1, normalizedEmbedding, 1, &dot, vDSP_Length(centroid.count))
            if dot > bestScore {
                bestScore = dot
                bestIndex = index
            }
        }

        if clusterCentroids.count < maxAnnClusters && bestScore < 0.6 {
            clusterCentroids.append(normalizedEmbedding)
            annClusters.append([chunkId])
            clusterSizes.append(1)
            return
        }

        annClusters[bestIndex].append(chunkId)
        clusterSizes[bestIndex] += 1

        // Incremental centroid update (running mean)
        var centroid = clusterCentroids[bestIndex]
        let count = Float(clusterSizes[bestIndex])
        for i in 0..<centroid.count {
            centroid[i] += (normalizedEmbedding[i] - centroid[i]) / count
        }
        clusterCentroids[bestIndex] = centroid
    }

    /// Select candidate chunks by ANN clusters
    private func candidateChunkIds(for normalizedQuery: [Float]) -> [String] {
        guard !clusterCentroids.isEmpty else { return Array(chunkMetadata.keys) }

        // Correctness over speed: the ANN index is built only from `loadedEmbeddings`.
        // If we haven't loaded embeddings for every document yet, using ANN would exclude
        // unseen documents entirely, harming recall.
        if loadedDocumentEmbeddings.count < documents.count {
            return Array(chunkMetadata.keys)
        }

        var scoredClusters: [(index: Int, score: Float)] = []
        for (index, centroid) in clusterCentroids.enumerated() {
            guard centroid.count == normalizedQuery.count else { continue }
            var dot: Float = 0
            vDSP_dotpr(centroid, 1, normalizedQuery, 1, &dot, vDSP_Length(centroid.count))
            scoredClusters.append((index: index, score: dot))
        }

        let topClusters = scoredClusters
            .sorted { $0.score > $1.score }
            .prefix(min(clusterSelectionCount, scoredClusters.count))

        var candidates: [String] = []
        for cluster in topClusters {
            candidates.append(contentsOf: annClusters[cluster.index])
        }

        if candidates.isEmpty {
            return Array(chunkMetadata.keys)
        }

        var seen: Set<String> = []
        var uniqueCandidates: [String] = []
        for id in candidates {
            if seen.insert(id).inserted {
                uniqueCandidates.append(id)
            }
        }

        return uniqueCandidates
    }

    /// Rebuild ANN clusters from persisted data
    private func rebuildANNIndex() async {
        normalizedEmbeddings.removeAll()
        clusterCentroids.removeAll()
        annClusters.removeAll()
        clusterSizes.removeAll()
        embeddingDimension = nil

        for (chunkId, embedding) in loadedEmbeddings {
            guard let normalized = normalizeVector(embedding) else { continue }
            normalizedEmbeddings[chunkId] = normalized
            addToANNIndex(chunkId: chunkId, normalizedEmbedding: normalized)
        }
    }

    // MARK: - Private: Int8 Quantization

    /// Quantize Float32 embeddings to Int8 for storage (75% size reduction)
    private func quantizeEmbeddings(_ embeddings: [[Float]], documentId: String) -> (Data, QuantizationParams)? {
        guard !embeddings.isEmpty, let first = embeddings.first, !first.isEmpty else { return nil }

        let dimension = first.count

        // Find global min/max across all embeddings for this document
        var globalMin: Float = .infinity
        var globalMax: Float = -.infinity

        for embedding in embeddings {
            for value in embedding {
                globalMin = min(globalMin, value)
                globalMax = max(globalMax, value)
            }
        }

        // Add small epsilon to avoid division by zero
        if globalMax - globalMin < 1e-6 {
            globalMax = globalMin + 1e-6
        }

        let params = QuantizationParams(minValue: globalMin, maxValue: globalMax, dimension: dimension)
        let scale = params.scale

        // Quantize to Int8 (0-255 range stored as UInt8)
        var quantizedData = Data(capacity: embeddings.count * dimension)

        for embedding in embeddings {
            for value in embedding {
                let normalized = (value - globalMin) / scale
                let quantized = UInt8(min(255, max(0, normalized)))
                quantizedData.append(quantized)
            }
        }

        quantizationParams[documentId] = params
        return (quantizedData, params)
    }

    /// Dequantize Int8 data back to Float32
    private func dequantizeEmbeddings(_ data: Data, params: QuantizationParams, count: Int) -> [[Float]] {
        let dimension = params.dimension
        let scale = params.scale
        let minValue = params.minValue

        var embeddings: [[Float]] = []
        embeddings.reserveCapacity(count)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for i in 0..<count {
                var embedding = [Float](repeating: 0, count: dimension)
                for j in 0..<dimension {
                    let quantized = Float(bytes[i * dimension + j])
                    embedding[j] = quantized * scale + minValue
                }
                embeddings.append(embedding)
            }
        }

        return embeddings
    }

    // MARK: - Private: Binary Persistence

    /// Save embeddings in binary format for a document
    private func saveEmbeddingsBinary(forDocumentId documentId: String) async {
        guard let chunkIds = documentChunks[documentId], !chunkIds.isEmpty else { return }

        // Collect embeddings in order
        var embeddings: [[Float]] = []
        var orderedChunkIds: [String] = []

        for chunkId in chunkIds {
            if let embedding = loadedEmbeddings[chunkId] {
                embeddings.append(embedding)
                orderedChunkIds.append(chunkId)
            }
        }

        guard !embeddings.isEmpty else { return }

        // Quantize and save
        guard let (quantizedData, params) = quantizeEmbeddings(embeddings, documentId: documentId) else {
            DiagnosticsLogger.log("[LocalRAG] Failed to quantize embeddings for document: \(documentId)")
            return
        }

        // Create header with chunk IDs for ordering
        let header = EmbeddingFileHeader(
            version: 1,
            chunkIds: orderedChunkIds,
            params: params,
            embeddingMetadata: documents[documentId]?.embeddingMetadata
        )

        let embeddingURL = embeddingFileURL(forDocumentId: documentId)

        do {
            let encoder = JSONEncoder()
            let headerData = try encoder.encode(header)
            let headerSizeData = withUnsafeBytes(of: Int32(headerData.count)) { Data($0) }

            var fileData = Data()
            fileData.append(headerSizeData)
            fileData.append(headerData)
            fileData.append(quantizedData)

            try fileData.write(to: embeddingURL)

            if FeatureFlags.AppleEmbeddingDebugLogging {
                let originalSize = embeddings.count * embeddings[0].count * 4 // Float32 = 4 bytes
                let savedSize = fileData.count
                let reduction = 100.0 * (1.0 - Double(savedSize) / Double(originalSize))
                DiagnosticsLogger.log("[LocalRAG] Saved binary embeddings for \(documentId): \(embeddings.count) vectors, \(String(format: "%.1f", reduction))% size reduction")
            }
        } catch {
            DiagnosticsLogger.log("[LocalRAG] Failed to save binary embeddings: \(error.localizedDescription)")
        }
    }

    /// Load embeddings from binary format for a document
    private func loadEmbeddingsBinary(forDocumentId documentId: String) async {
        let embeddingURL = embeddingFileURL(forDocumentId: documentId)

        guard FileManager.default.fileExists(atPath: embeddingURL.path) else {
            // Try legacy JSON format
            await loadLegacyEmbeddings(forDocumentId: documentId)
            return
        }

        do {
            let fileData = try Data(contentsOf: embeddingURL)
            guard fileData.count >= 4 else { return }

            // Read header size
            let headerSize = fileData.withUnsafeBytes { buffer -> Int32 in
                buffer.load(as: Int32.self)
            }

            guard fileData.count >= 4 + Int(headerSize) else { return }

            // Read header
            let headerData = fileData.subdata(in: 4..<(4 + Int(headerSize)))
            let decoder = JSONDecoder()
            let header = try decoder.decode(EmbeddingFileHeader.self, from: headerData)

            if var document = documents[documentId],
               document.embeddingMetadata == nil,
               let metadata = header.embeddingMetadata {
                document.embeddingMetadata = metadata
                documents[documentId] = document
                dirtyMetadata = true
            }

            // Read quantized embeddings
            let embeddingsData = fileData.subdata(in: (4 + Int(headerSize))..<fileData.count)
            let embeddings = dequantizeEmbeddings(embeddingsData, params: header.params, count: header.chunkIds.count)

            // Map back to chunk IDs
            for (index, chunkId) in header.chunkIds.enumerated() {
                if index < embeddings.count {
                    loadedEmbeddings[chunkId] = embeddings[index]

                    // Rebuild normalized cache and ANN index
                    if let normalized = normalizeVector(embeddings[index]) {
                        normalizedEmbeddings[chunkId] = normalized
                        addToANNIndex(chunkId: chunkId, normalizedEmbedding: normalized)
                    }
                }
            }

            loadedDocumentEmbeddings.insert(documentId)
            loadedDocumentOrder.removeAll { $0 == documentId }
            loadedDocumentOrder.append(documentId)
            quantizationParams[documentId] = header.params

            if FeatureFlags.AppleEmbeddingDebugLogging {
                DiagnosticsLogger.log("[LocalRAG] Loaded binary embeddings for \(documentId): \(embeddings.count) vectors")
            }
        } catch {
            DiagnosticsLogger.log("[LocalRAG] Failed to load binary embeddings: \(error.localizedDescription)")
        }
    }

    /// Load legacy JSON embeddings and convert to binary
    private func loadLegacyEmbeddings(forDocumentId documentId: String) async {
        // Check if this document has chunks in the legacy format
        guard documentChunks[documentId] != nil else { return }

        let legacyURL = storageURL.appendingPathComponent(legacyChunksFileName)
        guard FileManager.default.fileExists(atPath: legacyURL.path) else { return }

        do {
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601

            let data = try Data(contentsOf: legacyURL)
            let legacyChunks = try decoder.decode([LegacyChunk].self, from: data)

            // Find chunks for this document
            var foundAny = false
            for legacyChunk in legacyChunks where legacyChunk.documentId == documentId {
                loadedEmbeddings[legacyChunk.id] = legacyChunk.embedding

                if let normalized = normalizeVector(legacyChunk.embedding) {
                    normalizedEmbeddings[legacyChunk.id] = normalized
                    addToANNIndex(chunkId: legacyChunk.id, normalizedEmbedding: normalized)
                }
                foundAny = true
            }

            if foundAny {
                loadedDocumentEmbeddings.insert(documentId)
                loadedDocumentOrder.removeAll { $0 == documentId }
                loadedDocumentOrder.append(documentId)

                // Convert to binary format
                await saveEmbeddingsBinary(forDocumentId: documentId)

                if FeatureFlags.AppleEmbeddingDebugLogging {
                    DiagnosticsLogger.log("[LocalRAG] Migrated legacy embeddings for \(documentId)")
                }
            }
        } catch {
            DiagnosticsLogger.log("[LocalRAG] Failed to load legacy embeddings: \(error.localizedDescription)")
        }
    }

    private func embeddingFileURL(forDocumentId documentId: String) -> URL {
        return storageURL
            .appendingPathComponent(embeddingsDir)
            .appendingPathComponent("\(documentId).bin")
    }

    private func deleteEmbeddingFile(forDocumentId documentId: String) {
        let url = embeddingFileURL(forDocumentId: documentId)
        try? FileManager.default.removeItem(at: url)
    }

    private func triggerCloudKitSync(for documentId: String) {
        guard FeatureFlags.RAGCloudKitSyncEnabled else { return }
        guard let cloudSyncHandler else { return }

        Task {
            do {
                try await cloudSyncHandler(documentId)
            } catch {
                guard FeatureFlags.RAGCloudKitDebugLogging else { return }
                DiagnosticsLogger.log("[RAGCloudSync] Failed to sync document \(documentId): \(error.localizedDescription)")
            }
        }
    }

    // MARK: - Embedding Re-indexing

    /// Regenerate embeddings for a document when the embedding model changes.
    private func reindexDocumentEmbeddings(documentId: String, targetMetadata: EmbeddingMetadata) async {
        guard let chunkIds = documentChunks[documentId], !chunkIds.isEmpty else { return }

        let orderedChunks = chunkIds.compactMap { chunkMetadata[$0] }.sorted { $0.index < $1.index }
        guard !orderedChunks.isEmpty else { return }

        let embeddingService = AppleEmbeddingService.shared
        guard await embeddingService.isAvailable else { return }

        if FeatureFlags.AppleEmbeddingDebugLogging {
            DiagnosticsLogger.log("[LocalRAG] Re-indexing document \(documentId) due to embedding model change")
        }

        let chunkTexts = orderedChunks.map { $0.content }
        let embeddings = await embeddingService.embedBatch(texts: chunkTexts)

        var newEmbeddings: [String: [Float]] = [:]
        for (index, metadata) in orderedChunks.enumerated() {
            if let embedding = embeddings[index] {
                newEmbeddings[metadata.id] = embedding
            }
        }

        guard newEmbeddings.count == orderedChunks.count else {
            if FeatureFlags.AppleEmbeddingDebugLogging {
                DiagnosticsLogger.log("[LocalRAG] Re-index for \(documentId) skipped, embeddings missing (\(newEmbeddings.count)/\(orderedChunks.count))")
            }
            return
        }

        // Remove old embeddings for this document to avoid stale ANN entries
        for chunkId in chunkIds {
            loadedEmbeddings.removeValue(forKey: chunkId)
            normalizedEmbeddings.removeValue(forKey: chunkId)
        }

        // Apply new embeddings
        for (chunkId, embedding) in newEmbeddings {
            loadedEmbeddings[chunkId] = embedding
        }

        loadedDocumentEmbeddings.insert(documentId)
        loadedDocumentOrder.removeAll { $0 == documentId }
        loadedDocumentOrder.append(documentId)
        quantizationParams.removeValue(forKey: documentId)

        if var document = documents[documentId] {
            document.embeddingMetadata = targetMetadata
            document.updatedAt = Date()
            documents[documentId] = document
        }

        dirtyDocumentIds.insert(documentId)
        dirtyMetadata = true

        // Persist new embeddings and metadata
        await saveIncrementally()
    }

    /// Ensure embeddings are regenerated when the underlying model changes.
    private func reconcileEmbeddingMetadataIfNeeded() async {
        guard let currentMetadata = await currentEmbeddingMetadata(), !documents.isEmpty else { return }

        var reindexedAny = false

        for document in documents.values {
            if shouldReindex(document: document, currentMetadata: currentMetadata) {
                await reindexDocumentEmbeddings(documentId: document.id, targetMetadata: currentMetadata)
                reindexedAny = true
            }
        }

        if reindexedAny {
            clearSearchCache()
            await rebuildANNIndex()
        }
    }

    // MARK: - Private: Incremental Persistence

    /// Save only changed data
    private func saveIncrementally() async {
        // Save metadata if dirty
        if dirtyMetadata {
            await saveAllMetadata()
            dirtyMetadata = false
        }

        // Save only dirty document embeddings
        for documentId in dirtyDocumentIds {
            await saveEmbeddingsBinary(forDocumentId: documentId)
        }
        dirtyDocumentIds.removeAll()
    }

    /// Save all metadata (documents + chunk metadata, no embeddings)
    private func saveAllMetadata() async {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601

        // Save documents
        do {
            let docsData = try encoder.encode(Array(documents.values))
            let docsURL = storageURL.appendingPathComponent(documentsFileName)
            try docsData.write(to: docsURL)
        } catch {
            DiagnosticsLogger.log("[LocalRAG] Failed to save documents: \(error.localizedDescription)")
        }

        // Save chunk metadata (without embeddings)
        do {
            let metadataData = try encoder.encode(Array(chunkMetadata.values))
            let metadataURL = storageURL.appendingPathComponent(metadataFileName)
            try metadataData.write(to: metadataURL)
        } catch {
            DiagnosticsLogger.log("[LocalRAG] Failed to save metadata: \(error.localizedDescription)")
        }
    }

    /// Load metadata only (lazy load embeddings later)
    private func loadMetadataFromDisk() async {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        // Load documents
        let docsURL = storageURL.appendingPathComponent(documentsFileName)
        if let docsData = try? Data(contentsOf: docsURL),
           let loadedDocs = try? decoder.decode([Document].self, from: docsData) {
            for doc in loadedDocs {
                documents[doc.id] = doc
            }
            if FeatureFlags.AppleEmbeddingDebugLogging {
                DiagnosticsLogger.log("[LocalRAG] Loaded \(loadedDocs.count) documents from disk")
            }
        }

        // Try new metadata format first
        let metadataURL = storageURL.appendingPathComponent(metadataFileName)
        if let metadataData = try? Data(contentsOf: metadataURL),
           let loadedMetadata = try? decoder.decode([ChunkMetadata].self, from: metadataData) {
            for metadata in loadedMetadata {
                chunkMetadata[metadata.id] = metadata
                if documentChunks[metadata.documentId] == nil {
                    documentChunks[metadata.documentId] = []
                }
                documentChunks[metadata.documentId]?.append(metadata.id)
            }
            if FeatureFlags.AppleEmbeddingDebugLogging {
                DiagnosticsLogger.log("[LocalRAG] Loaded \(loadedMetadata.count) chunk metadata from disk")
            }
        } else {
            // Fall back to legacy format
            await loadLegacyChunksMetadata()
        }

        // Ensure embeddings stay compatible when the embedding model changes.
        await reconcileEmbeddingMetadataIfNeeded()
    }

    /// Load legacy chunk format and migrate
    private func loadLegacyChunksMetadata() async {
        let legacyURL = storageURL.appendingPathComponent(legacyChunksFileName)
        guard FileManager.default.fileExists(atPath: legacyURL.path) else { return }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        do {
            let data = try Data(contentsOf: legacyURL)
            let legacyChunks = try decoder.decode([LegacyChunk].self, from: data)

            for chunk in legacyChunks {
                let metadata = ChunkMetadata(
                    id: chunk.id,
                    documentId: chunk.documentId,
                    index: chunk.index,
                    content: chunk.content,
                    startOffset: chunk.startOffset,
                    endOffset: chunk.endOffset,
                    createdAt: chunk.createdAt
                )
                chunkMetadata[metadata.id] = metadata

                if documentChunks[metadata.documentId] == nil {
                    documentChunks[metadata.documentId] = []
                }
                documentChunks[metadata.documentId]?.append(metadata.id)
            }

            if FeatureFlags.AppleEmbeddingDebugLogging {
                DiagnosticsLogger.log("[LocalRAG] Migrated \(legacyChunks.count) chunks from legacy format")
            }

            // Save in new format
            dirtyMetadata = true
            await saveAllMetadata()

            // Migration will happen lazily when embeddings are accessed
        } catch {
            DiagnosticsLogger.log("[LocalRAG] Failed to load legacy chunks: \(error.localizedDescription)")
        }
    }

    // MARK: - Private: Helper Structures

    /// Binary file header for embeddings
    private struct EmbeddingFileHeader: Codable {
        let version: Int
        let chunkIds: [String]
        let params: QuantizationParams
        let embeddingMetadata: EmbeddingMetadata?
    }

    /// Legacy chunk format for migration
    private struct LegacyChunk: Codable {
        let id: String
        let documentId: String
        let index: Int
        let content: String
        let startOffset: Int
        let endOffset: Int
        let embedding: [Float]
        let createdAt: Date
    }

    private func chunkIdForDocument(documentId: String, chunkIndex: Int) -> String {
        "chunk_\(documentId)_\(chunkIndex)"
    }

    private func purgeDocumentInMemory(documentId: String, shouldDeleteEmbeddingFile: Bool) {
        // Remove all chunks for this document
        if let chunkIds = documentChunks[documentId] {
            for chunkId in chunkIds {
                chunkMetadata.removeValue(forKey: chunkId)
                loadedEmbeddings.removeValue(forKey: chunkId)
                normalizedEmbeddings.removeValue(forKey: chunkId)
            }
        }

        documents.removeValue(forKey: documentId)
        documentChunks.removeValue(forKey: documentId)
        loadedDocumentEmbeddings.remove(documentId)
        quantizationParams.removeValue(forKey: documentId)

        if shouldDeleteEmbeddingFile {
            deleteEmbeddingFile(forDocumentId: documentId)
        }
    }
}

// MARK: - Compatibility Extension

public extension LocalRAGStore {
    /// Legacy Chunk type for backward compatibility
    typealias LegacyChunkType = Chunk

    /// Convert LocalRAGStore.SearchResult to RAGManager.SearchResult for compatibility
    func toRAGManagerResult(_ result: SearchResult) -> RAGManager.SearchResult {
        return RAGManager.SearchResult(
            score: result.score,
            chunk_id: result.chunkId,
            document_id: result.documentId,
            title: result.title,
            source_type: result.sourceType,
            source_path: result.sourcePath,
            content: result.content,
            start_offset: result.startOffset,
            end_offset: result.endOffset
        )
    }

    /// Search and return RAGManager-compatible results
    func searchCompatible(
        query: String,
        topK: Int = 6,
        filters: [String: String]? = nil,
        minScore: Double? = nil,
        enableQueryExpansion: Bool = true,
        enableHybridSearch: Bool = true
    ) async -> [RAGManager.SearchResult] {
        let results = await search(
            query: query,
            topK: topK,
            filters: filters,
            minScore: minScore,
            enableQueryExpansion: enableQueryExpansion,
            enableHybridSearch: enableHybridSearch
        )
        return results.map { toRAGManagerResult($0) }
    }
}
