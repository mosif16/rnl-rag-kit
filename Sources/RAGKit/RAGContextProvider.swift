import Foundation

// MARK: - RAG Context Configuration

/// Configuration for RAG context injection during content generation
public struct RAGContextConfig: Sendable {
    /// Maximum characters of RAG context to inject per chunk
    public let maxContextChars: Int

    /// Maximum number of RAG results to consider
    public let topK: Int

    /// Minimum similarity score threshold (0.0-1.0)
    public let minScore: Double

    /// Source types to search (nil = all types)
    public let sourceTypes: [RAGSourceType]?

    /// Document ID to exclude (prevents self-referencing)
    public let excludeDocumentId: String?

    /// Whether to include source attribution in context
    public let includeAttribution: Bool

    public init(
        maxContextChars: Int = 600,
        topK: Int = 3,
        minScore: Double = 0.35,
        sourceTypes: [RAGSourceType]? = nil,
        excludeDocumentId: String? = nil,
        includeAttribution: Bool = true
    ) {
        self.maxContextChars = maxContextChars
        self.topK = topK
        self.minScore = minScore
        self.sourceTypes = sourceTypes
        self.excludeDocumentId = excludeDocumentId
        self.includeAttribution = includeAttribution
    }

    /// Standard config for content generation - balanced quality/performance
    public static let standard = RAGContextConfig(
        maxContextChars: 600,
        topK: 3,
        minScore: 0.35,
        sourceTypes: [.aiNote, .flashcard, .transcript],
        excludeDocumentId: nil,
        includeAttribution: true
    )

    /// Minimal config for speed-sensitive operations
    public static let minimal = RAGContextConfig(
        maxContextChars: 300,
        topK: 2,
        minScore: 0.45,
        sourceTypes: nil,
        excludeDocumentId: nil,
        includeAttribution: false
    )

    /// Rich config for maximum context (slower but higher quality)
    public static let rich = RAGContextConfig(
        maxContextChars: 900,
        topK: 5,
        minScore: 0.30,
        sourceTypes: nil,
        excludeDocumentId: nil,
        includeAttribution: true
    )

    /// Create a copy with a specific document excluded
    public func excluding(documentId: String?) -> RAGContextConfig {
        RAGContextConfig(
            maxContextChars: maxContextChars,
            topK: topK,
            minScore: minScore,
            sourceTypes: sourceTypes,
            excludeDocumentId: documentId,
            includeAttribution: includeAttribution
        )
    }
}

// MARK: - RAG Context Result

/// Result of a RAG context fetch operation
public struct RAGContextResult: Sendable {
    /// Formatted context string ready for prompt injection
    public let context: String

    /// Number of sources that contributed to the context
    public let sourceCount: Int

    /// Total character count of the context
    public let characterCount: Int

    /// Time taken for the RAG search (milliseconds)
    public let searchTimeMs: Int

    /// Whether context was found
    public var hasContext: Bool { !context.isEmpty && sourceCount > 0 }
}

// MARK: - RAG Context Provider

/// Actor that provides RAG context for content generation pipelines
///
/// This actor fetches related content from the user's study history to inject
/// as context during flashcard/quiz/notes generation, improving quality by
/// connecting new content to existing knowledge.
///
/// Thread-safe and designed for concurrent access from generation pipelines.
@available(iOS 26.0, macOS 26.0, *)
public actor RAGContextProvider {

    // MARK: - Properties

    private let ragManager = RAGManager.shared

    /// Cache of recent searches to avoid duplicate queries within a session
    private var recentSearchCache: [String: RAGContextResult] = [:]
    private let maxCacheSize = 20

    // MARK: - Initialization

    public init() {}

    // MARK: - Public API

    /// Fetch related context for a text chunk
    /// - Parameters:
    ///   - chunk: The text chunk to find related content for
    ///   - config: Configuration for the search
    /// - Returns: RAGContextResult with formatted context and metadata
    public func fetchRelatedContext(
        for chunk: String,
        config: RAGContextConfig = .standard
    ) async -> RAGContextResult {
        let startTime = Date()

        // Check cache for identical query
        let cacheKey = makeCacheKey(chunk: chunk, config: config)
        if let cached = recentSearchCache[cacheKey] {
            DiagnosticsLogger.log("[RAGContextProvider] Cache hit for chunk")
            return cached
        }

        // Perform RAG search
        let results = await ragManager.search(
            query: chunk,
            sourceTypes: config.sourceTypes
        )

        // Filter by score and excluded document
        let filtered = results.filter { result in
            // Check minimum score
            guard result.score >= config.minScore else { return false }

            // Check excluded document
            if let excludeId = config.excludeDocumentId,
               result.document_id == excludeId {
                return false
            }

            return true
        }

        // Take top K results
        let topResults = Array(filtered.prefix(config.topK))

        // Format context
        let formattedContext = formatContext(
            results: topResults,
            maxChars: config.maxContextChars,
            includeAttribution: config.includeAttribution
        )

        let searchTimeMs = Int(Date().timeIntervalSince(startTime) * 1000)

        let result = RAGContextResult(
            context: formattedContext,
            sourceCount: topResults.count,
            characterCount: formattedContext.count,
            searchTimeMs: searchTimeMs
        )

        // Cache the result
        cacheResult(result, forKey: cacheKey)

        if FeatureFlags.RAGDebugLogging {
            DiagnosticsLogger.log("[RAGContextProvider] Found \(topResults.count) results (\(formattedContext.count) chars) in \(searchTimeMs)ms")
        }

        return result
    }

    /// Fetch context for multiple chunks efficiently
    /// - Parameters:
    ///   - chunks: Array of text chunks
    ///   - config: Configuration for searches
    /// - Returns: Dictionary mapping chunk index to context result
    public func fetchContextBatch(
        for chunks: [String],
        config: RAGContextConfig = .standard
    ) async -> [Int: RAGContextResult] {
        var results: [Int: RAGContextResult] = [:]

        // Process chunks concurrently using TaskGroup
        await withTaskGroup(of: (Int, RAGContextResult).self) { group in
            for (index, chunk) in chunks.enumerated() {
                group.addTask {
                    let result = await self.fetchRelatedContext(for: chunk, config: config)
                    return (index, result)
                }
            }

            for await (index, result) in group {
                results[index] = result
            }
        }

        return results
    }

    /// Clear the search cache
    public func clearCache() {
        recentSearchCache.removeAll()
    }

    // MARK: - Private Methods

    private func formatContext(
        results: [RAGManager.SearchResult],
        maxChars: Int,
        includeAttribution: Bool
    ) -> String {
        guard !results.isEmpty else { return "" }

        var output = "RELATED KNOWLEDGE FROM YOUR STUDY HISTORY:\n"
        var usedChars = output.count

        for result in results {
            var entry = ""

            // Add attribution header if enabled
            if includeAttribution {
                let sourceLabel = formatSourceType(result.source_type)
                if let title = result.title, !title.isEmpty {
                    entry = "[\(sourceLabel): \(title)]\n"
                } else {
                    entry = "[\(sourceLabel)]\n"
                }
            }

            // Calculate available space for content
            let availableChars = maxChars - usedChars - entry.count - 4 // 4 for "\n\n"
            if availableChars <= 50 { break } // Need at least 50 chars for meaningful content

            // Truncate content to fit budget
            let content = result.content.trimmingCharacters(in: .whitespacesAndNewlines)
            let truncatedContent: String
            if content.count > availableChars {
                // Truncate at word boundary
                let prefix = String(content.prefix(availableChars))
                if let lastSpace = prefix.lastIndex(of: " ") {
                    truncatedContent = String(prefix[..<lastSpace]) + "..."
                } else {
                    truncatedContent = prefix + "..."
                }
            } else {
                truncatedContent = content
            }

            entry += truncatedContent + "\n\n"
            usedChars += entry.count
            output += entry

            if usedChars >= maxChars { break }
        }

        return output.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func formatSourceType(_ rawType: String) -> String {
        switch rawType {
        case "ai_note": return "Notes"
        case "flashcard": return "Flashcard"
        case "quiz": return "Quiz"
        case "transcript": return "Recording"
        case "document": return "Document"
        default: return rawType.capitalized
        }
    }

    private func makeCacheKey(chunk: String, config: RAGContextConfig) -> String {
        // Use first 100 chars of chunk + config hash for cache key
        let chunkPrefix = String(chunk.prefix(100))
        let configHash = "\(config.topK)-\(config.minScore)-\(config.excludeDocumentId ?? "none")"
        return "\(chunkPrefix.hashValue)-\(configHash)"
    }

    private func cacheResult(_ result: RAGContextResult, forKey key: String) {
        // Evict oldest entries if cache is full
        if recentSearchCache.count >= maxCacheSize {
            // Remove first (oldest) entry
            if let firstKey = recentSearchCache.keys.first {
                recentSearchCache.removeValue(forKey: firstKey)
            }
        }
        recentSearchCache[key] = result
    }
}

// MARK: - Convenience Extensions

@available(iOS 26.0, macOS 26.0, *)
extension RAGContextProvider {

    /// Fetch context formatted specifically for flashcard generation
    public func fetchFlashcardContext(
        for chunk: String,
        excludingDocument documentId: String? = nil
    ) async -> String? {
        let config = RAGContextConfig.standard.excluding(documentId: documentId)
        let result = await fetchRelatedContext(for: chunk, config: config)
        return result.hasContext ? result.context : nil
    }

    /// Fetch context formatted for quiz generation (slightly higher threshold)
    public func fetchQuizContext(
        for chunk: String,
        excludingDocument documentId: String? = nil
    ) async -> String? {
        let config = RAGContextConfig(
            maxContextChars: 500,
            topK: 3,
            minScore: 0.40, // Higher threshold for quiz - want more relevant context
            sourceTypes: [.aiNote, .flashcard, .quiz],
            excludeDocumentId: documentId,
            includeAttribution: true
        )
        let result = await fetchRelatedContext(for: chunk, config: config)
        return result.hasContext ? result.context : nil
    }

    /// Fetch context for notes generation (broader search)
    public func fetchNotesContext(
        for chunk: String,
        excludingDocument documentId: String? = nil
    ) async -> String? {
        let config = RAGContextConfig.rich.excluding(documentId: documentId)
        let result = await fetchRelatedContext(for: chunk, config: config)
        return result.hasContext ? result.context : nil
    }
}
