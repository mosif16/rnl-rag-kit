import Accelerate
import Foundation
import NaturalLanguage

enum EmbeddingStrategy: String {
    case contextual = "contextual_v1"
    case sentence = "sentence_v1"
    case word = "word_v1"
}

struct EmbeddingModelInfo {
    let version: String
    let strategy: EmbeddingStrategy
    let dimension: Int
    let languageCode: String
    let modelIdentifier: String?
}

/// On-device embedding service using Apple's NLEmbedding
/// Replaces server-side Gemini embeddings for RAG operations
///
/// Features:
/// - Sentence-level embeddings (iOS 14+)
/// - Word-level embedding fallback
/// - Batch embedding support
/// - Cosine similarity calculation
/// - Multi-language support (EN, ES, DE, FR, IT, PT, ZH, JA)
///
/// Usage:
/// ```swift
/// let embedding = await AppleEmbeddingService.shared.embed(text: "Hello world")
/// let similarity = await AppleEmbeddingService.shared.similarity(text1: "cat", text2: "dog")
/// ```
actor AppleEmbeddingService {
    static let shared = AppleEmbeddingService()

    // MARK: - Properties

    /// Backing storage for contextual embedding model (iOS 17+/macOS 14+).
    private var contextualEmbeddingStorage: Any?

    /// Disable contextual embeddings when the runtime environment cannot access required assets (e.g. Simulator).
    private var contextualEmbeddingDisabledUntil: Date?
    private var contextualEmbeddingDisabledReason: String?

    /// Sentence embedding model (preferred, iOS 14+)
    private var sentenceEmbedding: NLEmbedding?

    /// Word embedding model (fallback)
    private var wordEmbedding: NLEmbedding?

    /// Current language for embeddings
    private var currentLanguage: NLLanguage = .english

    /// Simple in-memory LRU cache for recent embeddings to avoid re-computation
    private var embeddingCache: [String: [Float]] = [:]
    private var cacheOrder: [String] = []
    private let maxCacheEntries = 128

    /// Supported languages for sentence embeddings
    private let supportedLanguages: [NLLanguage] = [
        .english,
        .spanish,
        .german,
        .french,
        .italian,
        .portuguese,
        .simplifiedChinese,
        .japanese
    ]

    /// Embedding dimension (Apple NLEmbedding uses ~512 dimensions for sentences)
    var dimension: Int {
        if #available(iOS 17.0, macOS 14.0, *),
           hasContextualEmbedding,
           let embedding = contextualEmbedding {
            return embedding.dimension
        }
        if let embedding = sentenceEmbedding {
            return embedding.dimension
        }
        if let embedding = wordEmbedding {
            return embedding.dimension
        }
        return 512 // Default expected dimension
    }

    /// Language code for the currently loaded embedding.
    var languageCode: String { currentLanguage.rawValue }

    private var canAttemptContextualEmbedding: Bool {
        guard let until = contextualEmbeddingDisabledUntil else { return true }
        return Date() >= until
    }

    @available(iOS 17.0, macOS 14.0, *)
    private var contextualEmbedding: NLContextualEmbedding? {
        get { contextualEmbeddingStorage as? NLContextualEmbedding }
        set { contextualEmbeddingStorage = newValue }
    }

    private var hasContextualEmbedding: Bool {
        if #available(iOS 17.0, macOS 14.0, *) {
            guard canAttemptContextualEmbedding else { return false }
            return contextualEmbedding?.hasAvailableAssets == true
        }
        return false
    }

    private var preferredStrategy: EmbeddingStrategy? {
        if hasContextualEmbedding { return .contextual }
        if sentenceEmbedding != nil { return .sentence }
        if wordEmbedding != nil { return .word }
        return nil
    }

    var modelVersion: String {
        if hasContextualEmbedding, let identifier = contextualModelIdentifier {
            return "contextual:\(identifier)"
        }
        return "\((preferredStrategy ?? .word).rawValue)_d\(dimension)"
    }

    var currentModelInfo: EmbeddingModelInfo {
        EmbeddingModelInfo(
            version: modelVersion,
            strategy: preferredStrategy ?? .word,
            dimension: dimension,
            languageCode: languageCode,
            modelIdentifier: contextualModelIdentifier
        )
    }

    private var contextualModelIdentifier: String? {
        guard #available(iOS 17.0, macOS 14.0, *), let contextualEmbedding else { return nil }
        return contextualEmbedding.modelIdentifier
    }

    // MARK: - Initialization

    private init() {
        // Initialize embeddings synchronously using nonisolated helper
        // This avoids Swift 6 actor isolation errors in init
        initializeEmbeddings(for: .english)
    }

    /// Nonisolated initialization helper that sets up embedding models
    /// Called from init to avoid actor isolation issues
    nonisolated private func initializeEmbeddings(for language: NLLanguage) {
        // Load contextual embedding if available (iOS 17+)
        if #available(iOS 17.0, macOS 14.0, *) {
            if let contextual = NLContextualEmbedding(language: language) {
                if !contextual.hasAvailableAssets {
                    Task { try? await contextual.requestAssets() }
                }
                Task { await self.setContextualEmbedding(contextual) }
            }
        }

        // Load sentence embedding
        if let sentence = NLEmbedding.sentenceEmbedding(for: language) {
            Task { await self.setSentenceEmbedding(sentence) }
        }

        // Load word embedding as fallback
        if let word = NLEmbedding.wordEmbedding(for: language) {
            Task { await self.setWordEmbedding(word) }
        }

        Task { await self.setCurrentLanguage(language) }
    }

    // Actor-isolated setters for properties
    private func setContextualEmbedding(_ embedding: Any) {
        if #available(iOS 17.0, macOS 14.0, *) {
            self.contextualEmbedding = embedding as? NLContextualEmbedding
        }
    }

    private func setSentenceEmbedding(_ embedding: NLEmbedding) {
        self.sentenceEmbedding = embedding
    }

    private func setWordEmbedding(_ embedding: NLEmbedding) {
        self.wordEmbedding = embedding
    }

    private func setCurrentLanguage(_ language: NLLanguage) {
        self.currentLanguage = language
    }

    /// Load embedding models for a specific language
    /// If loading fails for the requested language, falls back to English to maintain service availability
    private func loadEmbeddings(for language: NLLanguage) {
        // Store previous state in case we need to fallback
        let previousLanguage = currentLanguage
        let previousSentence = sentenceEmbedding
        let previousWord = wordEmbedding
        let previousContextual: Any?
        if #available(iOS 17.0, macOS 14.0, *) {
            previousContextual = contextualEmbedding
        } else {
            previousContextual = nil
        }

        // Try to load embeddings for requested language
        var newContextual: Any?
        if #available(iOS 17.0, macOS 14.0, *) {
            if canAttemptContextualEmbedding {
                let contextual = NLContextualEmbedding(language: language)
                if let contextual {
                    if !contextual.hasAvailableAssets {
                        Task { try? await contextual.requestAssets() }
                    }
                    newContextual = contextual
                }
            }
        }

        let newSentence = NLEmbedding.sentenceEmbedding(for: language)
        let newWord = NLEmbedding.wordEmbedding(for: language)

        // Check if at least one embedding loaded successfully
        let hasNewContextualAssets: Bool = {
            if #available(iOS 17.0, macOS 14.0, *),
               let contextual = newContextual as? NLContextualEmbedding {
                return contextual.hasAvailableAssets
            }
            return false
        }()

        if hasNewContextualAssets || newSentence != nil || newWord != nil {
            currentLanguage = language
            if #available(iOS 17.0, macOS 14.0, *) {
                contextualEmbedding = newContextual as? NLContextualEmbedding
            }
            sentenceEmbedding = newSentence
            wordEmbedding = newWord

            if canAttemptContextualEmbedding,
               #available(iOS 17.0, macOS 14.0, *),
               let contextual = newContextual as? NLContextualEmbedding,
               contextual.hasAvailableAssets {
                DiagnosticsLogger.log("[AppleEmbedding] Loaded contextual embedding for \(language.rawValue), dimension=\(contextual.dimension)")
            } else if let newSentence {
                DiagnosticsLogger.log("[AppleEmbedding] Loaded sentence embedding for \(language.rawValue), dimension=\(newSentence.dimension)")
            } else if let newWord {
                DiagnosticsLogger.log("[AppleEmbedding] Sentence embedding not available for \(language.rawValue), using word embedding fallback, dimension=\(newWord.dimension)")
            }
        } else {
            // Loading failed - keep previous embeddings if available, or fallback to English
            DiagnosticsLogger.log("[AppleEmbedding] WARNING: No embedding available for \(language.rawValue), keeping \(previousLanguage.rawValue)")

            if previousSentence == nil && previousWord == nil && language != .english {
                // No previous embeddings and not already trying English - force English fallback
                DiagnosticsLogger.log("[AppleEmbedding] Falling back to English embeddings")
                var englishContextual: Any?
                if #available(iOS 17.0, macOS 14.0, *) {
                    englishContextual = NLContextualEmbedding(language: .english)
                    if let contextual = englishContextual as? NLContextualEmbedding, !contextual.hasAvailableAssets {
                        Task { try? await contextual.requestAssets() }
                    }
                }
                let englishSentence = NLEmbedding.sentenceEmbedding(for: .english)
                let englishWord = NLEmbedding.wordEmbedding(for: .english)

                let hasEnglishContextualAssets: Bool = {
                    if #available(iOS 17.0, macOS 14.0, *),
                       let contextual = englishContextual as? NLContextualEmbedding {
                        return contextual.hasAvailableAssets
                    }
                    return false
                }()

                if hasEnglishContextualAssets || englishSentence != nil || englishWord != nil {
                    currentLanguage = .english
                    if #available(iOS 17.0, macOS 14.0, *) {
                        contextualEmbedding = englishContextual as? NLContextualEmbedding
                    }
                    sentenceEmbedding = englishSentence
                    wordEmbedding = englishWord
                    DiagnosticsLogger.log("[AppleEmbedding] English fallback successful")
                }
            }
            // Otherwise keep previous language/embeddings unchanged
            if #available(iOS 17.0, macOS 14.0, *) {
                contextualEmbedding = previousContextual as? NLContextualEmbedding
            }
            sentenceEmbedding = previousSentence
            wordEmbedding = previousWord
        }
    }

    // MARK: - Public API

    /// Check if embedding service is available
    var isAvailable: Bool {
        hasContextualEmbedding || sentenceEmbedding != nil || wordEmbedding != nil
    }

    /// Check if sentence-level embeddings are available (higher quality)
    var hasSentenceEmbedding: Bool {
        sentenceEmbedding != nil
    }

    /// Generate embedding vector for a single text
    /// - Parameter text: Text to embed (sentence or paragraph)
    /// - Returns: Float array representing the embedding vector, or nil if unavailable
    func embed(text: String) -> [Float]? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        // Detect language and switch embedding if needed
        let detectedLanguage = detectLanguage(text: trimmed)
        if detectedLanguage != currentLanguage && supportedLanguages.contains(detectedLanguage) {
            loadEmbeddings(for: detectedLanguage)
        }

        let cacheLanguage = currentLanguage
        let defaultStrategy = preferredStrategy ?? .word
        let key = cacheKey(for: trimmed, language: cacheLanguage, strategy: defaultStrategy)
        if let cached = embeddingCache[key] {
            touchCacheKey(key)
            return cached
        }

        if let contextual = embedWithContextual(text: trimmed) {
            storeInCache(contextual, for: trimmed, language: cacheLanguage, strategy: .contextual)
            return contextual
        }

        if let result = Self.computeEmbedding(
            text: trimmed,
            sentenceEmbedding: sentenceEmbedding,
            wordEmbedding: wordEmbedding
        ) {
            storeInCache(result.vector, for: trimmed, language: cacheLanguage, strategy: result.strategy)
            return result.vector
        }

        return nil
    }

    /// Generate embeddings for multiple texts in batch
    /// - Parameter texts: Array of texts to embed
    /// - Returns: Array of embedding vectors (same order as input, nil for failed embeddings)
    ///
    /// NOTE: NLEmbedding is NOT thread-safe. This method processes texts sequentially
    /// within the actor to avoid concurrent access crashes.
    func embedBatch(texts: [String]) async -> [[Float]?] {
        guard !texts.isEmpty else { return [] }

        // Normalize inputs and filter out empties early
        let normalized = texts.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        let nonEmpty = normalized.filter { !$0.isEmpty }
        guard !nonEmpty.isEmpty else { return Array(repeating: nil, count: texts.count) }

        // Pick the dominant language for the batch to avoid thrashing models
        // Only switch if the language is supported AND has available embeddings
        if let batchLanguage = dominantLanguage(for: nonEmpty),
           batchLanguage != currentLanguage,
           supportedLanguages.contains(batchLanguage),
           hasAvailableEmbedding(for: batchLanguage) {
            loadEmbeddings(for: batchLanguage)
        }

        // Ensure we have at least one embedding model
        guard hasContextualEmbedding || sentenceEmbedding != nil || wordEmbedding != nil else {
            return Array(repeating: nil, count: texts.count)
        }

        let cacheLanguage = currentLanguage
        var results = Array<[Float]?>(repeating: nil, count: texts.count)

        // Process each text sequentially (NLEmbedding is NOT thread-safe)
        // Using concurrent TaskGroup with NLEmbedding causes crashes
        for (index, text) in normalized.enumerated() {
            guard !text.isEmpty else { continue }

            // Check cache first
            let defaultStrategy = preferredStrategy ?? .word
            let key = cacheKey(for: text, language: cacheLanguage, strategy: defaultStrategy)
            if let cached = embeddingCache[key] {
                results[index] = cached
                touchCacheKey(key)
                continue
            }

            // Compute embedding (sequential, within actor)
            if let contextual = embedWithContextual(text: text) {
                results[index] = contextual
                storeInCache(contextual, for: text, language: cacheLanguage, strategy: .contextual)
            } else if let result = Self.computeEmbedding(
                text: text,
                sentenceEmbedding: sentenceEmbedding,
                wordEmbedding: wordEmbedding
            ) {
                results[index] = result.vector
                storeInCache(result.vector, for: text, language: cacheLanguage, strategy: result.strategy)
            }

            // Yield periodically to avoid blocking the actor for too long
            // This allows other actor-isolated work to proceed
            if index % 10 == 9 {
                await Task.yield()
            }
        }

        return results
    }

    private func embedWithContextual(text: String) -> [Float]? {
        guard hasContextualEmbedding else { return nil }
        guard #available(iOS 17.0, macOS 14.0, *), let contextualEmbedding else { return nil }

        do {
            let result = try contextualEmbedding.embeddingResult(for: text, language: currentLanguage)
            var sumVector = [Float](repeating: 0, count: contextualEmbedding.dimension)
            var tokenCount = 0

            result.enumerateTokenVectors(in: text.startIndex..<text.endIndex) { vector, _ in
                for i in 0..<vector.count {
                    sumVector[i] += Float(vector[i])
                }
                tokenCount += 1
                return true
            }

            guard tokenCount > 0 else { return nil }
            return sumVector.map { $0 / Float(tokenCount) }
        } catch {
            handleContextualEmbeddingFailure(error)
            return nil
        }
    }

    private func handleContextualEmbeddingFailure(_ error: Error) {
        let permissionIssue = Self.isLikelyPermissionIssue(error)

        if permissionIssue {
            contextualEmbeddingDisabledUntil = .distantFuture
#if targetEnvironment(simulator)
            contextualEmbeddingDisabledReason = "simulator sandbox permissions"
#else
            contextualEmbeddingDisabledReason = "permission denied"
#endif
        } else {
            contextualEmbeddingDisabledUntil = Date().addingTimeInterval(5 * 60)
            contextualEmbeddingDisabledReason = "runtime error"
        }

        if #available(iOS 17.0, macOS 14.0, *) {
            contextualEmbedding = nil
        }

        if FeatureFlags.AppleEmbeddingDebugLogging {
            let reason = contextualEmbeddingDisabledReason ?? "unknown"
            let until = contextualEmbeddingDisabledUntil
            let untilText = (until == .distantFuture) ? "app restart" : "about 5 minutes"
            DiagnosticsLogger.log("[AppleEmbedding] Contextual embedding disabled (\(reason)) until \(untilText): \(error.localizedDescription)")
        }
    }

    private static func isLikelyPermissionIssue(_ error: Error) -> Bool {
        let nsError = error as NSError

        if nsError.domain == NSCocoaErrorDomain {
            if nsError.code == CocoaError.Code.fileReadNoPermission.rawValue ||
                nsError.code == CocoaError.Code.fileWriteNoPermission.rawValue {
                return true
            }
        }

        if nsError.domain == NSPOSIXErrorDomain, nsError.code == 1 || nsError.code == 13 {
            return true
        }

        if nsError.localizedDescription.range(of: "permission", options: .caseInsensitive) != nil {
            return true
        }

        if let underlying = nsError.userInfo[NSUnderlyingErrorKey] as? Error {
            return isLikelyPermissionIssue(underlying)
        }

        return false
    }

    /// Generate embeddings for multiple texts, filtering out failures
    /// - Parameter texts: Array of texts to embed
    /// - Returns: Array of (index, embedding) tuples for successful embeddings
    func embedBatchWithIndices(texts: [String]) async -> [(index: Int, embedding: [Float])] {
        let embeddings = await embedBatch(texts: texts)
        var results: [(index: Int, embedding: [Float])] = []
        for (index, value) in embeddings.enumerated() {
            if let embedding = value {
                results.append((index: index, embedding: embedding))
            }
        }
        return results
    }

    /// Calculate cosine similarity between two texts
    /// - Returns: Similarity score (0.0 to 1.0), or nil if embedding fails
    func similarity(text1: String, text2: String) -> Double? {
        guard let vec1 = embed(text: text1),
              let vec2 = embed(text: text2),
              let cosine = cosineSimilarity(vector1: vec1, vector2: vec2) else {
            return nil
        }
        return Double(cosine)
    }

    /// Calculate cosine similarity between two embedding vectors
    /// - Returns: Similarity score (-1.0 to 1.0)
    func cosineSimilarity(vector1: [Float], vector2: [Float]) -> Float? {
        guard vector1.count == vector2.count, !vector1.isEmpty else { return nil }

        let length = vDSP_Length(vector1.count)

        var dot: Float = 0
        vDSP_dotpr(vector1, 1, vector2, 1, &dot, length)

        var norm1: Float = 0
        vDSP_svesq(vector1, 1, &norm1, length)

        var norm2: Float = 0
        vDSP_svesq(vector2, 1, &norm2, length)

        let denominator = sqrt(norm1) * sqrt(norm2)
        guard denominator > 0 else { return nil }

        return dot / denominator
    }

    /// Find the most similar text from a list
    /// - Parameters:
    ///   - query: The query text
    ///   - candidates: List of candidate texts to compare against
    ///   - topK: Number of top results to return
    /// - Returns: Array of (index, similarity) tuples, sorted by similarity descending
    func findSimilar(query: String, candidates: [String], topK: Int = 5) -> [(index: Int, similarity: Double)] {
        guard let queryEmbedding = embed(text: query) else { return [] }

        var results: [(index: Int, similarity: Double)] = []

        for (index, candidate) in candidates.enumerated() {
            if let candidateEmbedding = embed(text: candidate),
               let similarity = cosineSimilarity(vector1: queryEmbedding, vector2: candidateEmbedding) {
                results.append((index: index, similarity: Double(similarity)))
            }
        }

        // Sort by similarity descending and take top K
        return results
            .sorted { $0.similarity > $1.similarity }
            .prefix(topK)
            .map { $0 }
    }

    // MARK: - Language Detection

    /// Detect the dominant language of the text
    private func detectLanguage(text: String) -> NLLanguage {
        let recognizer = NLLanguageRecognizer()
        recognizer.processString(text)

        if let language = recognizer.dominantLanguage {
            return language
        }

        return .english // Default fallback
    }

    /// Determine the dominant language for a batch to reduce model swaps
    private func dominantLanguage(for texts: [String]) -> NLLanguage? {
        var counts: [NLLanguage: Int] = [:]
        for text in texts.prefix(10) { // limit work for long batches
            let language = detectLanguage(text: text)
            if supportedLanguages.contains(language) {
                counts[language, default: 0] += 1
            }
        }

        return counts.max { $0.value < $1.value }?.key
    }

    /// Get supported languages for embedding
    func getSupportedLanguages() -> [NLLanguage] {
        return supportedLanguages
    }

    /// Check if a language is supported
    func isLanguageSupported(_ language: NLLanguage) -> Bool {
        return supportedLanguages.contains(language)
    }

    /// Check if actual embedding models are available for a language on this device
    /// This verifies the model is downloaded, not just theoretically supported
    private func hasAvailableEmbedding(for language: NLLanguage) -> Bool {
        // Check if at least one type of embedding can be loaded for this language
        if #available(iOS 17.0, macOS 14.0, *) {
            if canAttemptContextualEmbedding,
               let contextual = NLContextualEmbedding(language: language),
               contextual.hasAvailableAssets {
                return true
            }
        }
        let sentence = NLEmbedding.sentenceEmbedding(for: language)
        let word = NLEmbedding.wordEmbedding(for: language)
        return sentence != nil || word != nil
    }

    // MARK: - Private Helpers

    // MARK: - Diagnostics

    /// Get diagnostic information about the embedding service
    func getDiagnostics() -> [String: Any] {
        var diagnostics: [String: Any] = [
            "isAvailable": isAvailable,
            "hasContextualEmbedding": hasContextualEmbedding,
            "hasSentenceEmbedding": hasSentenceEmbedding,
            "currentLanguage": currentLanguage.rawValue,
            "dimension": dimension,
            "supportedLanguages": supportedLanguages.map { $0.rawValue },
            "strategy": (preferredStrategy ?? .word).rawValue,
            "modelVersion": modelVersion
        ]

        if let identifier = contextualModelIdentifier {
            diagnostics["modelIdentifier"] = identifier
        }

        return diagnostics
    }
}

// MARK: - Convenience Extensions

extension AppleEmbeddingService {
    /// Embed text and return as base64 encoded string (for JSON transport)
    func embedAsBase64(text: String) -> String? {
        guard let embedding = embed(text: text) else { return nil }

        // Convert [Float] to Data
        let data = embedding.withUnsafeBufferPointer { buffer in
            Data(buffer: buffer)
        }

        return data.base64EncodedString()
    }

    /// Decode base64 embedding back to [Float]
    static func decodeBase64Embedding(_ base64: String) -> [Float]? {
        guard let data = Data(base64Encoded: base64) else { return nil }

        let count = data.count / MemoryLayout<Float>.size
        var floats = [Float](repeating: 0, count: count)

        _ = floats.withUnsafeMutableBufferPointer { buffer in
            data.copyBytes(to: buffer)
        }

        return floats
    }
}

// MARK: - Internal helpers

private extension AppleEmbeddingService {
    /// Compute embedding using provided models without touching actor state.
    static func computeEmbedding(
        text: String,
        sentenceEmbedding: NLEmbedding?,
        wordEmbedding: NLEmbedding?
    ) -> (vector: [Float], strategy: EmbeddingStrategy)? {
        if let embedding = sentenceEmbedding,
           let vector = embedding.vector(for: text) {
            return (vector.map { Float($0) }, .sentence)
        }

        if let embedding = wordEmbedding {
            if let vector = averageWordEmbedding(text: text, embedding: embedding) {
                return (vector, .word)
            }
        }

        return nil
    }

    /// Average word embeddings as fallback for sentence embedding
    static func averageWordEmbedding(text: String, embedding: NLEmbedding) -> [Float]? {
        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.string = text

        var vectors: [[Double]] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let word = String(text[range]).lowercased()
            if let vector = embedding.vector(for: word) {
                vectors.append(vector)
            }
            return true
        }

        guard !vectors.isEmpty else { return nil }

        // Average all word vectors
        let dim = vectors[0].count
        var averaged = [Double](repeating: 0.0, count: dim)
        for vector in vectors {
            guard vector.count == dim else { continue }
            for i in 0..<dim {
                averaged[i] += vector[i]
            }
        }

        let count = Double(vectors.count)
        return averaged.map { Float($0 / count) }
    }

    func cacheKey(for text: String, language: NLLanguage, strategy: EmbeddingStrategy) -> String {
        let shortened = text.count > 256 ? String(text.prefix(256)) : text
        return "\(language.rawValue)|\(strategy.rawValue)|\(shortened)"
    }

    func storeInCache(_ embedding: [Float], for text: String, language: NLLanguage, strategy: EmbeddingStrategy) {
        let key = cacheKey(for: text, language: language, strategy: strategy)
        embeddingCache[key] = embedding
        cacheOrder.removeAll { $0 == key }
        cacheOrder.append(key)

        // Enforce LRU bound
        if cacheOrder.count > maxCacheEntries {
            let overflow = cacheOrder.count - maxCacheEntries
            let evicted = cacheOrder.prefix(overflow)
            for key in evicted {
                embeddingCache[key] = nil
            }
            cacheOrder.removeFirst(overflow)
        }
    }

    func touchCacheKey(_ key: String) {
        cacheOrder.removeAll { $0 == key }
        cacheOrder.append(key)
    }
}
