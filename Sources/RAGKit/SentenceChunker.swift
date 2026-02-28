import Foundation
import NaturalLanguage

// MARK: - Sentence Chunk Configuration

/// Configuration for sentence-based micro-chunking optimized for high content yield
public struct SentenceChunkConfig: Sendable {
    /// Target sentences per chunk (used as upper bound; actual limit is maxContentChars)
    public let sentencesPerChunk: Int

    /// Minimum sentences to form a chunk (avoid fragments)
    public let minSentences: Int

    /// Overlap sentences for context continuity
    public let overlapSentences: Int

    /// Target flashcards per micro-chunk
    public let flashcardsPerMicroChunk: Int

    /// Target quiz questions per micro-chunk
    public let quizQuestionsPerMicroChunk: Int

    /// Maximum characters of content per chunk.
    /// This is the primary size constraint — chunks stop accumulating sentences once
    /// the joined content exceeds this limit. Sized to fit within the prompt budget
    /// (2,400 chars total) after accounting for instruction text (~200 chars) and
    /// optional context fields (~500 chars). Safe content budget is ~1,000-1,200 chars.
    public let maxContentChars: Int

    /// Maximum total chunks produced from any input. Prevents runaway processing on
    /// very long documents. When exceeded, the chunker re-chunks with a larger window.
    public let maxChunks: Int

    public init(
        sentencesPerChunk: Int = 4,
        minSentences: Int = 2,
        overlapSentences: Int = 1,
        flashcardsPerMicroChunk: Int = 2,
        quizQuestionsPerMicroChunk: Int = 1,
        maxContentChars: Int = 1_000,
        maxChunks: Int = 60
    ) {
        self.sentencesPerChunk = sentencesPerChunk
        self.minSentences = minSentences
        self.overlapSentences = overlapSentences
        self.flashcardsPerMicroChunk = flashcardsPerMicroChunk
        self.quizQuestionsPerMicroChunk = quizQuestionsPerMicroChunk
        self.maxContentChars = maxContentChars
        self.maxChunks = maxChunks
    }

    /// Default configuration: up to 6 sentences, max 1000 chars → 3 flashcards + 1 quiz
    /// Fits safely within Apple FM 2,400 char prompt budget with room for context fields.
    public static let standard = SentenceChunkConfig(
        sentencesPerChunk: 6,
        minSentences: 2,
        overlapSentences: 1,
        flashcardsPerMicroChunk: 3,
        quizQuestionsPerMicroChunk: 1,
        maxContentChars: 1_000,
        maxChunks: 80
    )

    /// Dense configuration: up to 4 sentences, max 700 chars → 2 flashcards + 1 quiz
    public static let dense = SentenceChunkConfig(
        sentencesPerChunk: 4,
        minSentences: 2,
        overlapSentences: 1,
        flashcardsPerMicroChunk: 2,
        quizQuestionsPerMicroChunk: 1,
        maxContentChars: 700,
        maxChunks: 100
    )

    /// Balanced configuration: up to 8 sentences, max 1200 chars → 4 flashcards + 1 quiz
    public static let balanced = SentenceChunkConfig(
        sentencesPerChunk: 8,
        minSentences: 3,
        overlapSentences: 1,
        flashcardsPerMicroChunk: 4,
        quizQuestionsPerMicroChunk: 1,
        maxContentChars: 1_200,
        maxChunks: 60
    )

    /// High-yield configuration: up to 10 sentences, max 1500 chars → 4 flashcards + 1 quiz
    public static let highYield = SentenceChunkConfig(
        sentencesPerChunk: 10,
        minSentences: 4,
        overlapSentences: 1,
        flashcardsPerMicroChunk: 4,
        quizQuestionsPerMicroChunk: 1,
        maxContentChars: 1_500,
        maxChunks: 50
    )

    /// Legacy thorough configuration
    public static let thorough = SentenceChunkConfig(
        sentencesPerChunk: 6,
        minSentences: 2,
        overlapSentences: 1,
        flashcardsPerMicroChunk: 3,
        quizQuestionsPerMicroChunk: 1,
        maxContentChars: 1_000,
        maxChunks: 80
    )
}

// MARK: - Sentence Chunk

/// A micro-chunk of text containing a small number of sentences
public struct SentenceChunk: Identifiable, Sendable {
    public let id: Int
    public let sentences: [String]
    public let content: String
    public let sentenceCount: Int
    public let startSentenceIndex: Int
    public let endSentenceIndex: Int
    public let contextFromPrevious: String?

    public init(
        id: Int,
        sentences: [String],
        startSentenceIndex: Int,
        contextFromPrevious: String? = nil
    ) {
        self.id = id
        self.sentences = sentences
        self.content = sentences.joined(separator: " ")
        self.sentenceCount = sentences.count
        self.startSentenceIndex = startSentenceIndex
        self.endSentenceIndex = startSentenceIndex + sentences.count - 1
        self.contextFromPrevious = contextFromPrevious
    }

    /// Estimated token count for this chunk
    public var estimatedTokens: Int {
        Int(ceil(Double(content.count) / 4.0))
    }
}

// MARK: - Sentence Chunker

/// Chunks text into small sentence-based micro-chunks for high-yield content generation
/// Thread-safe: Creates NLTokenizer per-call to avoid data races
public final class SentenceChunker: Sendable {

    private let config: SentenceChunkConfig
    private let maxSentenceChars = 500

    public init(config: SentenceChunkConfig = .standard) {
        self.config = config
        // NOTE: NLTokenizer is NOT stored as instance property because it's not thread-safe
        // Each method creates its own tokenizer instance
    }

    // MARK: - Public API

    /// Chunk text into sentence-based micro-chunks
    /// - Parameter text: The full transcript or document
    /// - Returns: Array of SentenceChunk objects for sequential processing
    public func chunk(_ text: String) -> [SentenceChunk] {
        let cleanedText = preprocessText(text)
        let sentences = extractSentences(cleanedText)

        guard !sentences.isEmpty else {
            return []
        }

        DiagnosticsLogger.log("[SentenceChunker] Extracted \(sentences.count) sentences from text")

        // If all sentences fit in one chunk (both count and char budget), return single chunk
        let totalChars = sentences.joined(separator: " ").count
        if sentences.count <= config.sentencesPerChunk && totalChars <= config.maxContentChars {
            return [SentenceChunk(
                id: 0,
                sentences: sentences,
                startSentenceIndex: 0,
                contextFromPrevious: nil
            )]
        }

        // Create budget-aware overlapping chunks
        var chunks = createBudgetAwareChunks(from: sentences)

        // Safety valve: if too many chunks, re-chunk with larger budget
        if chunks.count > config.maxChunks {
            DiagnosticsLogger.log("[SentenceChunker] \(chunks.count) chunks exceeds max \(config.maxChunks), re-chunking with larger budget")
            let scaleFactor = Double(chunks.count) / Double(config.maxChunks)
            let expandedBudget = Int(Double(config.maxContentChars) * scaleFactor * 1.1)
            chunks = createBudgetAwareChunks(from: sentences, contentBudgetOverride: expandedBudget)
            DiagnosticsLogger.log("[SentenceChunker] Re-chunked to \(chunks.count) chunks (budget: \(expandedBudget) chars)")
        }

        return chunks
    }

    /// Get the total number of chunks that would be created from text
    public func estimateChunkCount(_ text: String) -> Int {
        let sentences = extractSentences(preprocessText(text))
        let totalChars = sentences.joined(separator: " ").count
        if sentences.count <= config.sentencesPerChunk && totalChars <= config.maxContentChars {
            return 1
        }
        // Estimate average sentence length and how many fit per budget
        let avgSentenceChars = totalChars / max(1, sentences.count)
        let sentencesPerBudget = max(1, config.maxContentChars / max(1, avgSentenceChars))
        let effectiveSentencesPerChunk = min(config.sentencesPerChunk, sentencesPerBudget)
        let effectiveStep = max(1, effectiveSentencesPerChunk - config.overlapSentences)
        return min(config.maxChunks, max(1, Int(ceil(Double(sentences.count) / Double(effectiveStep)))))
    }

    /// Estimate total flashcards and quiz questions that would be generated
    public func estimateYield(_ text: String) -> (flashcards: Int, quizQuestions: Int) {
        let chunkCount = estimateChunkCount(text)
        return (
            flashcards: chunkCount * config.flashcardsPerMicroChunk,
            quizQuestions: chunkCount * config.quizQuestionsPerMicroChunk
        )
    }

    // MARK: - Sentence Extraction

    private func extractSentences(_ text: String) -> [String] {
        var sentences: [String] = []

        func splitIfTooLong(_ sentence: String) -> [String] {
            let trimmed = sentence.trimmingCharacters(in: .whitespacesAndNewlines)
            guard trimmed.count > maxSentenceChars else { return [trimmed] }

            var parts: [String] = []
            parts.reserveCapacity(max(2, trimmed.count / maxSentenceChars))

            var current = ""
            for word in trimmed.split(whereSeparator: \.isWhitespace) {
                let w = String(word)
                if current.isEmpty {
                    current = w
                    continue
                }
                if current.count + 1 + w.count <= maxSentenceChars {
                    current += " " + w
                } else {
                    parts.append(current)
                    current = w
                }
            }
            if !current.isEmpty { parts.append(current) }

            return parts.filter { $0.count >= 10 }
        }

        // THREAD-SAFETY FIX: Create a new tokenizer per call
        // NLTokenizer has mutable internal state and is NOT thread-safe
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = text
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let sentence = String(text[range]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !sentence.isEmpty && sentence.count >= 10 { // Filter out very short fragments
                sentences.append(contentsOf: splitIfTooLong(sentence))
            }
            return true
        }

        // Fallback for text without clear sentence boundaries
        if sentences.isEmpty {
            let rough = text.components(separatedBy: CharacterSet(charactersIn: ".!?"))
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty && $0.count >= 10 }
            for s in rough {
                sentences.append(contentsOf: splitIfTooLong(s))
            }
        }

        return sentences
    }

    // MARK: - Chunk Creation

    /// Creates chunks using character budget as the primary size constraint.
    /// Sentences are accumulated until adding the next sentence would exceed `maxContentChars`.
    /// The sentence-count cap (`sentencesPerChunk`) acts as a secondary upper bound.
    /// Overlap sentences are passed as `contextFromPrevious` only — they are NOT included
    /// in the chunk's `sentences` array, preventing double-counting in the prompt.
    private func createBudgetAwareChunks(from sentences: [String], contentBudgetOverride: Int? = nil) -> [SentenceChunk] {
        let budget = contentBudgetOverride ?? config.maxContentChars
        var chunks: [SentenceChunk] = []
        var chunkId = 0
        var cursor = 0

        while cursor < sentences.count {
            var chunkSentences: [String] = []
            var chunkChars = 0
            var i = cursor

            // Accumulate sentences until we hit the budget or sentence-count cap
            while i < sentences.count && chunkSentences.count < config.sentencesPerChunk {
                let sentence = sentences[i]
                let addedChars = chunkChars == 0 ? sentence.count : sentence.count + 1 // +1 for space separator
                if chunkChars + addedChars > budget && !chunkSentences.isEmpty {
                    break
                }
                // Always include at least one sentence even if it exceeds budget
                chunkSentences.append(sentence)
                chunkChars += addedChars
                i += 1
            }

            // Skip tiny fragments unless this is the last remaining content
            if chunkSentences.count < config.minSentences && i < sentences.count {
                // Merge remaining sentences into the previous chunk if possible
                if let lastChunk = chunks.last {
                    let merged = lastChunk.sentences + chunkSentences
                    chunks[chunks.count - 1] = SentenceChunk(
                        id: lastChunk.id,
                        sentences: merged,
                        startSentenceIndex: lastChunk.startSentenceIndex,
                        contextFromPrevious: lastChunk.contextFromPrevious
                    )
                    cursor = i
                    continue
                }
                // No previous chunk to merge into — include as-is
            }

            // Build overlap context from sentences BEFORE this chunk (not included in chunk content)
            var context: String? = nil
            if chunkId > 0 && config.overlapSentences > 0 {
                let overlapStart = max(0, cursor - config.overlapSentences)
                let overlapSentences = Array(sentences[overlapStart..<cursor])
                if !overlapSentences.isEmpty {
                    context = overlapSentences.joined(separator: " ")
                }
            }

            chunks.append(SentenceChunk(
                id: chunkId,
                sentences: chunkSentences,
                startSentenceIndex: cursor,
                contextFromPrevious: context
            ))

            chunkId += 1
            // Advance cursor by the number of new sentences (no overlap in content)
            cursor = i
        }

        DiagnosticsLogger.log("[SentenceChunker] Created \(chunks.count) budget-aware chunks (budget: \(budget) chars)")
        return chunks
    }

    // MARK: - Preprocessing

    private func preprocessText(_ text: String) -> String {
        var result = text

        // Normalize line endings
        result = result.replacingOccurrences(of: "\r\n", with: " ")
        result = result.replacingOccurrences(of: "\r", with: " ")
        result = result.replacingOccurrences(of: "\n", with: " ")

        // Collapse whitespace (single pass; avoids quadratic loops on long transcripts)
        result = result.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)

        // Handle common transcription artifacts
        result = result.replacingOccurrences(of: "...", with: ".")
        result = result.replacingOccurrences(of: "..", with: ".")

        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

// MARK: - Convenience Extensions

extension SentenceChunker {
    /// Create chunker optimized for maximum content yield per model call
    public static var highYield: SentenceChunker {
        SentenceChunker(config: .highYield)
    }

    /// Create chunker with standard settings (6 sentences/chunk)
    public static var standard: SentenceChunker {
        SentenceChunker(config: .standard)
    }

    /// Create chunker with balanced settings (8 sentences/chunk)
    public static var balanced: SentenceChunker {
        SentenceChunker(config: .balanced)
    }

    /// Create chunker for finer granularity (4 sentences/chunk)
    public static var dense: SentenceChunker {
        SentenceChunker(config: .dense)
    }

    /// Create chunker for thorough processing
    public static var thorough: SentenceChunker {
        SentenceChunker(config: .thorough)
    }
}
