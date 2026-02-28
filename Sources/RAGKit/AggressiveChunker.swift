import Foundation
import NaturalLanguage

// MARK: - Chunk Configuration

/// Configuration for aggressive text chunking optimized for small on-device models
public struct ChunkConfig: Sendable {
    /// Maximum tokens per chunk (Apple Foundation Model has ~4K context)
    /// We use 400 tokens to leave room for system prompt + output
    public let maxTokens: Int

    /// Overlap tokens for context continuity between chunks
    public let overlapTokens: Int

    /// Minimum chunk size to avoid tiny fragments
    public let minChunkTokens: Int

    /// Target flashcards to generate per chunk
    public let flashcardsPerChunk: Int

    /// Approximate characters per token (for estimation)
    public static let charsPerToken: Double = 4.0

    public init(
        maxTokens: Int = 400,
        overlapTokens: Int = 50,
        minChunkTokens: Int = 100,
        flashcardsPerChunk: Int = 4
    ) {
        self.maxTokens = maxTokens
        self.overlapTokens = overlapTokens
        self.minChunkTokens = minChunkTokens
        self.flashcardsPerChunk = flashcardsPerChunk
    }

    /// Aggressive config for very small models (2-3B params)
    public static let aggressive = ChunkConfig(
        maxTokens: 350,
        overlapTokens: 40,
        minChunkTokens: 80,
        flashcardsPerChunk: 3
    )

    /// Standard config for Apple Foundation Models (~3B on-device)
    public static let appleFoundation = ChunkConfig(
        maxTokens: 500,
        overlapTokens: 60,
        minChunkTokens: 120,
        flashcardsPerChunk: 4
    )

    /// Larger config for medium models (7-8B params)
    public static let medium = ChunkConfig(
        maxTokens: 800,
        overlapTokens: 100,
        minChunkTokens: 200,
        flashcardsPerChunk: 6
    )

    /// Note: For sentence-based micro-chunking with higher yield,
    /// use SentenceChunkConfig and SequentialMicroPipeline instead.
    /// Micro-chunking processes 4-5 sentences at a time for 2-3 flashcards + 1 quiz each.
}

// MARK: - Text Chunk

/// Represents a chunk of text with metadata for sequential processing
public struct TextChunk: Identifiable, Sendable {
    public let id: Int
    public let content: String
    public let estimatedTokens: Int
    public let startOffset: Int
    public let endOffset: Int
    public let previousSummary: String?

    public init(
        id: Int,
        content: String,
        estimatedTokens: Int,
        startOffset: Int,
        endOffset: Int,
        previousSummary: String? = nil
    ) {
        self.id = id
        self.content = content
        self.estimatedTokens = estimatedTokens
        self.startOffset = startOffset
        self.endOffset = endOffset
        self.previousSummary = previousSummary
    }

    /// Create a copy with an updated summary from prior processing
    public func withSummary(_ summary: String) -> TextChunk {
        TextChunk(
            id: id,
            content: content,
            estimatedTokens: estimatedTokens,
            startOffset: startOffset,
            endOffset: endOffset,
            previousSummary: summary
        )
    }
}

// MARK: - Aggressive Chunker

/// Aggressively chunks text for small on-device language models
/// Prioritizes semantic boundaries (paragraphs, sentences) while enforcing strict token limits
///
/// SAFETY(@unchecked Sendable): `final` class. NLTokenizer is not Sendable but instances
/// are only used within synchronous method scope (never shared across threads).
public final class AggressiveChunker: @unchecked Sendable {

    private let config: ChunkConfig
    private let sentenceTokenizer: NLTokenizer
    private let paragraphTokenizer: NLTokenizer

    public init(config: ChunkConfig = .appleFoundation) {
        self.config = config
        self.sentenceTokenizer = NLTokenizer(unit: .sentence)
        self.paragraphTokenizer = NLTokenizer(unit: .paragraph)
    }

    // MARK: - Public API

    /// Chunk text aggressively for small model processing
    /// - Parameter text: The full transcript or document text
    /// - Returns: Array of TextChunk objects ready for sequential processing
    public func chunk(_ text: String) -> [TextChunk] {
        let cleanedText = preprocessText(text)

        guard !cleanedText.isEmpty else {
            return []
        }

        // Estimate total tokens
        let totalTokens = estimateTokens(cleanedText)

        // If text is small enough, return as single chunk
        if totalTokens <= config.maxTokens {
            return [TextChunk(
                id: 0,
                content: cleanedText,
                estimatedTokens: totalTokens,
                startOffset: 0,
                endOffset: cleanedText.count,
                previousSummary: nil
            )]
        }

        // Try semantic chunking strategies in order of preference
        var chunks = chunkByParagraphsWithOverlap(cleanedText)

        // If paragraph chunking produced chunks that are too large, refine with sentences
        if chunks.contains(where: { $0.estimatedTokens > config.maxTokens }) {
            chunks = refineLargeChunks(chunks)
        }

        // Final pass: hard split any remaining oversized chunks
        chunks = enforceTokenLimits(chunks)

        // Merge tiny chunks
        chunks = mergeTinyChunks(chunks)

        DiagnosticsLogger.log("[AggressiveChunker] Created \(chunks.count) chunks from \(totalTokens) estimated tokens")

        return chunks
    }

    /// Estimate token count for text (rough approximation)
    public func estimateTokens(_ text: String) -> Int {
        // Simple estimation: ~4 characters per token for English text
        return Int(ceil(Double(text.count) / ChunkConfig.charsPerToken))
    }

    // MARK: - Preprocessing

    private func preprocessText(_ text: String) -> String {
        var result = text

        // Normalize whitespace
        result = result.replacingOccurrences(of: "\r\n", with: "\n")
        result = result.replacingOccurrences(of: "\r", with: "\n")

        // Collapse whitespace (single pass; avoids quadratic loops on long documents)
        result = result.replacingOccurrences(of: "\\n{3,}", with: "\n\n", options: .regularExpression)
        result = result.replacingOccurrences(of: "[ \\t]{2,}", with: " ", options: .regularExpression)

        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Paragraph-Based Chunking

    private func chunkByParagraphsWithOverlap(_ text: String) -> [TextChunk] {
        let paragraphs = extractParagraphs(text)

        guard !paragraphs.isEmpty else {
            return [createChunk(id: 0, content: text, startOffset: 0)]
        }

        var chunks: [TextChunk] = []
        var currentContent = ""
        var currentStartOffset = 0
        var chunkId = 0
        var overlapContent = ""

        for (index, paragraph) in paragraphs.enumerated() {
            let paragraphTokens = estimateTokens(paragraph.content)
            let currentTokens = estimateTokens(currentContent)

            // If this single paragraph exceeds limit, it needs sentence-level splitting
            if paragraphTokens > config.maxTokens {
                // First, save current chunk if not empty
                if !currentContent.isEmpty {
                    chunks.append(createChunk(
                        id: chunkId,
                        content: currentContent,
                        startOffset: currentStartOffset
                    ))
                    chunkId += 1
                    overlapContent = extractOverlapFromEnd(currentContent)
                }

                // Add large paragraph as-is (will be refined later)
                let paragraphChunk = createChunk(
                    id: chunkId,
                    content: overlapContent.isEmpty ? paragraph.content : overlapContent + "\n\n" + paragraph.content,
                    startOffset: paragraph.offset
                )
                chunks.append(paragraphChunk)
                chunkId += 1

                currentContent = ""
                currentStartOffset = index + 1 < paragraphs.count ? paragraphs[index + 1].offset : text.count
                overlapContent = extractOverlapFromEnd(paragraph.content)
                continue
            }

            // Check if adding this paragraph exceeds limit
            let combinedTokens = currentTokens + paragraphTokens + (currentContent.isEmpty ? 0 : 10) // ~10 tokens for separator

            if combinedTokens > config.maxTokens && !currentContent.isEmpty {
                // Save current chunk
                chunks.append(createChunk(
                    id: chunkId,
                    content: currentContent,
                    startOffset: currentStartOffset
                ))
                chunkId += 1

                // Start new chunk with overlap
                overlapContent = extractOverlapFromEnd(currentContent)
                currentContent = overlapContent.isEmpty ? paragraph.content : overlapContent + "\n\n" + paragraph.content
                currentStartOffset = paragraph.offset
            } else {
                // Add to current chunk
                if currentContent.isEmpty {
                    currentContent = overlapContent.isEmpty ? paragraph.content : overlapContent + "\n\n" + paragraph.content
                    currentStartOffset = paragraph.offset
                    overlapContent = ""
                } else {
                    currentContent += "\n\n" + paragraph.content
                }
            }
        }

        // Don't forget the last chunk
        if !currentContent.isEmpty {
            chunks.append(createChunk(
                id: chunkId,
                content: currentContent,
                startOffset: currentStartOffset
            ))
        }

        return chunks
    }

    private struct ParagraphInfo {
        let content: String
        let offset: Int
    }

    private func extractParagraphs(_ text: String) -> [ParagraphInfo] {
        var paragraphs: [ParagraphInfo] = []

        paragraphTokenizer.string = text
        paragraphTokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let content = String(text[range]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !content.isEmpty {
                let offset = text.distance(from: text.startIndex, to: range.lowerBound)
                paragraphs.append(ParagraphInfo(content: content, offset: offset))
            }
            return true
        }

        // Fallback if NLTokenizer doesn't find paragraphs
        if paragraphs.isEmpty {
            let parts = text.components(separatedBy: "\n\n")
            var offset = 0
            for part in parts {
                let trimmed = part.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    paragraphs.append(ParagraphInfo(content: trimmed, offset: offset))
                }
                offset += part.count + 2 // +2 for \n\n
            }
        }

        return paragraphs
    }

    // MARK: - Sentence-Based Refinement

    private func refineLargeChunks(_ chunks: [TextChunk]) -> [TextChunk] {
        var refinedChunks: [TextChunk] = []
        var chunkId = 0

        for chunk in chunks {
            if chunk.estimatedTokens <= config.maxTokens {
                refinedChunks.append(TextChunk(
                    id: chunkId,
                    content: chunk.content,
                    estimatedTokens: chunk.estimatedTokens,
                    startOffset: chunk.startOffset,
                    endOffset: chunk.endOffset,
                    previousSummary: chunk.previousSummary
                ))
                chunkId += 1
            } else {
                // Split by sentences
                let sentenceChunks = chunkBySentences(chunk.content, startingOffset: chunk.startOffset, startingId: chunkId)
                refinedChunks.append(contentsOf: sentenceChunks)
                chunkId += sentenceChunks.count
            }
        }

        return refinedChunks
    }

    private func chunkBySentences(_ text: String, startingOffset: Int, startingId: Int) -> [TextChunk] {
        let sentences = extractSentences(text)

        guard !sentences.isEmpty else {
            return [createChunk(id: startingId, content: text, startOffset: startingOffset)]
        }

        var chunks: [TextChunk] = []
        var currentContent = ""
        var currentStartOffset = startingOffset
        var chunkId = startingId
        var overlapContent = ""

        for sentence in sentences {
            let sentenceTokens = estimateTokens(sentence)
            let currentTokens = estimateTokens(currentContent)

            let combinedTokens = currentTokens + sentenceTokens + 2 // ~2 tokens for space

            if combinedTokens > config.maxTokens && !currentContent.isEmpty {
                chunks.append(createChunk(
                    id: chunkId,
                    content: currentContent,
                    startOffset: currentStartOffset
                ))
                chunkId += 1

                overlapContent = extractOverlapFromEnd(currentContent)
                currentContent = overlapContent.isEmpty ? sentence : overlapContent + " " + sentence
                currentStartOffset = startingOffset + text.distance(
                    from: text.startIndex,
                    to: text.range(of: sentence)?.lowerBound ?? text.startIndex
                )
            } else {
                if currentContent.isEmpty {
                    currentContent = overlapContent.isEmpty ? sentence : overlapContent + " " + sentence
                    overlapContent = ""
                } else {
                    currentContent += " " + sentence
                }
            }
        }

        if !currentContent.isEmpty {
            chunks.append(createChunk(
                id: chunkId,
                content: currentContent,
                startOffset: currentStartOffset
            ))
        }

        return chunks
    }

    private func extractSentences(_ text: String) -> [String] {
        var sentences: [String] = []

        sentenceTokenizer.string = text
        sentenceTokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let sentence = String(text[range]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !sentence.isEmpty {
                sentences.append(sentence)
            }
            return true
        }

        return sentences
    }

    // MARK: - Token Limit Enforcement

    private func enforceTokenLimits(_ chunks: [TextChunk]) -> [TextChunk] {
        var result: [TextChunk] = []
        var chunkId = 0

        for chunk in chunks {
            if chunk.estimatedTokens <= config.maxTokens {
                result.append(TextChunk(
                    id: chunkId,
                    content: chunk.content,
                    estimatedTokens: chunk.estimatedTokens,
                    startOffset: chunk.startOffset,
                    endOffset: chunk.endOffset,
                    previousSummary: chunk.previousSummary
                ))
                chunkId += 1
            } else {
                // Hard split by character count
                let hardSplitChunks = hardSplitChunk(chunk, startingId: chunkId)
                result.append(contentsOf: hardSplitChunks)
                chunkId += hardSplitChunks.count
            }
        }

        return result
    }

    private func hardSplitChunk(_ chunk: TextChunk, startingId: Int) -> [TextChunk] {
        let maxChars = Int(Double(config.maxTokens) * ChunkConfig.charsPerToken)
        let overlapChars = Int(Double(config.overlapTokens) * ChunkConfig.charsPerToken)

        var chunks: [TextChunk] = []
        var currentIndex = chunk.content.startIndex
        var chunkId = startingId

        while currentIndex < chunk.content.endIndex {
            let remainingDistance = chunk.content.distance(from: currentIndex, to: chunk.content.endIndex)
            let chunkLength = min(maxChars, remainingDistance)

            let endIndex = chunk.content.index(currentIndex, offsetBy: chunkLength)
            let content = String(chunk.content[currentIndex..<endIndex])

            let offset = chunk.startOffset + chunk.content.distance(from: chunk.content.startIndex, to: currentIndex)
            chunks.append(createChunk(id: chunkId, content: content, startOffset: offset))
            chunkId += 1

            // Move forward with overlap
            let advanceAmount = max(1, chunkLength - overlapChars)
            currentIndex = chunk.content.index(currentIndex, offsetBy: advanceAmount, limitedBy: chunk.content.endIndex) ?? chunk.content.endIndex
        }

        return chunks
    }

    // MARK: - Chunk Merging

    private func mergeTinyChunks(_ chunks: [TextChunk]) -> [TextChunk] {
        guard chunks.count > 1 else { return chunks }

        var result: [TextChunk] = []
        var pending: TextChunk? = nil
        var chunkId = 0

        for chunk in chunks {
            if let p = pending {
                let combinedTokens = p.estimatedTokens + chunk.estimatedTokens

                if combinedTokens <= config.maxTokens {
                    // Merge
                    pending = TextChunk(
                        id: chunkId,
                        content: p.content + "\n\n" + chunk.content,
                        estimatedTokens: combinedTokens,
                        startOffset: p.startOffset,
                        endOffset: chunk.endOffset,
                        previousSummary: p.previousSummary
                    )
                } else {
                    // Save pending and start new
                    result.append(TextChunk(
                        id: chunkId,
                        content: p.content,
                        estimatedTokens: p.estimatedTokens,
                        startOffset: p.startOffset,
                        endOffset: p.endOffset,
                        previousSummary: p.previousSummary
                    ))
                    chunkId += 1

                    if chunk.estimatedTokens < config.minChunkTokens {
                        pending = chunk
                    } else {
                        result.append(TextChunk(
                            id: chunkId,
                            content: chunk.content,
                            estimatedTokens: chunk.estimatedTokens,
                            startOffset: chunk.startOffset,
                            endOffset: chunk.endOffset,
                            previousSummary: chunk.previousSummary
                        ))
                        chunkId += 1
                        pending = nil
                    }
                }
            } else if chunk.estimatedTokens < config.minChunkTokens {
                pending = chunk
            } else {
                result.append(TextChunk(
                    id: chunkId,
                    content: chunk.content,
                    estimatedTokens: chunk.estimatedTokens,
                    startOffset: chunk.startOffset,
                    endOffset: chunk.endOffset,
                    previousSummary: chunk.previousSummary
                ))
                chunkId += 1
            }
        }

        // Don't forget pending chunk
        if let p = pending {
            result.append(TextChunk(
                id: chunkId,
                content: p.content,
                estimatedTokens: p.estimatedTokens,
                startOffset: p.startOffset,
                endOffset: p.endOffset,
                previousSummary: p.previousSummary
            ))
        }

        return result
    }

    // MARK: - Helpers

    private func createChunk(id: Int, content: String, startOffset: Int) -> TextChunk {
        TextChunk(
            id: id,
            content: content,
            estimatedTokens: estimateTokens(content),
            startOffset: startOffset,
            endOffset: startOffset + content.count,
            previousSummary: nil
        )
    }

    private func extractOverlapFromEnd(_ text: String) -> String {
        let overlapChars = Int(Double(config.overlapTokens) * ChunkConfig.charsPerToken)

        guard text.count > overlapChars else { return "" }

        let startIndex = text.index(text.endIndex, offsetBy: -overlapChars)
        var overlap = String(text[startIndex...])

        // Try to start at a sentence boundary
        if let lastPeriod = overlap.lastIndex(of: ".") {
            let afterPeriod = overlap.index(after: lastPeriod)
            if afterPeriod < overlap.endIndex {
                overlap = String(overlap[afterPeriod...]).trimmingCharacters(in: .whitespaces)
            }
        }

        return overlap
    }
}
