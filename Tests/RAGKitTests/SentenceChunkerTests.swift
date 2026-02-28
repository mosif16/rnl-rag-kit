import Foundation
import Testing
@testable import RAGKit

@Suite("SentenceChunker")
struct SentenceChunkerTests {

    @Test("Empty text produces no chunks")
    func emptyText() {
        let chunker = SentenceChunker(config: .standard)
        let chunks = chunker.chunk("")
        #expect(chunks.isEmpty)
    }

    @Test("Whitespace-only text produces no chunks")
    func whitespaceOnly() {
        let chunker = SentenceChunker(config: .standard)
        let chunks = chunker.chunk("   \n\n   ")
        #expect(chunks.isEmpty)
    }

    @Test("Short text produces single chunk")
    func shortTextSingleChunk() {
        let chunker = SentenceChunker(config: .standard)
        let chunks = chunker.chunk(SampleTranscripts.short)
        #expect(chunks.count == 1)
        #expect(chunks[0].id == 0)
    }

    @Test("Medium text produces multiple chunks")
    func mediumTextMultipleChunks() {
        let chunker = SentenceChunker(config: .standard)
        let chunks = chunker.chunk(SampleTranscripts.medium)
        #expect(chunks.count > 1)
    }

    @Test("Long text produces many chunks")
    func longTextManyChunks() {
        let chunker = SentenceChunker(config: .standard)
        let chunks = chunker.chunk(SampleTranscripts.long)
        #expect(chunks.count >= 2)
    }

    @Test("Chunk IDs are sequential starting from 0")
    func sequentialIds() {
        let chunker = SentenceChunker(config: .standard)
        let chunks = chunker.chunk(SampleTranscripts.long)
        for (index, chunk) in chunks.enumerated() {
            #expect(chunk.id == index)
        }
    }

    @Test("Each chunk has at least one sentence")
    func minimumSentences() {
        let chunker = SentenceChunker(config: .standard)
        let chunks = chunker.chunk(SampleTranscripts.long)
        for chunk in chunks {
            #expect(chunk.sentenceCount >= 1)
            #expect(!chunk.sentences.isEmpty)
        }
    }

    @Test("Chunk content matches joined sentences")
    func contentMatchesSentences() {
        let chunker = SentenceChunker(config: .standard)
        let chunks = chunker.chunk(SampleTranscripts.medium)
        for chunk in chunks {
            #expect(chunk.content == chunk.sentences.joined(separator: " "))
        }
    }

    @Test("First chunk has no context from previous")
    func firstChunkNoContext() {
        let chunker = SentenceChunker(config: .standard)
        let chunks = chunker.chunk(SampleTranscripts.long)
        #expect(chunks.first?.contextFromPrevious == nil)
    }

    @Test("Subsequent chunks have context from previous")
    func subsequentChunksHaveContext() {
        let chunker = SentenceChunker(config: .standard)
        let text = SampleTranscripts.repeated(sentences: 30)
        let chunks = chunker.chunk(text)
        guard chunks.count > 1 else { return }
        for chunk in chunks.dropFirst() {
            #expect(chunk.contextFromPrevious != nil)
        }
    }

    @Test("Chunk content respects maxContentChars budget")
    func respectsCharBudget() {
        let config = SentenceChunkConfig(maxContentChars: 500, maxChunks: 100)
        let chunker = SentenceChunker(config: config)
        let chunks = chunker.chunk(SampleTranscripts.repeated(sentences: 50))
        for chunk in chunks {
            // Allow some tolerance for single long sentences
            #expect(chunk.content.count <= 600, "Chunk content \(chunk.content.count) chars exceeds budget")
        }
    }

    @Test("Dense config produces more chunks than standard")
    func denseProducesMoreChunks() {
        let standard = SentenceChunker(config: .standard)
        let dense = SentenceChunker(config: .dense)
        let text = SampleTranscripts.repeated(sentences: 30)
        let standardChunks = standard.chunk(text)
        let denseChunks = dense.chunk(text)
        #expect(denseChunks.count >= standardChunks.count)
    }

    @Test("Safety valve caps chunk count at maxChunks")
    func maxChunksCap() {
        let config = SentenceChunkConfig(maxContentChars: 100, maxChunks: 5)
        let chunker = SentenceChunker(config: config)
        let text = SampleTranscripts.repeated(sentences: 100)
        let chunks = chunker.chunk(text)
        #expect(chunks.count <= 5)
    }

    @Test("Estimated token count is roughly chars/4")
    func estimatedTokens() {
        let chunker = SentenceChunker(config: .standard)
        let chunks = chunker.chunk(SampleTranscripts.short)
        guard let chunk = chunks.first else { return }
        let expected = Int(ceil(Double(chunk.content.count) / 4.0))
        #expect(chunk.estimatedTokens == expected)
    }

    @Test("Handles text with ellipsis artifacts")
    func handlesEllipsis() {
        let chunker = SentenceChunker(config: .standard)
        let text = "First sentence... Second sentence. Third sentence... Fourth sentence."
        let chunks = chunker.chunk(text)
        #expect(!chunks.isEmpty)
    }

    @Test("Handles text with mixed line endings")
    func handlesLineEndings() {
        let chunker = SentenceChunker(config: .standard)
        let text = "First sentence.\r\nSecond sentence.\rThird sentence.\nFourth sentence. Fifth sentence here. Sixth important sentence."
        let chunks = chunker.chunk(text)
        #expect(!chunks.isEmpty)
    }

    @Test("Start and end sentence indices are valid")
    func validSentenceIndices() {
        let chunker = SentenceChunker(config: .standard)
        let chunks = chunker.chunk(SampleTranscripts.long)
        for chunk in chunks {
            #expect(chunk.startSentenceIndex >= 0)
            #expect(chunk.endSentenceIndex >= chunk.startSentenceIndex)
            #expect(chunk.endSentenceIndex == chunk.startSentenceIndex + chunk.sentenceCount - 1)
        }
    }

    @Test("Very long single sentence is handled without crash")
    func veryLongSentence() {
        let chunker = SentenceChunker(config: .standard)
        let longSentence = String(repeating: "word ", count: 500)
        let chunks = chunker.chunk(longSentence)
        #expect(!chunks.isEmpty)
    }
}
