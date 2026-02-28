import Foundation
import Testing
@testable import RAGKit

@Suite("SentenceChunker Config")
struct SentenceChunkerConfigTests {

    @Test("Standard config has correct values")
    func standardConfig() {
        let config = SentenceChunkConfig.standard
        #expect(config.sentencesPerChunk == 6)
        #expect(config.minSentences == 2)
        #expect(config.overlapSentences == 1)
        #expect(config.flashcardsPerMicroChunk == 3)
        #expect(config.quizQuestionsPerMicroChunk == 1)
        #expect(config.maxContentChars == 1_000)
        #expect(config.maxChunks == 80)
    }

    @Test("Dense config has smaller chunks")
    func denseConfig() {
        let config = SentenceChunkConfig.dense
        #expect(config.sentencesPerChunk == 4)
        #expect(config.maxContentChars == 700)
        #expect(config.maxChunks == 100)
    }

    @Test("Balanced config has larger chunks")
    func balancedConfig() {
        let config = SentenceChunkConfig.balanced
        #expect(config.sentencesPerChunk == 8)
        #expect(config.maxContentChars == 1_200)
    }

    @Test("HighYield config has largest chunks")
    func highYieldConfig() {
        let config = SentenceChunkConfig.highYield
        #expect(config.sentencesPerChunk == 10)
        #expect(config.maxContentChars == 1_500)
    }

    @Test("estimateChunkCount returns 1 for short text")
    func estimateChunkCountShort() {
        let chunker = SentenceChunker(config: .standard)
        let count = chunker.estimateChunkCount(SampleTranscripts.short)
        #expect(count == 1)
    }

    @Test("estimateChunkCount returns more for longer text")
    func estimateChunkCountLong() {
        let chunker = SentenceChunker(config: .standard)
        let shortCount = chunker.estimateChunkCount(SampleTranscripts.short)
        let longCount = chunker.estimateChunkCount(SampleTranscripts.long)
        #expect(longCount >= shortCount)
    }

    @Test("estimateYield scales with chunk count")
    func estimateYieldScales() {
        let chunker = SentenceChunker(config: .standard)
        let yield = chunker.estimateYield(SampleTranscripts.long)
        let chunkCount = chunker.estimateChunkCount(SampleTranscripts.long)
        #expect(yield.flashcards == chunkCount * 3) // standard has 3 per chunk
        #expect(yield.quizQuestions == chunkCount * 1) // standard has 1 per chunk
    }

    @Test("Custom config with custom values")
    func customConfig() {
        let config = SentenceChunkConfig(
            sentencesPerChunk: 12,
            minSentences: 4,
            overlapSentences: 2,
            flashcardsPerMicroChunk: 5,
            quizQuestionsPerMicroChunk: 2,
            maxContentChars: 2_000,
            maxChunks: 40
        )
        #expect(config.sentencesPerChunk == 12)
        #expect(config.maxChunks == 40)
    }
}
