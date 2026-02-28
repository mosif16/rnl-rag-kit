import Testing
@testable import RAGKit

@Suite("RAG Chunker")
struct RAGChunkerTests {

    @Test("Empty input produces no chunks")
    func emptyInput() {
        let chunks = RAGChunker.chunk(text: "")
        #expect(chunks.isEmpty)
    }

    @Test("Chunks cover entire text with no gaps")
    func chunksCoverText() {
        let para = String(repeating: "Lorem ipsum dolor sit amet. ", count: 80)
        let text = [para, para, para].joined(separator: "\n\n")
        let chunks = RAGChunker.chunk(text: text, maxChars: 900, overlapRatio: 0.15)
        #expect(!chunks.isEmpty)
        #expect(chunks.first?.startOffset == 0)
        #expect(chunks.last?.endOffset == text.count)
        #expect(chunks.allSatisfy { !$0.text.isEmpty })
        for i in 1..<chunks.count {
            #expect(chunks[i].startOffset <= chunks[i-1].endOffset)
        }
    }

    @Test("Long single line produces overlap")
    func overlapForFallback() {
        let text = String(repeating: "a", count: 4000)
        let chunks = RAGChunker.chunk(text: text, maxChars: 500, overlapRatio: 0.2)
        #expect(!chunks.isEmpty)
        #expect(chunks.allSatisfy { $0.text.count <= 500 })
        var hasOverlap = false
        for i in 1..<chunks.count {
            if chunks[i].startOffset < chunks[i-1].endOffset { hasOverlap = true; break }
        }
        #expect(hasOverlap)
    }

    @Test("Short text produces single chunk")
    func shortTextSingleChunk() {
        let text = "Short text here."
        let chunks = RAGChunker.chunk(text: text, maxChars: 1000)
        #expect(chunks.count == 1)
        #expect(chunks.first?.text == text)
    }

    @Test("Whitespace-only text produces no chunks")
    func whitespaceOnlyNoChunks() {
        let chunks = RAGChunker.chunk(text: "   \n\n\t  ")
        #expect(chunks.isEmpty)
    }

    @Test("All chunks have valid offsets")
    func validOffsets() {
        let text = String(repeating: "Hello world. ", count: 100)
        let chunks = RAGChunker.chunk(text: text, maxChars: 200)
        for chunk in chunks {
            #expect(chunk.startOffset >= 0)
            #expect(chunk.endOffset <= text.count)
            #expect(chunk.endOffset > chunk.startOffset)
        }
    }
}
