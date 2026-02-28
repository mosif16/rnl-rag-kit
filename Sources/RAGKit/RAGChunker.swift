import Foundation
import NaturalLanguage

/// Sentence/paragraph-aware chunker producing overlapping chunks with byte offsets.
/// Keeps things deterministic for stable citations.
public struct RAGChunker {
    public struct Chunk: Hashable {
        public let index: Int
        public let range: Range<String.Index>
        public let startOffset: Int
        public let endOffset: Int
        public let text: String
        public let wordCount: Int

        public init(
            index: Int,
            range: Range<String.Index>,
            startOffset: Int,
            endOffset: Int,
            text: String,
            wordCount: Int
        ) {
            self.index = index
            self.range = range
            self.startOffset = startOffset
            self.endOffset = endOffset
            self.text = text
            self.wordCount = wordCount
        }
    }

    /// Splits text into chunks of ~maxChars with an overlap ratio (0.0–0.5 typical).
    /// Prefers sentence boundaries for semantic coherence; falls back to naive slicing when needed.
    public static func chunk(text: String, maxChars: Int = 900, overlapRatio: Double = 0.15) -> [Chunk] {
        guard !text.isEmpty, maxChars > 0,
              text.contains(where: { $0.isLetter || $0.isNumber }) else { return [] }
        let overlapRatioClamped = max(0.0, min(0.5, overlapRatio))
        let overlapChars = max(0, Int(Double(maxChars) * overlapRatioClamped))

        func makeChunk(index: Int, range: Range<String.Index>) -> Chunk? {
            guard range.lowerBound < range.upperBound else { return nil }
            let slice = text[range]
            let startOffset = text.distance(from: text.startIndex, to: range.lowerBound)
            let endOffset = text.distance(from: text.startIndex, to: range.upperBound)
            let textSlice = String(slice)
            let words = textSlice.split { !$0.isLetter && !$0.isNumber }.count
            return Chunk(
                index: index,
                range: range,
                startOffset: startOffset,
                endOffset: endOffset,
                text: textSlice,
                wordCount: words
            )
        }

        // Sentence-first chunking for better embeddings.
        let sentenceRanges: [Range<String.Index>] = {
            let tokenizer = NLTokenizer(unit: .sentence)
            tokenizer.string = text
            var out: [Range<String.Index>] = []
            tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
                out.append(range)
                return true
            }
            return out
        }()

        var chunks: [Chunk] = []
        chunks.reserveCapacity(max(1, text.count / maxChars))

        if !sentenceRanges.isEmpty {
            var idx = 0
            var startSentenceIndex = 0

            func clampStartSentenceIndex(afterChunkFrom start: Int, to end: Int) -> Int {
                guard overlapChars > 0 else { return end + 1 }

                let chunkStart = sentenceRanges[start].lowerBound
                let chunkEnd = sentenceRanges[end].upperBound
                let chunkLen = text.distance(from: chunkStart, to: chunkEnd)
                let back = min(overlapChars, max(0, chunkLen))
                let desiredStart = text.index(chunkEnd, offsetBy: -back, limitedBy: chunkStart) ?? chunkStart

                var newStart = end
                while newStart > start, sentenceRanges[newStart].lowerBound > desiredStart {
                    newStart -= 1
                }

                // Ensure progress (avoid infinite loops when overlap >= chunk length).
                return max(start + 1, newStart)
            }

            while startSentenceIndex < sentenceRanges.count {
                var endSentenceIndex = startSentenceIndex
                var bestEndIndex = startSentenceIndex

                while endSentenceIndex < sentenceRanges.count {
                    let startIdx = sentenceRanges[startSentenceIndex].lowerBound
                    let endIdx = sentenceRanges[endSentenceIndex].upperBound
                    let proposedLen = text.distance(from: startIdx, to: endIdx)

                    if proposedLen <= maxChars || endSentenceIndex == startSentenceIndex {
                        bestEndIndex = endSentenceIndex
                        endSentenceIndex += 1
                        continue
                    }

                    break
                }

                let range = sentenceRanges[startSentenceIndex].lowerBound..<sentenceRanges[bestEndIndex].upperBound
                // If a single sentence is longer than maxChars, split it naively.
                if bestEndIndex == startSentenceIndex, text.distance(from: range.lowerBound, to: range.upperBound) > maxChars {
                    var cursor = range.lowerBound
                    while cursor < range.upperBound {
                        let sliceEnd = text.index(cursor, offsetBy: maxChars, limitedBy: range.upperBound) ?? range.upperBound
                        if let c = makeChunk(index: idx, range: cursor..<sliceEnd) {
                            chunks.append(c)
                            idx += 1
                        }
                        if sliceEnd >= range.upperBound { break }
                        cursor = overlapChars > 0 ? (text.index(sliceEnd, offsetBy: -overlapChars, limitedBy: cursor) ?? sliceEnd) : sliceEnd
                    }
                    startSentenceIndex += 1
                    continue
                }

                if let chunk = makeChunk(index: idx, range: range) {
                    chunks.append(chunk)
                    idx += 1
                }

                startSentenceIndex = clampStartSentenceIndex(afterChunkFrom: startSentenceIndex, to: bestEndIndex)
                if chunks.count > 400 { break }
            }
        }

        // Fallback: naive slicing if sentence segmentation fails or produces pathological output.
        if chunks.isEmpty || chunks.count > 400 {
            chunks.removeAll(keepingCapacity: true)
            var idx = 0
            var i = text.startIndex
            while i < text.endIndex {
                let end = text.index(i, offsetBy: maxChars, limitedBy: text.endIndex) ?? text.endIndex
                if let chunk = makeChunk(index: idx, range: i..<end) {
                    chunks.append(chunk)
                    idx += 1
                }
                if end >= text.endIndex { break }
                i = overlapChars > 0 ? (text.index(end, offsetBy: -overlapChars, limitedBy: i) ?? end) : end
                if chunks.count > 400 { break }
            }
        }

        // Ensure stable indices even when we appended extra splits.
        if !chunks.isEmpty, chunks.last?.endOffset != text.count {
            if let last = chunks.last, last.endOffset < text.count {
                let tailStart = last.range.upperBound
                let tailEnd = text.endIndex
                if let tail = makeChunk(index: chunks.count, range: tailStart..<tailEnd) {
                    chunks.append(tail)
                }
            }
        }

        return chunks
    }
}
