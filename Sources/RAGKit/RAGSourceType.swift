import Foundation

/// Canonical source types for RAG-indexable content.
public enum RAGSourceType: String, Codable, CaseIterable, Sendable {
    case transcript = "transcript"
    case document = "document"
    case aiNote = "ai_note"
    case flashcard = "flashcard"
    case quiz = "quiz"
    case slideDeck = "slide_deck"
}
