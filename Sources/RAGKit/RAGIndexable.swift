import Foundation

/// Common interface for any content that can be indexed by RAGManager.
public protocol RAGIndexable {
    var ragDocumentID: String { get }
    var ragTitle: String { get }
    var ragSourceType: RAGSourceType { get }
    var ragSourcePath: String? { get }
    func ragTextContent() -> String
}
