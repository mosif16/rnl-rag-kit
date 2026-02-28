import CryptoKit
import Foundation

public enum RAGKitConfiguration {
    /// Global switch for RAG indexing and retrieval paths.
    public static var ragEnabled: Bool = true

    /// Enables verbose diagnostics for manager/store/context operations.
    public static var debugLoggingEnabled: Bool = false

    /// Enables verbose diagnostics for embedding model operations.
    public static var embeddingDebugLoggingEnabled: Bool = false

    /// Enables Cloud sync callback execution when provided.
    public static var cloudSyncEnabled: Bool = false

    /// Enables Cloud sync error logging.
    public static var cloudSyncDebugLoggingEnabled: Bool = false
}

enum FeatureFlags {
    static var RAGEnabled: Bool { RAGKitConfiguration.ragEnabled }
    static var RAGDebugLogging: Bool { RAGKitConfiguration.debugLoggingEnabled }
    static var AppleEmbeddingDebugLogging: Bool { RAGKitConfiguration.embeddingDebugLoggingEnabled }
    static var RAGCloudKitSyncEnabled: Bool { RAGKitConfiguration.cloudSyncEnabled }
    static var RAGCloudKitDebugLogging: Bool { RAGKitConfiguration.cloudSyncDebugLoggingEnabled }
}

enum DiagnosticsLogger {
    static func log(_ message: @autoclosure () -> String) {
        guard FeatureFlags.RAGDebugLogging || FeatureFlags.AppleEmbeddingDebugLogging || FeatureFlags.RAGCloudKitDebugLogging else {
            return
        }

#if DEBUG
        print("[RAGKit] \(message())")
#else
        NSLog("[RAGKit] %@", message())
#endif
    }
}

enum DeduplicationService {
    static func hash(_ content: String) -> String {
        let data = Data(content.utf8)
        let digest = SHA256.hash(data: data)
        return digest.prefix(16).map { String(format: "%02x", $0) }.joined()
    }
}
