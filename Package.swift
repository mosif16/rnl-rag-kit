// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "RAGKit",
    platforms: [
        .iOS(.v16),
        .macOS(.v13)
    ],
    products: [
        .library(
            name: "RAGKit",
            targets: ["RAGKit"]
        )
    ],
    targets: [
        .target(
            name: "RAGKit"
        ),
        .testTarget(
            name: "RAGKitTests",
            dependencies: ["RAGKit"]
        )
    ]
)
