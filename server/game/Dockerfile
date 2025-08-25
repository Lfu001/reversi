# syntax=docker/dockerfile:1

# Use the official Rust image as a parent image
FROM rust:1.89.0-trixie as builder

# Set the working directory
WORKDIR /usr/src/reversi-game

# Install build dependencies
RUN apt-get update && \
    apt-get install -y cmake pkg-config libclang-dev libssl-dev

# Copy the manifests
COPY Cargo.toml Cargo.lock ./

# Create a dummy source file to build dependencies
RUN mkdir -p src && \
    echo "fn main() {}" > src/main.rs

# Build the dependencies (this step is cached as long as Cargo.toml and Cargo.lock don't change)
RUN cargo build --release

# Now copy the actual source code
COPY src/ src/

# Build the application
RUN cargo install --path . --root /usr/local

# Use a smaller image for the final container
FROM debian:trixie-20250811-slim

# Install CA certificates for HTTPS connections
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Copy the binary from the builder stage
COPY --from=builder /usr/local/bin/game /usr/local/bin/game

# Set environment variables with defaults
ENV HOST=0.0.0.0
ENV PORT=8080

# Expose the port the app runs on
EXPOSE 8080

# Run the application
CMD ["game"]
