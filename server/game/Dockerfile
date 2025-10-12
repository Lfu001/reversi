# syntax=docker/dockerfile:1

# Use the official Rust image as a parent image
FROM rust:1.89.0-trixie AS builder

# Set the working directory
WORKDIR /usr/src/reversi/server

# Install build dependencies
RUN apt-get update && \
    apt-get install -y cmake pkg-config libclang-dev libssl-dev

# Copy the workspace
COPY . .

# Build the application
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/src/reversi/server/game/target \
    cargo build --release -p game

# Use a smaller image for the final container
FROM debian:trixie-20250811-slim

# Install CA certificates for HTTPS connections
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Copy the binary from the builder stage
COPY --from=builder /usr/src/reversi/server/target/release/game /usr/local/bin/game

# Set environment variables with defaults
ENV HOST=0.0.0.0
ENV PORT=8080

# Expose the port the app runs on
EXPOSE 8080

# Run the application
CMD ["game"]
