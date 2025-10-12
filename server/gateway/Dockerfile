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

# Build the application using mount cache
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/src/reversi/server/gateway/target \
    cargo build --release -p gateway

# Use a smaller image for the final container
FROM debian:trixie-20250811-slim

# Install OpenSSL and other required libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends openssl ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Copy the binary from the builder stage
COPY --from=builder /usr/src/reversi/server/target/release/gateway /usr/local/bin/gateway

# Set environment variables with defaults
ENV HOST=0.0.0.0
ENV PORT=8081
ENV REDIS_URL=redis://redis:6379
ENV NEW_GAME_API_URL=http://game:8080/new
ENV STEP_GAME_API_URL=http://game:8080/step
ENV SUGGESTION_API_URL=http://proxy:80/models

# Expose the port the app runs on
EXPOSE 8081

# Run the application
CMD ["gateway"]
