# Stage 1: Build liboqs C library
FROM ubuntu:24.04 AS liboqs-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake ninja-build gcc g++ libssl-dev git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch main https://github.com/open-quantum-safe/liboqs.git /liboqs
RUN cmake -S /liboqs -B /liboqs/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DOQS_ALGS_ENABLED=ML_KEM \
      -DBUILD_SHARED_LIBS=OFF \
      -GNinja \
    && cmake --build /liboqs/build \
    && cmake --install /liboqs/build

# Stage 2: Build Go proxy
FROM golang:1.22-bookworm AS go-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy liboqs artifacts from stage 1.
COPY --from=liboqs-builder /usr/local/include/oqs /usr/local/include/oqs
COPY --from=liboqs-builder /usr/local/lib/liboqs.a /usr/local/lib/liboqs.a

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download

COPY . .

ENV CGO_ENABLED=1
ENV CGO_CFLAGS="-I/usr/local/include"
ENV CGO_LDFLAGS="-L/usr/local/lib -loqs -lssl -lcrypto"

RUN go build -o /bin/pqc-proxy ./cmd/proxy

# Stage 3: Minimal runtime image
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=liboqs-builder /usr/local/lib/liboqs.a /usr/local/lib/liboqs.a
COPY --from=go-builder /bin/pqc-proxy /usr/local/bin/pqc-proxy

RUN useradd -r -u 1001 -s /sbin/nologin pqcproxy
USER pqcproxy

EXPOSE 8443

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8443/health || exit 1

ENTRYPOINT ["/usr/local/bin/pqc-proxy"]
