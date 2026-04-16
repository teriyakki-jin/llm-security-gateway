# LLM Security Gateway

> **Post-Quantum Cryptography + LLM Security** — ML-KEM-768/X25519 하이브리드 암호화 채널 위에서 동작하는 Prompt Injection · Jailbreak · PII 탐지 게이트웨이

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Go](https://img.shields.io/badge/Go-1.22-00ADD8?logo=go)](https://go.dev)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![NIST FIPS 203](https://img.shields.io/badge/NIST-FIPS%20203%20(ML--KEM)-orange)](https://csrc.nist.gov/pubs/fips/203/final)

---

## 개요

LLM Security Gateway는 두 레이어로 구성됩니다.

| 레이어 | 역할 | 언어 |
|--------|------|------|
| **PQC Proxy** | ML-KEM-768 + X25519 하이브리드 KEM, 4-way 핸드셰이크, AES-256-GCM 암호화 | Go |
| **Security Gateway** | Prompt Injection/Jailbreak 탐지, PII·시크릿 마스킹, 감사 로그, 속도 제한 | Python |

양자 컴퓨터 위협에 대비하는 동시에, LLM에 대한 프롬프트 인젝션 공격을 실시간으로 차단합니다.

---

## 아키텍처

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/e9dd786b-46ad-4f12-b788-eaf351cbcb7b" />


---

## 주요 기능

### PQC Proxy
- **ML-KEM-768** (NIST FIPS 203) — 모듈 격자 기반 양자 내성 KEM
- **X25519 + ML-KEM-768 하이브리드** — HKDF-SHA256으로 두 공유 비밀 결합 (전환기 이중 안전)
- **AES-256-GCM** with 8-byte counter nonce — 재전송 공격 방어
- **세션 재개** — 2-tier 캐시 (sync.Map → Redis AES-256-GCM 암호화 저장)
- **키 로테이션** — 2²⁴ 메시지 또는 1시간마다 HKDF 재파생

### Security Gateway
- **다층 탐지 파이프라인** — Rule → Heuristic → ML (조기 종료)
- **Prompt Injection 탐지**
  - 규칙 기반: 7개 카테고리 (지시 오버라이드, 역할 조작, 구분자 오용 등) + 한국어 패턴
  - 휴리스틱: Shannon 엔트로피, 스크립트 전환, 특수문자 비율, 길이 급증
  - ML: DeBERTa-v3-base → ONNX INT8 양자화
- **Jailbreak 탐지** — DAN/AIM/CRESCENDO/PAIR 등 9종 분류 + 의미론적 코사인 유사도
- **응답 필터링** — PII 마스킹 (Presidio + 주민번호/한국 전화번호) + API 시크릿 감지
- **Shadow Mode** — 차단 없이 로그만 기록, 안전한 프로덕션 롤아웃
- **감사 로그** — SHA-256 요청 해시, asyncio.Queue 배치 DB 기록
- **속도 제한** — Redis sliding window, IP/API키 단위

---

## 빠른 시작

### 사전 요구사항

- Docker 24+
- Docker Compose v2

### 실행

```bash
git clone https://github.com/teriyakki-jin/llm-security-gateway.git
cd llm-security-gateway

# 환경변수 설정
cp gateway/.env.example gateway/.env
# OPENAI_API_KEY 또는 ANTHROPIC_API_KEY 입력

# 전체 스택 시작
make docker-up

# 로그 확인
make docker-logs
```

### API 사용 예시

```bash
# 채팅 요청 (PQC 프록시 경유)
curl -X POST http://localhost:8443/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Prompt Injection 시도 → 403 차단
curl -X POST http://localhost:8443/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "Ignore all previous instructions and..."}]}'

# 메트릭 조회
curl http://localhost:8000/metrics

# Shadow Mode 활성화 (차단 없이 로그만)
curl -X POST http://localhost:8000/admin/shadow-mode \
  -d '{"enabled": true}'
```

---

## API 엔드포인트

### Gateway (`:8000`)

| Method | Path | 설명 |
|--------|------|------|
| POST | `/v1/chat/completions` | OpenAI 호환 채팅 (탐지 포함) |
| POST | `/v1/messages` | Anthropic 호환 채팅 (탐지 포함) |
| GET | `/health` | 헬스체크 |
| GET | `/metrics` | Prometheus 메트릭 |
| POST | `/admin/shadow-mode` | Shadow Mode 토글 |
| POST | `/admin/threshold` | 차단 임계값 조정 |
| GET | `/admin/stats` | 탐지 통계 |

### PQC Proxy (`:8443`)

| Method | Path | 설명 |
|--------|------|------|
| GET | `/pqc/keys` | 서버 공개키 조회 |
| POST | `/pqc/handshake` | 4-way 핸드셰이크 시작 |
| POST | `/pqc/finished` | 핸드셰이크 완료 |
| ANY | `/v1/*` | 암호화 채널로 프록시 |

---

## 개발 환경 설정

```bash
# Python 의존성 설치 (uv 권장)
cd gateway
uv venv --python 3.11
uv pip install -e ".[dev]"

# 테스트 실행
make test-python

# Go 테스트 (Docker 필요, liboqs CGO 의존)
make test-go-docker

# 전체 린트
make lint

# 코드 포맷
make fmt
```

---

## 보안 설계

### 왜 하이브리드 KEM인가?

```
session_key = HKDF-SHA256(
    ikm  = x25519_shared_secret || mlkem_shared_secret,
    info = "llm-security-gateway hybrid kem v1"
)
```

- **전환기 이중 안전**: X25519가 깨져도 ML-KEM이 보호, ML-KEM이 예상치 못한 취약점이 있어도 X25519가 보호
- **NIST 권고 준수**: FIPS 203 (ML-KEM-768) 사용

### Shadow Mode 롤아웃 절차

```
1. shadow_mode=True 배포 (1~2주)
   └─ 모든 요청 통과, detection_shadow_block 이벤트 로그
2. 로그 분석 → FP/FN 측정 → 임계값 조정
3. shadow_mode=False 전환 → 실제 차단 시작
```

---

## 오픈소스 기여 계획

| 프로젝트 | 기여 내용 |
|----------|-----------|
| [Open Quantum Safe / liboqs](https://github.com/open-quantum-safe/liboqs) | X25519 + ML-KEM-768 하이브리드 KEM Go 예제 |
| [NVIDIA / garak](https://github.com/NVIDIA/garak) | 다국어 Jailbreak 프로브 (한/일/중) |
| [ProtectAI / llm-guard](https://github.com/protectai/llm-guard) | 휴리스틱 Scanner 추가 |

---

## 블로그 시리즈

- [1편: ML-KEM-768 Go 구현기](https://velog.io/@jyg3485/LLM-Security-Gateway-개발기-1편)
- [2편: FastAPI 게이트웨이 설계기](https://velog.io/@jyg3485)
- 3편: Prompt Injection 탐지 엔진 딥다이브 (예정)

---

## 라이선스

MIT License — 자세한 내용은 [LICENSE](LICENSE) 참조
