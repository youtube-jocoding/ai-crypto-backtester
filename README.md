# AI Crypto Backtester

AI 기반의 암호화폐 트레이딩 전략을 백테스팅할 수 있는 Python 도구입니다. Binance API로 과거 데이터를 수집하고, OpenAI 모델 등 다양한 AI 엔진을 활용해 매매 의사결정을 시뮬레이션합니다. 결과는 인터랙티브한 HTML 리포트로 저장되어 손익 그래프와 주요 지표를 쉽게 확인할 수 있습니다.

## 주요 특징

- **다중 타임프레임 데이터 지원**: 1분봉부터 4시간봉까지 원하는 범위의 데이터를 한 번에 불러와 AI 입력으로 사용할 수 있습니다.
- **AI 의사결정 엔진**: OpenAI API를 사용해 매수/매도 방향과 포지션 규모, 손절·익절 기준을 자동으로 생성합니다. 원한다면 무작위(random) 결정 방식도 선택 가능합니다.
- **SQLite 기반 데이터 캐싱**: 수집한 시장 데이터를 `data/` 폴더의 데이터베이스에 저장하여 동일 구간 백테스트 시 재다운로드 시간을 절약합니다.
- **결과 시각화**: Plotly를 활용해 트레이딩 로그와 성과 지표를 그래프로 확인할 수 있으며, `results/` 폴더에 HTML 파일 형태로 저장됩니다.

## 요구 사항

- Python 3.10 이상
- pip가 설치된 환경
- OpenAI API 키(또는 다른 AI provider 키)

## 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/youtube-jocoding/ai-crypto-backtester.git
cd ai-crypto-backtester
```

2. (선택) 가상 환경 생성
```bash
python -m venv .venv
source .venv/bin/activate
```

3. 의존성 설치
```bash
pip install -r requirements.txt
```

4. 환경 변수 설정
프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 필요한 API 키 (예: `OPENAI_API_KEY`) 를 설정하세요.
```
OPENAI_API_KEY="your_openai_api_key"
```

## 사용 방법

1. **설정 파일 수정 (`config.json`)**
프로젝트 루트 디렉토리에 있는 `config.json` 파일을 열어 백테스팅 및 AI 관련 설정을 수정합니다.

```json
{
  "backtest_settings": {
    "symbol": "BTC/USDT",         // 트레이딩 페어
    "start_date": "2025-01-01",   // 백테스팅 시작 날짜 (YYYY-MM-DD)
    "end_date": "2025-01-15",     // 백테스팅 종료 날짜 (YYYY-MM-DD)
    "initial_capital": 10000,    // 초기 자본금
    "fee": 0.0005                // 거래 수수료 (0.0005 = 0.05%)
  },
  "ai_settings": {
    "provider": "openai",        // 의사결정 주체 ("openai" or "random")
    "model": "gpt-4o-mini",      // 사용할 OpenAI 모델
    "system_prompt_file": "prompts/system_prompt.txt",  // 시스템 프롬프트 파일 경로
    "reasoning_effort": "medium", // 추론 모델의 추론 강도 (low, medium, high)
    "input_data": {
      "timeframes": {          // AI 입력으로 사용할 데이터 설정
        "15m": {"limit": 96},  // 예: 15분봉 과거 96개 데이터 사용
        "1h": {"limit": 48},
        "4h": {"limit": 30}
      }
    }
  }
}
```

주요 설정 항목 설명

- `symbol`: 백테스트할 거래 페어를 지정합니다.
- `start_date`/`end_date`: 테스트할 기간을 YYYY-MM-DD 형식으로 입력합니다.
- `initial_capital`: 시작 자본금을 의미하며 결과 수익률 계산에 사용됩니다.
- `fee`: 거래 한 건당 적용될 수수료 비율입니다.
- `provider`: `openai` 또는 `random` 중 선택합니다.
- `model`: 사용할 AI 모델 이름을 지정합니다.
- `system_prompt_file`: AI에게 전달할 시스템 프롬프트가 저장된 파일 경로입니다.
- `reasoning_effort`: AI가 의사결정을 위해 사용할 추론 강도 옵션입니다.
- `input_data.timeframes`: AI 입력으로 활용할 각 시간대별 캔들 수(limit)를 정의합니다.

2. **시스템 프롬프트 관리**
AI 모델이 사용할 시스템 프롬프트를 별도 파일로 관리하는 것이 편리합니다. 
`prompts/system_prompt.txt` 파일을 생성하고 그 안에 시스템 프롬프트를 작성하세요:

```
프로젝트 루트/
└── prompts/
    └── system_prompt.txt  <-- 이 파일에 시스템 프롬프트 내용을 작성하세요
```

시스템 프롬프트 예시:
```
You are an expert cryptocurrency trading AI specializing in technical analysis and risk management. 
Your task is to analyze market data across multiple timeframes and make informed trading decisions.

You must respond in the following JSON format:
{
    "direction": "LONG" or "SHORT",
    "position_size": investment ratio against total capital (decimal between 0.1-1.0),
    "leverage": an integer between 1-20,
    "stop_loss": percentage distance from entry as decimal, e.g., 0.005 for 0.5%,
    "take_profit": percentage distance from entry as decimal, e.g., 0.005 for 0.5%
}

Return ONLY the raw JSON object.
```

프롬프트는 AI의 의사결정 품질에 큰 영향을 미칩니다. 전략에 맞게 내용을 수정하되, 지나치게 길어지면 API 사용량이 증가할 수 있으니 주의하세요. 여러 전략을 테스트하려면 프롬프트 파일을 각각 만들어 `config.json`의 `system_prompt_file` 경로만 변경하면 됩니다.

3. **백테스팅 실행**
```bash
python main.py
```
백테스트가 실행되고 결과는 `results/` 디렉토리에 HTML 파일로 저장됩니다.

실행이 끝나면 `results/backtest_*.html` 파일을 브라우저에서 열어 손익 그래프, 트레이드 로그, 각종 성과 지표를 확인할 수 있습니다. 동일한 설정으로 여러 번 테스트하면 파일명이 날짜와 모델 이름을 기준으로 자동 생성됩니다.

> **주의**: 본 저장소는 연구 및 학습 목적으로 제공됩니다. 실제 자금을 투자하기 전에 반드시 충분한 검증과 위험 관리가 필요합니다. 개발자는 본 도구 사용으로 발생하는 손실에 대해 책임지지 않습니다.

## 프로젝트 구조

```
ai-crypto-backtester/
├── backtester/             # 백테스팅 관련 모듈
│   ├── __init__.py         # 모듈 초기화 파일
│   ├── analysis.py         # 성과 분석 모듈
│   ├── backtest.py         # 백테스팅 핵심 로직
│   ├── data.py             # 데이터 수집 및 관리
│   ├── decision_maker.py   # AI 결정 엔진
│   └── visualizer.py       # 결과 시각화 모듈
├── data/                   # 데이터베이스 및 데이터 파일
├── results/                # 백테스팅 결과 저장 (성과 분석, 차트 등)
├── main.py                 # 메인 실행 파일
├── config.json             # 백테스팅 및 AI 설정 파일
├── .env                    # 환경 변수 파일(생성 필요)
├── .gitignore              # Git 무시 파일 목록
├── requirements.txt        # 프로젝트 의존성
├── prompts/                # 프롬프트 파일 디렉토리
│   └── system_prompt.txt   # AI 시스템 프롬프트
└── README.md               # 프로젝트 문서
```

## 라이선스
이 프로젝트는 Apache License 2.0 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.
