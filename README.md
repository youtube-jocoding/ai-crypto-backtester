# AI Crypto Backtester

AI 기반의 암호화폐 트레이딩 전략을 백테스팅할 수 있는 Python 기반 도구입니다. 이 프로젝트는 Binance API를 사용하여 과거 데이터를 가져오고, 다양한 AI 모델을 활용하여 백테스트를 진행할 수 있습니다.

## 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/youtube-jocoding/ai-crypto-backtester.git
cd ai-crypto-backtester
```

2. 의존성 설치
```bash
pip install -r requirements.txt
```

3. 환경 변수 설정
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

3. **백테스팅 실행**
```bash
python main.py
```
백테스트가 실행되고 결과는 `results/` 디렉토리에 HTML 파일로 저장됩니다.

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
