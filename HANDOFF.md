# CL-SFP 인수인계 노트

작성: 2026-09-22 (세션 끊김으로 새 세션 전환)
갱신: 2026-09-22 — transport / tool_hang 추가, 아래 «태스크 확장» 절 참조

## 프로젝트 요약

SFP(Streaming Flow Policy)를 **폐루프로 바꾸면 외란에 강인해지는가**를 검증.
`pyproject.toml` 저자: Siddharth Ancha, Sunshine Jiang (MIT). git 저장소 아님 —
원본 백업은 `algo/.orig/`에 수동 복사.

- 학습 코드: `algo/` / 시뮬레이터·데이터: `env/` / 결과: `outputs/`
- 태스크: `pusht`(pymunk 2D, perturb=pixels/step),
  `can`/`lift`/`square`/`transport`/`tool_hang`(robomimic, m/step)
- 태스크별로 다른 사실(obs 키·에피소드 예산·그리퍼 축·외란 대상)은
  `env/robomimic/tasks.py` 한 곳에 있음
- `pred_horizon=16, obs_horizon=2, action_horizon=8` → `dt = 1/14`,
  chunk 8스텝 실행 시 flow clock `t = 0 → 7/14 = 0.5`

## 실험 목록

### A. 실행(rollout) 축 — 가중치 동일, 실행만 변경
| 이름 | 내용 |
|---|---|
| SFP | 베이스라인. chunk 시작 obs 1장을 8스텝 고정, t=0→0.5 |
| SFP-1step | 매 스텝 chunk 갱신(obs 새로 읽음), clock은 **t=0 고정** |
| SFP-ah4 / ah2 | chunk 길이만 4 / 2. ah8↔ah1 사이 중간 지점 |
| CL-SFP | 매 스텝 obs 갱신 + clock 전진. **학습·추론 둘 다 반영** |
| CL-SFP-1step | CL-SFP 가중치, 실행은 매 스텝 t=0 → 2×2 ablation 완성 |
| CL-SFP+i-ah4 | interp 가중치를 ah4로 실행 |

### B. 학습(training) 축
| 이름 | 내용 |
|---|---|
| CL-SFP+i (`--cond-interp`) | 창을 `lerp(win_i, win_{i+1}, λ)`로 연속 보간. floor 인덱싱이 bin 경계에서 모순된 supervision을 주는 문제를 해결. **현재 최고 성능** |
| +σ_min 0.02 / 0.05 | `std = clamp(σ₀·exp(−kt), min=σ_min)` — 타깃 field는 그대로, 샘플링 support만 확장 |
| +SWA (`swa.py`) | 여러 체크포인트 EMA 가중치 균일 평균 |

### C. 적응형 flow clock (`algo/cl_sfp_gated.py`)
학습은 CL-SFP 그대로, 추론 시 clock 되감기.
`Δξ ≈ ‖v_new − v_old‖/k` → `t* = log(σ₀/γΔξ)/k` → `t ← min(t, t*)`, 격자에 snap.
게이트 4종: `soft`(γ) / `hard`(τ) / `excess`(정상 진행분 `‖v‖·dt` 초과분만) / `fixed`(t 상수).
`--mode calib`로 γ/τ 스윕 가능.

## 커버리지 (결과 JSON 기준, 숫자 = 학습 seed 수)

| | pusht | can | square | lift | transport | tool_hang |
|---|---|---|---|---|---|---|
| SFP | 3 | 3 | 3 | 3 | 진행 중 | 진행 중 |
| SFP-1step | 3 | 3 | 3 | 3 | — | — |
| SFP-ah4 | 3 | 3 | 3 | — | — | — |
| SFP-ah2 | 3 | **1** | **1** | — | — | — |
| CL-SFP | 3 | 3 | 3 | 3 | 진행 중 | 진행 중 |
| CL-SFP-1step | 3 | 3 | 3 | 3 | — | — |
| CL-SFP+i | 3 | 3 | 3 | — | 진행 중 | 진행 중 |
| CL-SFP+i-ah4 | 3 | 3 | 3 | — | — | — |
| +σ_min 0.02 | 3 | 3 | — | — | — | — |
| +σ_min 0.05 | 3 | 3 | — | — | — | — |
| +SWA | 3 | 3 | 3 | — | — | — |
| gated excess γ=3 | 3 | — | — | — | — | — |

transport / tool_hang은 **외란 0.0(정적)만** 측정한다. 나머지 열은 전부 외란 커브가 있다.

`lift`는 SFP가 이미 1.000 만점 → 의도적 제외로 보임.
**미완: square의 σ_min, gated 계열 전반(soft/hard/fixed 전부, robomimic 전부).**

## 주요 결과

`outputs/paired_tests.csv` (100 eval seeds, Wilcoxon/McNemar):
- perturb=0: SFP 근소 우위 또는 동률 (pusht 0.923 vs CL-SFP 0.920)
- 고외란: CL-SFP 우위. pusht p=1.4에서 0.512→0.748 (p<1e-4),
  square pooled 0.282→0.346 (p=0.007), can pooled 0.832→0.868
- **반증**: `sfp_1step`이 pusht 고외란에서 CL-SFP보다 좋음 →
  "시간 정렬 conditioning"보다 "obs 신선도"가 더 중요할 수 있음.
  이게 `cl_sfp_gated.py`의 동기.
- `final_curves.csv`: CL-SFP+i가 square 0.800 / can 0.947로 최고

## 열린 질문 / 다음 할 일

1. **square σ_min, gated 계열 채우기** — 위 표의 빈 칸
2. ~~smin/swa가 interp를 실제로 이겼는지 검증~~ → **2026-09-22 완료. 아래 참조**
3. ~~SFP-ah4 수치 출처 확정~~ → **2026-09-22 완료. 아래 참조**
4. **(신규) square/sfp_seed2 ah4를 ckpt 300으로 재평가** — 3-seed 동일-ckpt ah4 비교에
   빠진 유일한 셀. `.sfpckpt` 파일이 seed0/seed1엔 있고 seed2엔 없음
5. **(신규) square ah8 sweep을 ep700-1000까지 확장** — ah4/ah2는 1000까지 훑었는데
   ah8(베이스라인)은 100-600만 훑음. 선택 예산이 불균등

## 2026-09-22 감사 결과

### (a) SFP-ah4 수치 출처 — 확정

`final_summary.py`의 `robomimic_sources()`는 `perturb_ah4_s100_episodes.json`(= `.sfpckpt`
**아닌** 쪽)을 읽는다. 이 파일들은 **해당 ah에서 다시 돌린 static sweep의 argmax ckpt**를 쓴다.

| | ah8(SFP 베이스) ckpt | ah4 ckpt | 일치? |
|---|---|---|---|
| can seed0 | 400 | 300 | ✗ |
| can seed1 | 300 | 300 | ✓ |
| can seed2 | 300 | 300 | ✓ |
| square seed0 | 300 | 600 | ✗ |
| square seed1 | 400 | 1000 | ✗ |
| square seed2 | 300 | 600 | ✗ |

**영향이 크다.** square seed0에서 ah4를 SFP와 같은 ckpt 300으로 돌리면 static 0.74 → **0.50**,
모든 외란 레벨에서 하락(0.0005: 0.40→0.32). can seed0은 반대로 같은 ckpt 400이 **더 좋다**
(0.0005: 0.71→0.88). 즉 현 `final_curves.csv`의 SFP-ah4는 square에선 부풀려져 있고
can에선 저평가돼 있다.

**원인:** `fair_eval.sh`는 `--ckpt 100-600`으로 훑지만, ah4/ah2 sweep(`sweep_s100_ah4.json`)은
**100-1000**으로 수동 실행됨. square seed1의 ah4가 고른 ckpt 1000은 ah8 베이스라인이
애초에 볼 수 없던 구간. → ah 축 비교가 "실행 길이" 효과와 "ckpt 선택 예산" 효과를 섞고 있음.

정리 방향 두 가지 중 택일이 필요:
- **동일 ckpt 프로토콜**: 전부 SFP의 ah8 ckpt로 고정(`.sfpckpt` 파일들이 이 용도).
  누락 1칸 = square/sfp_seed2 @ ckpt 300.
- **동일 예산 프로토콜**: ah8도 ep700-1000까지 훑어서 재선택.

### (b) σ_min / SWA vs interp — `compare_variant.py` 3-seed, 100 eval seeds

| 변형 | 태스크 | static | robust band | 판정 |
|---|---|---|---|---|
| σ_min 0.02 | pusht | 0.878 → **0.785** | 0.556 → 0.554 (p=0.87) | **패배**. static만 깎임 |
| σ_min 0.05 | pusht | 0.878 → **0.761** | 0.556 → **0.514** (p=9e-10) | **완패** |
| SWA | pusht | 0.878 → **0.890** | 0.556 → **0.573** (p=5e-3) | **승리**. 양쪽 다 개선 |
| σ_min 0.02 | can | 0.947 → 0.960 | 0.624 → **0.576** (p=5e-4) | **패배**. 고외란에서 붕괴 |
| σ_min 0.05 | can | 0.947 → **0.890** | 0.624 → **0.573** (p=9e-4) | **완패** |
| SWA | can | 0.947 → **0.893** | 0.624 → **0.656** (p=1.8e-2) | **트레이드오프** |
| SWA | square | 0.800 → 0.763 (p=0.14) | 0.382 → **0.434** (p=2.6e-6) | **승리**(robust) |

**결론: σ_min은 실패한 아이디어다.** 두 태스크 모두에서 robust band가 나빠지거나 그대로이고
static은 항상 깎인다. 샘플링 support를 넓히는 것만으로는 외란 강인성이 오지 않음.
표의 빈 칸(square σ_min)을 채울 이유가 약해짐 — 우선순위 내릴 것.

**SWA는 살아있다.** pusht는 static·robust 동시 개선, square는 robust +0.052(p=2.6e-6),
can은 static −0.053 대신 robust +0.032. 3태스크 중 2개에서 확실한 이득.

## 2026-09-22 태스크 확장 — transport, tool_hang

robomimic에서 low_dim이 있는 시뮬 태스크는 5개뿐이고, 남아 있던 두 개를 붙였다.
`transport`(양팔, TwoArmTransport)와 `tool_hang`(ToolHang). 나머지 3개
(`*_real`)는 실물 데이터라 `raw`만 있고 시뮬레이터가 없다.

### 데이터셋 — 반드시 레거시 URL을 쓸 것

`robomimic/scripts/download_datasets.py`는 **v1.5 데이터셋**을 받는데, 이건
robosuite 1.5 포맷(`BASIC` 합성 컨트롤러, `lite_physics` kwarg)이라 이 프로젝트의
**robosuite 1.4.1로는 env 생성 자체가 실패**한다 (`TypeError: TwoArmTransport.
__init__() got an unexpected keyword argument 'lite_physics'`). 기존 can/lift/square는
`env_version=None`인 구버전 파일이고, robosuite를 올리면 그 30G가 전부 무효가 된다.

    curl -L -o env/robomimic/data/<task>/low_dim.hdf5 \
      http://downloads.cs.stanford.edu/downloads/rt_benchmark/<task>/ph/low_dim.hdf5

받은 파일의 `env_version`이 `None`인지 확인할 것. 값이 있으면 잘못된 파일이다.

### 에피소드 예산 (`MAX_STEPS`)

`max_steps`가 세 rollout 함수에 250으로 하드코딩돼 있던 것을 태스크별 표로 옮겼다.

| | can/lift/square | transport/tool_hang |
|---|---|---|
| robomimic 공식 horizon | 400 | 700 |
| 이 프로젝트 | **250 (유지)** | **700** |

기존 셋을 250으로 둔 이유: `outputs/` 아래 모든 결과가 250에서 측정됐고, 올리면
비교가 깨진다. **즉 현재 표는 robomimic 표준 프로토콜이 아니다** — 논문에 그렇게
쓰면 안 된다. `--max-steps 400`으로 재측정할 수 있고, 기본값과 다르면 sweep 파일이
`..._ms400.json`으로 분리돼 기존 파일을 덮지 않는다. 측정값은 결과 JSON의
`max_steps` 필드에 기록된다.

robomimic env는 `ignore_done: true`라 **모든 에피소드가 예산을 끝까지 채운다.**
성공해도 조기 종료하지 않으므로 250→700은 비용이 그대로 2.8배다.

### 고친 버그 3건 (전부 조용히 잘못된 결과를 내는 종류)

1. **외란 대상 물체가 임의로 정해짐.** `_resolve_object_joint`가 workspace 안
   첫 free joint를 잡고 `break`했다. can/lift/square는 workspace 내 free joint가
   정확히 1개라 결과에 영향은 없었지만, transport는 **6개**
   (payload, trash, 시작/목표/쓰레기 bin, lid), tool_hang은 **3개**
   (stand, frame, tool)다. 이제 `tasks.PERTURB_OBJECT`로 명시하고, 지정 없이
   모호하면 **에러**를 낸다. 외란은 이 실험의 독립변수라 추측하면 같은 이름으로
   다른 실험을 하는 셈이다.
2. **transport의 두 번째 팔이 관측에서 빠짐.** `DEFAULT_OBS_KEYS`가 `robot0_*`만
   있었고, `env.py`와 `data.py`에 **중복 정의**돼 있었다. `tasks.py`로 합치고
   transport에 `robot1_eef_pos/quat/gripper_qpos`를 추가 (obs 50 → 59).
3. **transport의 그리퍼 램프가 한쪽 팔에만 적용됨.**
   `interpolate_binary_gripper_transitions(gripper_dim=-1)`이 마지막 축만 부드럽게
   만든다. transport는 그리퍼가 축 **6과 13** 두 개(데이터로 확인: 둘 다 100%가
   ±1, 고유값 2개)라 오른팔 그리퍼가 binary step으로 남았다. 이건 이 함수가
   존재하는 이유("step은 속도가 무한대라 flow policy가 표현 못 함") 자체가
   한쪽 팔에서 무효가 된다는 뜻이다. 여러 축을 받도록 일반화했고, 단일 그리퍼
   경로는 기존과 바이트 단위로 동일함을 확인했다.

### 검증한 것

| | obs | action | max_steps | 외란 대상 |
|---|---|---|---|---|
| transport | 59 | 14 | 700 | `payload_joint0` |
| tool_hang | 53 | 7 | 700 | `frame_joint0` |

- 드리프트 물리 검증: perturb 0.001로 50스텝 → xy 이동 0.050 m (= 50×0.001), 레벨 0에서는 0
- 학습 루프 6개 설정(2 task × sfp/cl_sfp/interp) 전부 정상, 65M 파라미터
- rollout 4경로(sfp / cl_sfp / 1step / gated) 전부 `max_steps=700` 기록·실행 확인
- **기존 3태스크 회귀 없음**: obs 차원, 250스텝, 선택되는 물체 joint 모두 이전과 동일

### 측정된 비용

| | 학습 1000ep | 에피소드 1개 | sweep 1조합(600 ep) |
|---|---|---|---|
| square (기존, 250스텝) | 10분 | — | — |
| transport | 37분 | 26.6초 | 4.4 h |
| tool_hang | 33분 | 9.0초 | 1.5 h |

학습 속도는 윈도 수에 정확히 비례 (square 28,754 / transport 92,352 / tool_hang 94,562).
**평가가 압도적으로 비싸다** — 학습은 전체의 15% 미만.

### 진행 중인 실행

`train_newtasks.sh`. method는 **sfp / cl_sfp / cl_sfp+interp 3종만**, 외란은
**0.0만** (정적 ckpt 스윕). 두 태스크를 GPU 0/1에 나눠 병렬 실행 중이고,
로그는 `logs/transport_*.log`, `logs/tool_hang_*.log`.

결과 집계는 `static_summary.py` (신규). `final_summary.py` / `paired_test.py`는
외란 커브가 있다고 가정하므로 이 결과를 읽지 못한다 — 나중에 외란을 붙이면
그때 `robomimic_sources()` / `ROBUST_BAND` / `MID_BAND`에 새 태스크를 추가해야 한다.

### 미결

1. **tool_hang의 외란 대상은 아직 실질적으로 안 정했다.** 지금 `frame`으로 둔 건
   perturb 0.0에서는 무의미하기 때문(드리프트가 적용되지 않음). 외란 실험을 붙일 때
   실제 선택이 필요하다:
   - `frame` — 정책이 먼저 집는 물체, can/nut의 직접 유추. 단 집어서 2cm 이상 들면
     드리프트가 멈춰 후반부는 무외란
   - `tool` — 태스크 이름이 가리키는 물체. 에피소드 대부분 테이블에 남아 계속 밀리고,
     1단계(frame 삽입)는 무외란으로 남음
   - `stand` — 유일하게 그랩-리프트 예외에 걸리지 않아 전 구간 드리프트. 두 번의
     삽입 모두가 움직이는 표적
2. **외란 레벨이 새 태스크에 맞게 조정되지 않았다.** `fair_eval.sh`의
   `0.0001~0.0015 m/step`은 can/square 기준이다. tool_hang은 정밀 삽입이라 과할
   가능성이 높고 transport는 반대로 모자랄 수 있다. 3시드 돌리기 전에 1시드로
   커브 모양부터 볼 것.
3. **`sweep_driver`의 조기 종료가 method별로 다른 ckpt 예산을 준다.** ep600 이후
   개선될 때만 연장하므로, 한 method가 ep1000까지 보고 다른 method가 600에서 멈추면
   ah 축에서 이미 겪은 «선택 예산» 문제가 method 축에서 반복된다.
   `static_summary.py`가 선택된 ckpt를 같이 출력하는 이유다.

## 실행 시 주의

- `train_variant.sh` / `fair_eval.sh` / `train_interp.sh`가
  `LD_LIBRARY_PATH=/workspace/IKEA-Assembly/.native-cuda/lib`를 넣음 → 경로 존재 확인 필요
- `MUJOCO_GL=egl`, pusht는 `PUSHT_PERTURB_DIR=random`
- 파이썬은 `.venv/bin/python`
- `pyproject.toml`이 README.md를 선언하지만 실제 파일 없음
- robomimic 데이터셋은 **레거시(v0.1) URL**에서 받을 것. 공식 다운로드
  스크립트는 robosuite 1.5용 v1.5 파일을 주고, 이 환경에서는 못 연다
- `--max-steps`를 기본값과 다르게 주면 sweep 파일이 `_ms<N>`으로 분리된다
