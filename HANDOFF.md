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
| SFP | 3 | 3 | 3 | 3 | 3† | 3† |
| SFP-1step | 3 | 3 | 3 | 3 | — | — |
| SFP-ah4 | 3 | 3 | 3 | — | — | — |
| SFP-ah2 | 3 | **1** | **1** | — | — | — |
| CL-SFP | 3 | 3 | 3 | 3 | 3† | 3† |
| CL-SFP-1step | 3 | 3 | 3 | 3 | — | — |
| CL-SFP+i | 3 | 3 | 3 | — | 3† | 3† |
| CL-SFP+i-ah4 | 3 | 3 | 3 | — | — | — |
| +σ_min 0.02 | 3 | 3 | — | — | — | — |
| +σ_min 0.05 | 3 | 3 | — | — | — | — |
| +SWA | 3 | 3 | 3 | — | — | — |
| gated excess γ=3 | 3 | — | — | — | — | — |

† transport / tool_hang은 **외란 0.0(정적)만**, 예산 700스텝. 나머지 열은 전부 외란 커브가 있다.

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

### 실행 기록

`train_newtasks.sh`(학습) → `eval_newtasks_parallel.sh`(평가). method는 **sfp / cl_sfp /
cl_sfp+interp 3종만**, 외란은 **0.0만** (정적 ckpt 스윕). 두 태스크를 GPU 0/1에 나눠
병렬 실행, 2026-09-22 21:43 완료. 로그는 `logs/`.

결과 집계는 `static_summary.py` (신규). 새 태스크 붙일 때 사전 점검은
`check_dataset_env.py <task>`, 학습 후 적합도 확인은 `check_train_fit.py <task> <ep>`. `final_summary.py` / `paired_test.py`는
외란 커브가 있다고 가정하므로 이 결과를 읽지 못한다 — 나중에 외란을 붙이면
그때 `robomimic_sources()` / `ROBUST_BAND` / `MID_BAND`에 새 태스크를 추가해야 한다.

### 결과 — 정적 성공률, 3 학습시드 × 100 eval seeds (2026-09-22 21:43 완료)

`outputs/static_summary.csv`. ckpt는 시드별 static argmax(`sweep_driver` 규칙: ep100–600 필수,
이후 개선될 때만 연장).

| | SFP | CL-SFP | CL-SFP+i |
|---|---|---|---|
| **transport** | .403 ±.021 | .387 ±.038 | .407 ±.040 |
| **tool_hang** | .240 ±.044 | **.093 ±.035** | **.337 ±.047** |

**transport: 세 방법이 노이즈 안에서 동률.** 기존 태스크와 같은 패턴(외란 0에서는 폐루프가
이득도 손해도 없음). 외란을 걸어야 분리가 생길 자리다. 학습 곡선이 **ep100–400에서 정점을
찍고 내려간다**(SFP seed0: .41@200 → .26@600). 9개 중 4개가 ep100을 골랐다. 200 데모 /
65M 파라미터 / 1000 epoch cosine이라 과적합 쪽 신호로 읽히고, 이 태스크는 epoch 예산을
줄이거나 ckpt를 더 촘촘히(50 단위) 훑는 게 맞을 수 있다.

**tool_hang: 순서가 선명하다 — CL-SFP+i > SFP > CL-SFP.**
- 평범한 CL-SFP(floor 인덱싱)가 SFP의 **40% 수준**(.093 vs .240)으로 무너진다. 기존 태스크에서
  static 손실은 .00–.06이었는데 여기선 .15다.
- interp가 그걸 뒤집어 SFP를 **+.10(≈2 SE)** 앞선다. can/square에서 "interp가 최고"였던
  결과가 정밀 삽입 태스크에서 가장 크게 재현된 것.
- 해석: floor 인덱싱의 격자 경계 불일치는 conditioning이 한 스텝 어긋나는 오차인데,
  tool_hang은 그 한 스텝이 삽입 성패를 가르는 태스크다. interp는 그 오차를 제거한다.
  **"정밀도가 중요할수록 시간 정렬 conditioning의 정합성이 중요하다"**는 주장의 가장 강한 증거.
- 학습은 ep300부터 시작되고(그 전 0%) 정점이 ep500–700. 조기종료가 정점 근처에서 끊는다.

주의: static argmax는 winner's curse가 있다. 100 seeds에서 p≈.4의 이항 SE는 ≈.05이므로 ckpt
간 .05 차이는 노이즈이고, 6–8개 중 최대를 고르면 대략 그만큼 부풀려진다. 세 방법에 동일하게
걸리지만, 절대값을 읽을 때 감안할 것.

### 외란 커브 — 3 학습시드 × 100 eval seeds (2026-09-23 08:02 완료)

`perturb_s100_episodes.json`, 그림 `outputs/perf_transport_toolhang.png` (`plot_drift_newtasks.py`).
레벨은 파일럿(`perturb_pilot_*_s50.json`)으로 정했다: 700스텝은 같은 per-step 드리프트가 2.8배
누적되므로 can/square 밴드(0.1–1.5 mm)보다 아래. tool_hang은 **frame** 드리프트 — stand는
0.05 mm/step부터 전원 0이라 반응성이 아니라 표적 이탈을 재는 실험이 되어 기각.

| transport (mm/step) | 0 | 0.05 | 0.10 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|---|---|
| SFP | .403 | .360 | .343 | .187 | .057 | .053 | .043 |
| CL-SFP | .387 | .330 | .350 | .200 | .097 | .047 | .030 |
| CL-SFP+i | .407 | .403 | .390 | .223 | .093 | .067 | .050 |
| **+i − SFP, McNemar p** | 1.0 | .26 | .22 | .25 | **.05** | .56 | .82 |

밴드 0.1–0.5 mm pooled(n=900): **+.040, p=.017**. 레벨별로는 유의하지 않고 pooled에서만
걸린다. 크기는 작다 — 세 방법이 거의 같은 속도로 무너진다.

| tool_hang (mm/step) | 0 | 0.10 | 0.25 | 0.50 | 0.75 | 1.00 | 1.50 |
|---|---|---|---|---|---|---|---|
| SFP | .240 | .193 | .220 | .110 | .090 | .067 | .013 |
| CL-SFP | .093 | .047 | .073 | .067 | .037 | .023 | .007 |
| CL-SFP+i | .337 | .317 | .337 | .187 | .143 | .090 | .033 |
| **+i − SFP, McNemar p** | .014 | .001 | .001 | .011 | .036 | .36 | .15 |

밴드 0.25–1 mm pooled(n=1200): **+.068, p<1e-4**. **7레벨 중 5개에서 유의**, 부호는 7/7.
플레인 CL-SFP는 전 구간에서 SFP 아래 — floor 인덱싱의 격자 경계 불일치가 정밀 삽입 태스크에서
치명적이라는 정적 결과가 외란에서도 유지된다.

**읽는 법.** 두 태스크 모두 격차가 외란과 함께 **벌어지지 않는다** — tool_hang은 비율(~1.4배)로
유지되고 transport는 노이즈 수준. 이는 앞서 적은 혼동 요인 그대로다: 700스텝 태스크에서 chunk 내
낡음이 의미 있어지는 레벨(≥1 mm/step, chunk당 ≥8 mm)은 곧 누적 이탈(≥30 cm)이 전부를 파괴하는
레벨이다. **이 프로토콜로는 긴 horizon 태스크에서 "폐루프의 외란 강인성"을 분리해 보일 수 없다.**
tool_hang의 interp 이득은 강인성이 아니라 정밀 태스크를 더 잘 푸는 것으로 읽어야 한다.
분리하려면 누적 없는 외란(무작위 시점 impulse 변위 등)이 별도 축으로 필요하다 — 설계 변경이라 미결.

### 절대 좌표 액션 실험 — DP 레시피 검증 (2026-09-24)

DP와의 격차(square .75 vs .96, tool_hang .24 vs .93)가 액션 공간 때문인지 확인. `--abs-action`:
`actions_abs`(pos + 축각 + 그리퍼, `convert_abs_actions.py`)를 rot6d로 바꿔 10차원으로 학습,
OSC `control_delta=False`. 나머지 전부 고정. 3시드 정적 스윕. `outputs/static_summary_abs.csv`.

| | square 델타 | square **절대** | tool_hang 델타 | tool_hang **절대** |
|---|---|---|---|---|
| SFP | .750 | .750 (=) | .240 | **.433** (+.19) |
| CL-SFP | .740 | .563 (−.18) | .093 | .123 (+.03) |
| CL-SFP+i | .800 | .530 (−.27) | .337 | .273 (−.06) |

**결론.**
1. **DP 격차는 액션 공간이 아니다.** SFP는 square에서 그대로(.75)이고 tool_hang에서 올라도(.43)
   DP의 .93에는 한참 못 미친다. 격차의 주인은 방법(스트리밍 vs 16스텝 결합 denoising)이거나
   학습 레시피다 → DP를 같은 프로토콜로 직접 돌려 확인 중(`algo/dp.py`).
2. **절대 액션은 open-loop 실행(SFP)에는 이득이고 폐루프 변형에는 손해다.** 정밀 태스크에서
   SFP가 거의 2배가 된 건 절대 목표가 오차를 누적하지 않기 때문(리플레이 0%→33%와 일치).
   반면 CL-SFP(+i)는 두 태스크 모두에서 무너진다.
3. 왜 폐루프가 무너지는지는 **미확정**. "현재 자세를 복사해 제자리에 멈춘다"는 가설은 기각됐다
   (롤아웃 이동거리 동일, 명령 목표가 현재 eef보다 2.7 cm 앞). 남는 후보는 학습 loss 신호다 —
   절대 모드에서 CL-SFP의 loss가 SFP의 절반 이하(tool_hang .0095 vs .026, square .034 vs .05)인데
   성공률은 낮다. 매 스텝 갱신되는 창 속 eef 자세가 직전 절대 목표를 거의 그대로 담고 있어
   "다음 목표 ≈ 현재 자세 + 관성"이라는 지름길을 배웠을 가능성(인과 혼동). SFP는 창이
   1~8스텝 낡아 그 지름길이 약하다. 확인하려면 절대 모드 학습 적합도(`check_train_fit.py`)를
   델타와 비교하면 된다.

이 프로젝트의 주장(폐루프 + interp)은 **델타 액션 위에서** 성립하고, 절대 액션으로 옮기면
성립하지 않는다. 논문에 절대 액션을 쓰려면 폐루프 conditioning에서 자기 자세를 빼거나
지연시키는 설계가 먼저 필요하다.

### 첫 결과와 "버그 아닌가" 진단 (2026-09-22)

첫 시드의 정적 스윕이 낮게 나왔다 — `tool_hang/sfp` 최고 0.26(ep500), `tool_hang/cl_sfp`
0.09, `transport/sfp` ep100–300에서 0.35–0.41. 문헌(Diffusion Policy: tool_hang ~0.7–0.9,
transport ~0.9–1.0)에 크게 못 미쳐서 데이터셋·학습·추론 배관을 전부 점검했다.
`check_dataset_env.py`와 `check_train_fit.py`가 그 검사이고, 새 태스크를 붙일 때마다 먼저 돌릴 것.

| 검사 | can (대조) | square | tool_hang | transport |
|---|---|---|---|---|
| 기록된 데모 성공률 | 100% | — | 100% | 100% |
| 정규화 상수 차원 | 없음 | — | 없음 | 없음 |
| 초기 상태 리셋 오차 | 1e-7 | — | 3e-7 | 1e-7 |
| **1스텝 물리 오차** (eef, 중앙값) | 3.6e-4 m | — | 9.4e-5 m | 1.8e-4 m |
| 데모 액션 open-loop 리플레이 | 100% | — | 0% | 33% |
| **학습 적합도** model/stay 오차비 | 0.46 | 0.59 | 0.46 | 0.52 |

**결론: 버그 증거 없음.**
- 물리는 일치한다. mujoco 3.8.1 + robosuite 1.4.1이 v0.1 데이터셋의 전이를 세 태스크 모두
  서브밀리미터로 재현한다(tool_hang이 가장 정확). "구버전 데이터셋 = MuJoCo 2.0 물리" 우려는 기각.
- 학습은 됐다. 데이터셋 obs로 flow를 8스텝 적분했을 때 기록 액션에 맞아가는 정도가
  0.94짜리 can과 0.26짜리 tool_hang에서 **같다**. 네트워크는 두 태스크를 똑같이 잘 맞춘다.
- open-loop 리플레이 0%는 버그 지표가 아니다. 1e-4 m/스텝이 500스텝 누적되면 삽입 허용 오차
  (수 mm)를 넘는다. 폐루프 정책은 매 스텝 보정하므로 무관하고, robomimic 자체 재생 도구도
  같은 이유로 액션 아닌 **상태**를 재생한다.

**낮은 이유로 남는 것** (전부 method 간 비교에는 동일하게 걸리므로 상대 비교는 유효):
1. 태스크 난이도 × 모델 한계 — robomimic 논문에서 tool_hang은 history 없는 BC가 무너지고
   BC-RNN만 살아남는 태스크. 이 SFP는 `obs_horizon=2`, RNN 없음. square에서도 DP보다 0.2 낮았다.
2. 예산 700이 빡빡함 — tool_hang 데모의 18%가 560스텝 초과, 3%가 700 초과. 사람보다 느린
   정책은 성공 직전에 잘린다 (robomimic 공식 horizon도 700이라 참조 수치도 같은 조건).
3. 하이퍼파라미터가 can/square용 그대로.

절대 수치를 문헌 옆에 놓지 말 것. 250스텝 프로토콜과 같은 맥락이다.

### 병렬 평가와 pids 제한 (2026-09-22)

평가는 MuJoCo 단일 스레드에 묶여 GPU를 5%만 쓴다. `sweep_driver`를 (method, seed) 9조합
순차로 돌리면 transport가 ~40h라, `eval_newtasks_parallel.sh`로 9개를 동시에 띄운다(~5h).

그때 드러난 함정: 이 컨테이너는 **pids 4096 제한**이고, 128코어라 OpenMP 풀(numpy·torch·
MuJoCo)이 프로세스마다 스레드 130~380개를 만든다. 9개 이상 동시에 뜨면 한도를 넘고, 실패 양상이
조용하다 — MjModel 컴파일에서 `Caught an unknown exception!` 또는 `libgomp: Thread creation
failed`, 혹은 **에러 없이 futex 데드락**(시작 시에도, 작업을 다 끝낸 뒤 종료 단계에서도).
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`로 프로세스당 ~20스레드가 되고
(pids 2027 → 193), 평가는 단일 스레드 시뮬 + batch-1 추론이라 손실이 없다. 런처가 이를
설정하고 기동 간격 5초·실패 시 1회 재시도를 넣는다. **이 env 없이 병렬로 띄우지 말 것.**

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
