

<div align="center">
  <h1 style="font-size: 32px; font-weight: bold;"> [ICRA 2026] ActiveVLN: Towards Active Exploration via Multi-Turn RL in Vision-and-Language Navigation</h1>

  <br>

  <a href="https://arxiv.org/abs/2509.12618">
    <img src="https://img.shields.io/badge/ArXiv-ActiveVLN-brown?logo=arxiv" alt="Paper">
  </a>
  <a href="https://huggingface.co/collections/Arvil/activevln-68e7367d8bd2b426985f0c4a">
    <img src="https://img.shields.io/badge/🤗 huggingface-Model-purple" alt="checkpoint">
  </a>
</div>

## ActiveVLN
- 도커 서버 컨테이너 하나와, 추론 컨테이너를 각각 띄워서 api 형태로 추론 수행

#### 도커 서버 컨테이너 구축 및 실행
```
# Terminal1
docker pull vllm/vllm-openai:latest
docker run --rm --gpus all \
    --network host \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v /home/aprl/Desktop/when2reason/ActiveVLN_setup/ActiveVLN/model:/model:ro \
    nvcr.io/nvidia/vllm:25.09-py3 \
    vllm serve /model \
        --trust-remote-code \
        --limit-mm-per-prompt '{"image": 200, "video": 0}' \
        --mm_processor_kwargs '{"max_pixels": 80000}' \
        --max-model-len 32768 \
        --enable-prefix-caching \
        --gpu-memory-utilization 0.5 \
        --port 8003


export OPENAI_API_KEY=EMPTY
export OPENAI_API_BASE=http://172.17.0.1:8003/v1

# 연결 테스트
curl http://172.17.0.1:8003/v1/models

# 추론 실행
python run_infer_socialact_by_activevln.py --mission Nonverbal_001

```

#### 도커 추론 컨테이너 실행
```
python run_infer_socialact_by_activevln.py (--mission ###)
```
<img width="1081" height="1640" alt="image" src="https://github.com/user-attachments/assets/c8e49dcb-3bd8-429c-94a1-85d6fec1ddb4" />

- 특이점: ActiveVLN은 로컬 다이렉트 추론이 아닌 서버를 띄워서하는 api 형태의 추론으로 구성됨 그 이유는:
  + **핵심은 KV cache 재사용** — 로컬 model.generate()는 매 턴 대화 히스토리 전체를 다시 어텐션 계산 (O(N²)), vllm은 --enable-prefix-caching으로 공통 앞부분 어텐션 결과를 저장·재사용해서 새 토큰만 처리 (O(N))
  + **보조 가속은 vllm 엔진 최적화** — PagedAttention(메모리 파편화 제거), CUDA graphs(커널 런치 오버헤드 제거), FlashAttention 커스텀 커널이 더해져 순수 속도 자체도 빠름
  + **컨테이너 분리 자체는 속도와 무관**— 같은 GPU 쓰는 한 프로세스만 나뉜 것이고, 6배 차이는 전적으로 "raw PyTorch generate() vs 프로덕션 서빙 엔진 vllm"의 차이
