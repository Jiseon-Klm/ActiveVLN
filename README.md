

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
