#!/usr/bin/env bash
set -euo pipefail
baseline_dir=/data/lmy/lmy_home/Robbyant/deployment/robotwin-baseline-20261014
project_dir=/data/lmy/lmy_home/Robbyant/lingbot-vla-v2
sim_dir=/data/lmy/lmy_home/Robbyant/RoboTwin
export BASELINE_ORIGINAL_LD="${LD_LIBRARY_PATH:-}"
source "${baseline_dir}/conda-scoped.sh"
case "${1:-}" in
    env-check)
        conda activate lingbotvla
        python -c 'import sys; print("inference",sys.prefix)'
        conda activate RoboTwin
        python -c 'import sys,sapien; print("simulation",sys.prefix)'
        conda deactivate
        python -c 'import sys; print("deactivated",sys.prefix)'
        conda activate lingbotvla
        python -c 'import sys; print("inference again",sys.prefix)'
        ;;
    gpu-status)
        exec nvidia-smi --query-gpu=index,uuid,memory.total,memory.used,memory.free --format=csv -i 0
        ;;
    after-download)
        download_pid="$2"
        while [[ ! -f "${baseline_dir}/model-download-verification.json" ]]; do
            if ! kill -0 "$download_pid" 2>/dev/null; then
                echo 'Download process exited without final verification; refusing inference.' >&2
                exit 1
            fi
            sleep 5
        done
        bash "${baseline_dir}/run-baseline.sh" policy > "${baseline_dir}/verify-policy.log" 2>&1
        bash "${baseline_dir}/run-baseline.sh" evaluate > "${baseline_dir}/evaluate-launcher.log" 2>&1
        ;;
    client)
        conda activate RoboTwin
        cd "$sim_dir"
        exec python -c 'import sys; sys.path[:0]=["./policy","./script","./description/utils"]; import ACT, websockets.sync.client, msgpack, generate_episode_instructions; from deploy.websocket_client_policy import WebsocketClientPolicy; print("Official eval client imports passed")'
        ;;
    client-deps)
        conda activate RoboTwin
        unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
        exec python -m pip install websockets==15.0.1 msgpack==1.1.1
        ;;
    processor)
        conda activate lingbotvla
        exec python -c 'import json; from transformers import AutoConfig,AutoProcessor; p="/data/lmy/lmy_home/Robbyant/models/Qwen3-VL-4B-Instruct"; c=AutoConfig.from_pretrained(p,local_files_only=True); a=AutoProcessor.from_pretrained(p,local_files_only=True); result=dict(model_type=c.model_type,processor=type(a).__name__,tokenizer=type(a.tokenizer).__name__,vocab_size=len(a.tokenizer)); print(json.dumps(result)); open("/data/lmy/lmy_home/Robbyant/deployment/robotwin-baseline-20261014/processor-verification.json","w").write(json.dumps(result,indent=2))'
        ;;
    gpu)
        conda activate RoboTwin
        exec python -c 'import json,sapien; scene=sapien.Scene(); print(json.dumps(dict(pci=scene.render_system.device.pci_string,device=str(scene.render_system.device))))'
        ;;
    download-models|download-assets)
        conda activate lingbotvla
        mode="${1#download-}"
        unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
        exec python -u "${baseline_dir}/download_inputs.py" "$mode"
        ;;
    scene)
        conda activate RoboTwin
        cd "$sim_dir"
        python script/update_embodiment_config_path.py
        exec python -u "${baseline_dir}/verify_scene.py"
        ;;
    expert)
        conda activate RoboTwin
        cd "$sim_dir"
        exec python -u "${baseline_dir}/verify_scene.py" --expert
        ;;
    policy)
        conda activate lingbotvla
        cd "$project_dir"
        export QWEN3VL_PATH=/data/lmy/lmy_home/Robbyant/models/Qwen3-VL-4B-Instruct
        exec python -u "${baseline_dir}/verify_policy.py"
        ;;
    evaluate)
        conda activate lingbotvla
        cd "$project_dir"
        export QWEN3VL_PATH=/data/lmy/lmy_home/Robbyant/models/Qwen3-VL-4B-Instruct
        baseline_port="$(python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"
        exec bash experiment/robotwin/start_robotwin_infer_and_eval.sh \
            --model_path /data/lmy/lmy_home/Robbyant/models/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt \
            --eval_workdir "$sim_dir" --inference_workdir "${project_dir}/" \
            --output_base "${baseline_dir}/results" --conda_sh "${baseline_dir}/conda-scoped.sh" \
            --inference_env lingbotvla --sim_env RoboTwin --task_config demo_clean \
            --num_tasks 1 --num_gpus 1 --num_per_gpu 1 \
            --start_port "$baseline_port" \
            --use_bf16 False --use_fp32 True --use_compile True --no_video
        ;;
    *) echo 'Usage: run-baseline.sh {download-models|download-assets|scene|policy|evaluate}' >&2; exit 2 ;;
esac
