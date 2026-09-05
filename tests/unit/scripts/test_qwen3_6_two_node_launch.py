"""Qwen3.6 双节点入口的无 GPU 回归测试，替代服务只记录实际启动参数。"""

import json
import os
import shlex
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from torch.distributed.run import get_args_parser

from speculators.train.config import TrainConfig

REPO = Path(__file__).resolve().parents[3]
SCRIPT_DIR = REPO / "examples/train/nnode"
METHODS = ("domino", "dspark", "dflash2")
NETWORK_ENV = {
    "NCCL_IB_QPS_PER_CONNECTION": "4",
    "NCCL_GDR_LEVEL": "2",
    "NCCL_IB_PCI_RELAXED_ORDERING": "1",
    "NCCL_IB_TC": "160",
    "NCCL_NVLS_ENABLE": "0",
    "NCCL_IB_GID_INDEX": "3",
    "GLOO_SOCKET_IFNAME": "eth-test",
    "NCCL_SOCKET_IFNAME": "eth-test",
    "NCCL_DEBUG": "INFO",
    "NCCL_IB_TIMEOUT": "23",
    "NCCL_IB_RETRY_CNT": "7",
    "NCCL_IB_HCA": "mlx5_0,mlx5_1",
}


@pytest.fixture
def launch_env(tmp_path):
    model = tmp_path / "model_weights/teacher"
    data = tmp_path / "datasets/prepared"
    mock_bin = tmp_path / "bin"
    capture = tmp_path / "capture"
    for path in (model, data, mock_bin, capture):
        path.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    (data / "state.json").write_text("{}")
    (data / "dataset_info.json").write_text("{}")

    fake_launcher = mock_bin / "capture_launch"
    fake_launcher.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            """\
            import json
            import os
            import signal
            import sys
            import time
            from pathlib import Path

            kind = "train" if "--nproc_per_node" in sys.argv else "vllm"
            capture = Path(os.environ["NNODE_TEST_CAPTURE"])
            rank = os.environ["PET_NODE_RANK"]
            if kind == "vllm" and os.environ.get("NNODE_TEST_VLLM_FAIL") == "1":
                sys.exit(7)
            env = {
                key: value for key, value in os.environ.items()
                if key.startswith(("PET_", "NCCL_", "GLOO_")) or key in (
                    "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
                    "MASTER_ADDR", "MASTER_PORT", "CUDA_VISIBLE_DEVICES",
                    "NO_PROXY", "no_proxy",
                )
            }
            (capture / f"{kind}_node{rank}.json").write_text(json.dumps({
                "argv": sys.argv[1:], "env": env, "pid": os.getpid(),
            }))
            if kind == "vllm":
                signal.pause()
            else:
                expected = int(os.environ.get("NNODE_TEST_EXPECT_NODES", "1"))
                deadline = time.monotonic() + 10
                while not all(
                    (capture / f"train_node{i}.json").exists()
                    for i in range(expected)
                ):
                    if time.monotonic() >= deadline:
                        sys.exit(19)
                    time.sleep(0.05)
                time.sleep(0.2)
                sys.exit(int(os.environ.get("NNODE_TEST_TRAIN_EXIT", "0")))
            """
        )
    )
    fake_launcher.chmod(0o755)
    curl = mock_bin / "curl"
    curl.write_text(
        '#!/bin/bash\n[[ -f "$NNODE_TEST_CAPTURE/vllm_node${PET_NODE_RANK}.json" ]]\n'
    )
    curl.chmod(0o755)

    # 只继承运行测试所需的基础环境，避免真实训练配方变量影响断言。
    env = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "HOME", "LD_LIBRARY_PATH", "LANG", "SYSTEMROOT"}
    }
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env.update(
        NETWORK_ENV,
        ROOT=str(tmp_path),
        REPO=str(tmp_path / "speculators"),
        MODEL=str(model),
        DATA_DIR=str(data),
        SPEC_PYTHON=sys.executable,
        VLLM_PYTHON=str(fake_launcher),
        TORCHRUN=str(fake_launcher),
        LAUNCH_VLLM=str(REPO / "scripts/launch_vllm.py"),
        TRAIN_SCRIPT=str(REPO / "scripts/train.py"),
        WANDB_MODE="offline",
        PET_NNODES="2",
        PET_NPROC_PER_NODE="8",
        PET_NODE_RANK="0",
        PET_MASTER_ADDR="pet-master.example",
        PET_MASTER_PORT="29501",
        MASTER_ADDR="master.example",
        MASTER_PORT="29500",
        WORLD_SIZE="16",
        RANK="8",
        LOCAL_WORLD_SIZE="8",
        LOCAL_RANK="0",
        CUDA_VISIBLE_DEVICES="7,6,5,4,3,2,1,0",
        VLLM_PORT=str(port),
        NO_PROXY="existing.example",
        no_proxy="existing.example",
        PATH=f"{mock_bin}:{os.environ['PATH']}",
        NNODE_TEST_CAPTURE=str(capture),
        PYTHONDONTWRITEBYTECODE="1",
    )
    return env


def run_nodes(method, env, ranks):
    script = SCRIPT_DIR / f"{method}_qwen3_6_35b_a3b_perfectblend_online_2node.sh"
    processes = []
    try:
        for rank in ranks:
            processes.append(
                subprocess.Popen(  # noqa: S603
                    ["/bin/bash", str(script)],
                    env={**env, "PET_NODE_RANK": str(rank)},
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
            )
        return [(p.communicate(timeout=20)[0], p.returncode) for p in processes]
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.communicate(timeout=40)


def flag(argv, name):
    return argv[argv.index(name) + 1]


def read_capture(env, kind, rank):
    path = Path(env["NNODE_TEST_CAPTURE"]) / f"{kind}_node{rank}.json"
    return json.loads(path.read_text())


def assert_teacher_stopped(record):
    with pytest.raises(ProcessLookupError):
        os.kill(record["pid"], 0)
    assert not Path(flag(record["argv"], "--hidden-states-path")).exists()


@pytest.mark.parametrize("method", METHODS)
def test_two_nodes_share_ddp_but_isolate_teacher(method, launch_env, monkeypatch):
    env = launch_env
    env["NNODE_TEST_EXPECT_NODES"] = "2"
    if method == "dspark":
        # 同时覆盖 PET_MASTER 回退，以及保留平台显式网络设置。
        env.pop("MASTER_ADDR")
        env.pop("MASTER_PORT")
        env.pop("GLOO_SOCKET_IFNAME")
        env["NCCL_CROSS_NIC"] = "1"
    master_addr = env.get("MASTER_ADDR", env["PET_MASTER_ADDR"])
    master_port = env.get("MASTER_PORT", env["PET_MASTER_PORT"])
    outputs = run_nodes(method, env, (0, 1))
    for output, returncode in outputs:
        assert returncode == 0, output

    configs = []
    for rank in (0, 1):
        teacher = read_capture(env, "vllm", rank)
        train = read_capture(env, "train", rank)
        assert teacher["env"]["CUDA_VISIBLE_DEVICES"] == "7,6"
        assert train["env"]["CUDA_VISIBLE_DEVICES"] == "5,4,3,2,1,0"
        for record in (teacher, train):
            for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
                assert name not in record["env"]
            for name, value in NETWORK_ENV.items():
                assert record["env"][name] == value
            assert record["env"]["NCCL_CROSS_NIC"] == env.get("NCCL_CROSS_NIC", "0")
            for name in ("NO_PROXY", "no_proxy"):
                assert record["env"][name] == "existing.example,127.0.0.1,localhost"
        assert "MASTER_ADDR" not in teacher["env"]
        assert "MASTER_PORT" not in teacher["env"]
        for name, value in {
            "--tensor-parallel-size": "1",
            "--data-parallel-size": "2",
            "--data-parallel-backend": "mp",
            "--nnodes": "1",
            "--node-rank": "0",
            "--master-addr": "127.0.0.1",
            "--data-parallel-address": "127.0.0.1",
            "--max-model-len": "3088",
        }.items():
            assert flag(teacher["argv"], name) == value

        # 用真实 torchrun 解析器确认命令行 6 进程覆盖平台 PET_NPROC_PER_NODE=8。
        monkeypatch.setenv("PET_NPROC_PER_NODE", "8")
        distributed = get_args_parser().parse_args(train["argv"])
        assert distributed.nnodes == "2"
        assert distributed.nproc_per_node == "6"
        assert distributed.node_rank == rank
        assert distributed.master_addr == master_addr
        assert str(distributed.master_port) == master_port
        assert distributed.rdzv_backend == "static"
        assert not distributed.standalone

        cfg = TrainConfig.resolve(distributed.training_script_args).flatten()
        configs.append(cfg)
        for name, value in {
            "speculator_type": method,
            "optimizer": "muon",
            "lr": 1e-4,
            "muon_lr": 2e-4,
            "scheduler_type": "cosine",
            "scheduler_warmup_ratio": 0.01,
            "fsdp_shard": False,
            "epochs": 3,
            "train_data_ratio": 0.98,
            "total_seq_len": 8192,
            "block_size": 7,
            "max_anchors": 2048,
            "num_layers": 5,
            "full_attention_indices": [0, 1, 2, 3, 4],
            "target_layer_ids": [1, 10, 19, 28, 37],
            "checkpoint_freq": 0.5,
        }.items():
            assert cfg[name] == value
        assert cfg["vllm_endpoint"] == f"http://127.0.0.1:{env['VLLM_PORT']}/v1"
        assert flag(train["argv"], "--hidden-states-path") == flag(
            teacher["argv"], "--hidden-states-path"
        )
        assert_teacher_stopped(teacher)
        log_dir = Path(cfg["log_dir"])
        assert (log_dir / f"train_node{rank}.log").exists()
        assert (log_dir / f"vllm_node{rank}.log").exists()

    assert configs[0]["save_path"] == configs[1]["save_path"]
    assert configs[0]["run_name"] == configs[1]["run_name"]
    assert configs[0]["log_dir"] == configs[1]["log_dir"]
    assert flag(read_capture(env, "vllm", 0)["argv"], "--hidden-states-path") != flag(
        read_capture(env, "vllm", 1)["argv"], "--hidden-states-path"
    )


@pytest.mark.parametrize("method", [*METHODS, "dflash", "dfly", "peagle", "eagle3"])
def test_single_node_recipes_use_full_attention(method):
    script = (
        REPO
        / "examples/train/qwen3_6_35b_a3b"
        / f"{method}_qwen3_6_35b_a3b_perfectblend_online_full.sh"
    ).read_text()
    assert 'WS="/inspire/sfs/project/inf-multimodal/public/wumengke"' in script
    assert "${ROOT" not in script
    assert "NODE_RANK" not in script
    assert "--nnodes" not in script
    assert '--standalone --nproc_per_node "$NUM_TRAIN_GPUS"' in script
    # 只执行配置赋值并展开训练参数，不运行 mkdir、服务或训练命令。
    config = script.split("\nTRAIN_SCRIPT=", 1)[0]
    run_paths = "\n".join(
        line
        for line in script.splitlines()
        if line.startswith(("RUN_NAME=", "SAVE_DIR="))
    )
    flags = script[script.index("    --verifier-name-or-path") :].split(
        '\necho "Done.', 1
    )[0]
    capture = shlex.join(
        [sys.executable, "-c", "import json, sys; print(json.dumps(sys.argv[1:]))"]
    )
    result = subprocess.run(  # noqa: S603
        ["/bin/bash", "-c", f"{config}\n{run_paths}\n{capture} \\\n{flags}"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    argv = json.loads(result.stdout)
    cfg = TrainConfig.resolve(argv).flatten()
    assert cfg["speculator_type"] == method
    assert cfg["num_layers"] == (1 if method == "eagle3" else 5)
    assert cfg["full_attention_indices"] == list(range(cfg["num_layers"]))
    assert cfg["target_layer_ids"] == [1, 10, 19, 28, 37]
    assert cfg["scheduler_type"] == "cosine"
    assert cfg["lr"] == 1e-4
    assert cfg["muon_lr"] == 2e-4
    assert "--sliding-window" not in argv
    assert "--sliding-window-non-causal" not in argv


def test_training_failure_stops_local_teacher(launch_env):
    launch_env["NNODE_TEST_TRAIN_EXIT"] = "23"
    [(output, returncode)] = run_nodes("domino", launch_env, (0,))
    assert returncode == 23, output
    assert_teacher_stopped(read_capture(launch_env, "vllm", 0))


def test_teacher_startup_failure_does_not_start_training(launch_env):
    launch_env["NNODE_TEST_VLLM_FAIL"] = "1"
    [(output, returncode)] = run_nodes("domino", launch_env, (0,))
    assert returncode != 0
    assert "本机 vLLM 在就绪前退出" in output
    assert not list(Path(launch_env["NNODE_TEST_CAPTURE"]).iterdir())
    assert not (Path(launch_env["REPO"]) / "tmp").exists()


def test_dflash2_rejects_pruned_vocab_before_launch(launch_env):
    mapping = Path(launch_env["DATA_DIR"]) / "d2t.npy"
    mapping.touch()
    [(output, returncode)] = run_nodes("dflash2", launch_env, (0,))
    assert returncode != 0
    assert "DFlash2 不支持裁剪词表" in output
    assert mapping.exists()
    assert not list(Path(launch_env["NNODE_TEST_CAPTURE"]).iterdir())
