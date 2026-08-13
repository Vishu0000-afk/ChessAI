import os
import sys
import tempfile

PROJECT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT, "ChessAI"))

import torch
from configs.config import SelfPlayConfig, resolve_device
from src.selfplay.coordinator import SelfPlayCoordinator

tmp_root = tempfile.mkdtemp(prefix="chessai_sweep_")


def run(workers, concurrency, inference_batch, games=500):
    tmp = os.path.join(tmp_root, f"w{workers}_c{concurrency}_ib{inference_batch}")
    cfg = SelfPlayConfig(
        game_mode="benchmark",
        games_total=games,
        num_workers=workers,
        self_play_concurrency=concurrency,
        temperature=1.0,
        temp_final=0.3,
        temp_decay_games=1000000,
        max_game_moves=120,
        replay_buffer_size=100_000,
        train_enabled=False,
        evaluate_enabled=False,
        use_mixed_precision=True,
        nn_conv_channels=64,
        nn_res_blocks=2,
        checkpoint_dir=os.path.join(tmp, "checkpoints"),
        checkpoint_every_n_games=0,
        dataset_dir=os.path.join(tmp, "data"),
        dataset_chunk_size=100_000,
        persist_every_n_games=0,
        inference_max_batch=inference_batch,
        device=resolve_device("cuda"),
        auto_resume=False,
        seed=0,
    )
    coordinator = SelfPlayCoordinator(cfg)
    stats = coordinator.run()
    s = stats.snapshot()
    vram = torch.cuda.max_memory_allocated() / 1e9
    line = (
        f"w={workers:2d} c={concurrency:2d} ib={inference_batch:4d} | "
        f"games={s['games']:4d} games/s={s['games_per_sec']:6.1f} "
        f"pos/s={s['samples_per_sec']:8.0f} vram={vram:5.2f}GB"
    )
    print(line, flush=True)
    return s["games_per_sec"]


def main():
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), flush=True)
    print("== Sweep A: workers=8, concurrency x inference_batch ==", flush=True)
    best = (0, None)
    for ib in (512, 1024, 2048):
        for c in (4, 8, 16, 32):
            gps = run(8, c, ib)
            if gps > best[0]:
                best = (gps, (8, c, ib))
    print("best A:", best, flush=True)

    w, c, ib = best[1]
    print("== Sweep B: workers around best ==", flush=True)
    for w_ in (4, 16):
        gps = run(w_, c, ib)
        if gps > best[0]:
            best = (gps, (w_, c, ib))
    print("BEST:", best, flush=True)


if __name__ == "__main__":
    main()