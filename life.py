# -----------------------
# CONFIG
# -----------------------
ROWS, COLS = 30, 30
CORE_H, CORE_W = 3, 3
HALO = 1
REGION_H = CORE_H + 2*HALO
REGION_W = CORE_W + 2*HALO
REGION_N = REGION_H * REGION_W

BIRTH_SET = {2, 5}
SURVIVE_SET = {2, 4, 5}

DEFAULT_SHOTS = 64  # ★256→64でさらに高速化（精度は十分）
UNITARY_ENTANGLE_DEPTH = 1

CLASSICAL_WEIGHT = 0.9
QUANTUM_WEIGHT = 1.0 - CLASSICAL_WEIGHT

MARKER_SIZE = 130
FIG_SIZE_PARAM = 10
CMAP = 'inferno'

# ★並列処理用のワーカー数
N_WORKERS = min(multiprocessing.cpu_count(), 8)

BACKEND = AerSimulator(
    method='statevector',
    max_parallel_threads=1,  # ★並列処理と併用するため1に
    max_parallel_shots=8,
)

# -----------------------
# パラメトリック回路（キャッシュ）
# -----------------------
_cached_circuit = None
_cached_params = None
_cached_transpiled = None

def get_or_build_parametric_circuit():
    """パラメトリック回路を1回だけ構築"""
    global _cached_circuit, _cached_params, _cached_transpiled

    if _cached_transpiled is not None:
        return _cached_transpiled, _cached_params

    print("Building parametric quantum circuit (once)...")

    params = [Parameter(f"θ{i}") for i in range(REGION_N)]
    qc = QuantumCircuit(REGION_N, REGION_N)

    # エンコード
    for i, p in enumerate(params):
        qc.ry(p, i)

    # 対称的エンタングルメント
    for layer in range(UNITARY_ENTANGLE_DEPTH):
        # 順方向
        for r in range(REGION_H):
            for c in range(REGION_W - 1):
                qc.cx(r*REGION_W + c, r*REGION_W + c + 1)
        for r in range(REGION_H - 1):
            for c in range(REGION_W):
                qc.cx(r*REGION_W + c, (r+1)*REGION_W + c)

        # 逆方向
        for r in range(REGION_H):
            for c in range(REGION_W - 1, 0, -1):
                qc.cx(r*REGION_W + c, r*REGION_W + c - 1)
        for r in range(REGION_H - 1, 0, -1):
            for c in range(REGION_W):
                qc.cx(r*REGION_W + c, (r-1)*REGION_W + c)
        
        # 斜め方向も追加
        for r in range(REGION_H-1):
            for c in range(REGION_W - 1, 0, -1):
                qc.cx(r*REGION_W + c, (r+1)*REGION_W + c - 1)
        for r in range(REGION_H - 1, 0, -1):
            for c in range(REGION_W - 1):
                qc.cx(r*REGION_W + c, (r-1)*REGION_W + c + 1)

    qc.measure(range(REGION_N), range(REGION_N))

    print("Transpiling... (optimization_level=3)")
    tqc = transpile(qc, BACKEND, optimization_level=3)

    _cached_circuit = qc
    _cached_params = params
    _cached_transpiled = tqc

    print("Done! Circuit is ready for reuse.")
    return tqc, params

# -----------------------
# Utilities
# -----------------------
def place_pattern(grid, pattern, top_left):
    ph, pw = np.array(pattern).shape
    r0, c0 = top_left
    for i in range(ph):
        for j in range(pw):
            if 0 <= r0 + i < grid.shape[0] and 0 <= c0 + j < grid.shape[1]:
                grid[r0 + i, c0 + j] = 1 if pattern[i][j] else 0

# -----------------------
# 量子サンプリング
# -----------------------
def sample_region_probs(region_grid, backend, shots=DEFAULT_SHOTS):
    """パラメトリック回路を使って高速実行"""
    tqc, params = get_or_build_parametric_circuit()

    # パラメータ値を計算
    flat = region_grid.flatten()
    param_vals = {}
    for i, p in enumerate(flat):
        if p <= 0:
            theta = 0.0
        elif p >= 1:
            theta = math.pi
        else:
            theta = 2 * math.asin(math.sqrt(float(p)))
        param_vals[params[i]] = theta

    # パラメータをバインド
    bound_qc = tqc.assign_parameters(param_vals, inplace=False)

    job = backend.run(bound_qc, shots=shots)
    result = job.result()
    counts = result.get_counts()

    # 確率を計算
    probs = np.zeros(REGION_N, dtype=float)
    total = sum(counts.values())
    if total == 0:
        return probs.reshape(REGION_H, REGION_W)

    for bitstr, cnt in counts.items():
        bs = bitstr[::-1]
        for i, ch in enumerate(bs):
            if ch == '1':
                probs[i] += cnt

    probs = probs / total
    return probs.reshape(REGION_H, REGION_W)

# -----------------------
# Classical rule
# -----------------------
def compute_next_core_from_probs(region_probs, threshold=0.5):
    rh, rw = region_probs.shape
    grid_bin = (region_probs > threshold).astype(int)
    next_core = np.zeros((CORE_H, CORE_W), dtype=int)

    for i in range(CORE_H):
        for j in range(CORE_W):
            rr = i + HALO
            cc = j + HALO
            count = 0
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,1),(1,-1)]:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < rh and 0 <= nc < rw:
                    count += grid_bin[nr, nc]

            cur = grid_bin[rr, cc]
            if cur == 1:
                nxt = 1 if (count in SURVIVE_SET) else 0
            else:
                nxt = 1 if (count in BIRTH_SET) else 0
            next_core[i, j] = nxt

    return next_core

def blend_next_probs(measured_region_probs, next_core_bits, blend_weights=(CLASSICAL_WEIGHT, QUANTUM_WEIGHT)):
    cw, qw = blend_weights
    next_probs = np.zeros((CORE_H, CORE_W), dtype=float)

    for i in range(CORE_H):
        for j in range(CORE_W):
            measured_p = float(measured_region_probs[HALO + i, HALO + j])
            classical_bit = int(next_core_bits[i, j])
            next_probs[i, j] = max(0.0, min(1.0, cw * classical_bit + qw * measured_p))

    return next_probs

# -----------------------
# ★並列処理用のチャンク処理関数
# -----------------------
def process_single_chunk(args):
    """単一チャンクを処理（並列実行用）"""
    top, left, grid_snapshot, shots, threshold = args
    rows, cols = grid_snapshot.shape
    
    r0 = max(0, top - HALO)
    c0 = max(0, left - HALO)
    
    region = np.zeros((REGION_H, REGION_W), dtype=float)
    for rr in range(REGION_H):
        for cc in range(REGION_W):
            gr = r0 + rr
            gc = c0 + cc
            if 0 <= gr < rows and 0 <= gc < cols:
                region[rr, cc] = float(grid_snapshot[gr, gc])
    
    measured_probs = sample_region_probs(region, BACKEND, shots=shots)
    next_core_bits = compute_next_core_from_probs(measured_probs, threshold=threshold)
    next_core_probs = blend_next_probs(measured_probs, next_core_bits)
    
    return (top, left, next_core_probs)

# -----------------------
# ★並列化されたグリッドステップ
# -----------------------
def step_full_grid_quantumified(global_prob_grid, backend, shots=DEFAULT_SHOTS, threshold=0.5):
    rows, cols = global_prob_grid.shape
    new_prob_grid = np.zeros_like(global_prob_grid)

    # スナップショット
    grid_snapshot = global_prob_grid.copy()

    # グリッド中心
    center_r = rows / 2.0
    center_c = cols / 2.0

    # チャンクリストを作成
    chunks = []
    for top in range(0, rows, CORE_H):
        for left in range(0, cols, CORE_W):
            if top + CORE_H <= rows and left + CORE_W <= cols:
                chunk_center_r = top + CORE_H / 2.0
                chunk_center_c = left + CORE_W / 2.0
                dist = (chunk_center_r - center_r)**2 + (chunk_center_c - center_c)**2
                chunks.append((dist, top, left))

    # 中心から外側へソート
    chunks.sort(key=lambda x: x[0])
    
    # ★並列処理用のタスクリスト
    tasks = [(top, left, grid_snapshot, shots, threshold) for (_, top, left) in chunks]
    
    # ★並列実行
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        results = list(executor.map(process_single_chunk, tasks))
    
    # 結果を書き戻し
    for top, left, next_core_probs in results:
        for i in range(CORE_H):
            for j in range(CORE_W):
                rr = top + i
                cc = left + j
                new_prob_grid[rr, cc] = next_core_probs[i, j]

    return new_prob_grid

# -----------------------
# 可視化
# -----------------------
def hex_coordinates(h, w):
    x = np.zeros(h*w)
    y = np.zeros(h*w)
    for j in range(h):
        x[w*j:w*(j + 1)] = np.arange(w) + 0.5 * j
        y[w*j:w*(j + 1)] = np.ones(w) * j * 0.866

    x = x - np.mean(x)
    y = y - np.mean(y)
    return x, y

def save_frame(i, prob_grid, size=FIG_SIZE_PARAM, marker_size=MARKER_SIZE, cmap=CMAP):
    H, W = prob_grid.shape
    x, y = hex_coordinates(H, W)

    fig = plt.figure(figsize=(size*1.5, size*0.866))

    x_margin = (x.max() - x.min()) * 0.1
    y_margin = (y.max() - y.min()) * 0.1
    plt.xlim([x.min() - x_margin, x.max() + x_margin])
    plt.ylim([y.max() + y_margin, y.min() - y_margin])

    plt.axis('off')
    plt.scatter(x, y, c=prob_grid.flatten(), marker='h', s=marker_size, cmap=cmap, vmin=0, vmax=1)
    filename = f"frame_{i:04d}_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
    plt.savefig(filename, dpi=120, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return filename

# -----------------------
# Main simulation
# -----------------------
def run_simulation_full(initial_pattern, position=(0,0), rows=ROWS, cols=COLS,
                        steps=30, shots=DEFAULT_SHOTS, threshold=0.5, out_gif=True):
    grid = np.zeros((rows, cols), dtype=float)
    place_pattern(grid, initial_pattern, position)

    # 事前に回路を構築
    get_or_build_parametric_circuit()

    frames = []
    outdir = "frames_qhex"
    os.makedirs(outdir, exist_ok=True)

    for t in range(steps):
        print(f"Step {t+1}/{steps} (parallel processing with {N_WORKERS} workers)")
        fname = save_frame(t, grid)
        dst = os.path.join(outdir, os.path.basename(fname))
        os.replace(fname, dst)
        frames.append(dst)

        grid = step_full_grid_quantumified(grid, BACKEND, shots=shots, threshold=threshold)

    fname = save_frame(steps, grid)
    dst = os.path.join(outdir, os.path.basename(fname))
    os.replace(fname, dst)
    frames.append(dst)

    gif_name = None
    if out_gif and frames:
        imgs = [Image.open(f) for f in frames]
        gif_name = datetime.now().strftime("quantum_hexlife_%Y%m%d_%H%M%S.gif")
        imgs[0].save(gif_name, save_all=True, append_images=imgs[1:], duration=120, loop=0)
        print("GIF saved:", gif_name)

    return frames, grid, gif_name

# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    pattern = [
        [0,0,0,0,0,0,1,0,0],
        [0,0,0,0,0,1,1,0,0],
        [0,0,1,1,0,0,0,1,1],
        [0,0,1,0,1,1,0,1,0],
        [0,0,0,1,0,1,0,0,0],
        [0,1,0,1,1,0,1,0,0],
        [1,1,0,0,0,1,1,0,0],
        [0,0,1,1,0,0,0,0,0],
        [0,0,1,0,0,0,0,0,0]
    ]

    pos = (10, 10)
    steps = 10
    shots = 64

    frames, final_grid, gif_name = run_simulation_full(
        pattern, position=pos, rows=ROWS, cols=COLS,
        steps=steps, shots=shots, threshold=0.5, out_gif=True
    )

    print("Done. Frames produced:", len(frames))
    if gif_name:
        print("GIF:", gif_name)
