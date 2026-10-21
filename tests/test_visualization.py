"""Tests for tetris_rl.visualization (read-only tooling around the DQN)."""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
import torch

from tetris_rl.agents.dqn_agent import DQNAgent, QNetwork, improved_dqn_config, original_dqn_config
from tetris_rl.agents.heuristic_agent import HeuristicAgent
from tetris_rl.engine import BOARD_HEIGHT, BOARD_WIDTH, PieceType
from tetris_rl.features import (
    board_metrics,
    placement_feature_batch,
    state_action_feature_size,
    state_action_features,
)
from tetris_rl.visualization import explorer as explorer_mod
from tetris_rl.visualization.analysis import (
    PlacementExplorer,
    board_to_str,
    load_dqn_agent,
    make_env,
    parse_board,
)
from tetris_rl.visualization.model_info import (
    SCALAR_FEATURES,
    describe_network,
    feature_groups,
    feature_names,
    forward_with_activations,
)


@pytest.fixture(scope="module")
def ckpt_ln(tmp_path_factory):
    """A small *trained-ish* checkpoint (random init + a few perturbations) with LayerNorm."""
    cfg = improved_dqn_config(seed=3, device="cpu")
    agent = DQNAgent(cfg)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():  # make LN affine params non-trivial so the trace test is meaningful
        agent.online.input_norm.weight.add_(0.3 * torch.randn(232, generator=g))
        agent.online.input_norm.bias.add_(0.1 * torch.randn(232, generator=g))
    p = tmp_path_factory.mktemp("ck") / "ln.pt"
    agent.save(p, include_buffer=False)
    return p


@pytest.fixture(scope="module")
def ckpt_plain(tmp_path_factory):
    agent = DQNAgent(original_dqn_config(seed=5, device="cpu"))
    p = tmp_path_factory.mktemp("ck") / "plain.pt"
    agent.save(p, include_buffer=False)
    return p


def _midgame_state(seed=11, pieces=15):
    from tetris_rl.environment import TetrisEnv

    env = TetrisEnv(seed=seed)
    env.reset(seed=seed)
    heur = HeuristicAgent(seed=seed)
    for _ in range(pieces):
        env.step(heur.select_action(env))
    e = env.engine
    return e.board.copy(), int(e.current_piece), int(e.next_piece)


# ------------------------------------------------------------------ feature layout


def test_feature_groups_cover_input_exactly():
    groups = feature_groups()
    assert groups[0].start == 0
    for a, b in zip(groups, groups[1:]):
        assert a.stop == b.start
    assert groups[-1].stop == state_action_feature_size() == 232
    assert len(feature_names()) == 232
    assert groups[-1].size == len(SCALAR_FEATURES)


def test_feature_layout_matches_features_py():
    """Each documented slice must contain what features.state_action_features puts there."""
    board, cur, nxt = _midgame_state()
    env = make_env(board, cur, nxt)
    eng = env.engine
    pl = env.legal_placements()[3]
    phi = state_action_features(eng, pl)
    new_board, lines, _ = eng.simulate_placement(pl)
    m, before = board_metrics(new_board), board_metrics(board)
    g = {x.key: x for x in feature_groups()}
    hw = BOARD_HEIGHT * BOARD_WIDTH

    np.testing.assert_array_equal(phi[g["board"].start : g["board"].stop], (new_board != 0).ravel())
    cur_oh = phi[g["current"].start : g["current"].stop]
    nxt_oh = phi[g["next"].start : g["next"].stop]
    assert cur_oh.argmax() == int(pl.piece) and cur_oh.sum() == 1
    assert nxt_oh.argmax() == int(eng.next_piece) and nxt_oh.sum() == 1
    np.testing.assert_allclose(
        phi[g["heights"].start : g["heights"].stop], m["heights"] / BOARD_HEIGHT, rtol=1e-6
    )
    sc = phi[g["scalars"].start : g["scalars"].stop]
    expected = [
        m["aggregate_height"] / hw,
        m["holes"] / hw,
        m["bumpiness"] / hw,
        m["max_height"] / BOARD_HEIGHT,
        lines / 4.0,
        (m["holes"] - before["holes"]) / hw,
        (m["aggregate_height"] - before["aggregate_height"]) / hw,
        (m["bumpiness"] - before["bumpiness"]) / hw,
    ]
    # NOTE: the scalar slots are ordered as in features.py; SCALAR_FEATURES documents the same order.
    np.testing.assert_allclose(sc, expected, rtol=1e-5, atol=1e-7)
    assert [k for k, _ in SCALAR_FEATURES][:5] == [
        "aggregate_height", "holes", "bumpiness", "max_height", "lines_cleared"
    ]


# ------------------------------------------------------------------ network description


def test_describe_network_layernorm():
    net = QNetwork(232, (256, 256, 128), use_layer_norm=True)
    info = describe_network(net)
    assert info.use_layer_norm and info.hidden_sizes == (256, 256, 128)
    assert [l.kind for l in info.layers] == [
        "input", "layernorm", "linear", "linear", "linear", "linear", "output"
    ]
    lin = [l for l in info.layers if l.kind == "linear"]
    assert [(l.in_dim, l.out_dim) for l in lin] == [(232, 256), (256, 256), (256, 128), (128, 1)]
    assert [l.activation for l in lin] == ["ReLU", "ReLU", "ReLU", None]
    assert info.total_params == sum(p.numel() for p in net.parameters()) == 158_929


def test_describe_network_no_layernorm():
    info = describe_network(QNetwork(232, (64, 32), use_layer_norm=False))
    assert not info.use_layer_norm
    assert [l.kind for l in info.layers] == ["input", "linear", "linear", "linear", "output"]


@pytest.mark.parametrize("ln", [True, False])
def test_forward_with_activations_matches_forward(ln):
    torch.manual_seed(0)
    net = QNetwork(232, (256, 256, 128), use_layer_norm=ln)
    x = np.random.default_rng(0).random((7, 232)).astype(np.float32)
    trace = forward_with_activations(net, x)
    with torch.no_grad():
        ref = net(torch.as_tensor(x)).numpy()
    np.testing.assert_allclose(trace[-1]["values"].reshape(-1), ref, atol=1e-5)
    assert trace[0]["kind"] == "input"
    assert (trace[1]["kind"] == "layernorm") == ln
    relu = [t for t in trace if t["kind"] == "act"]
    assert len(relu) == 3 and all((t["values"] >= 0).all() for t in relu)


# ------------------------------------------------------------------ loading


def test_load_dqn_agent_infers_architecture(ckpt_ln, ckpt_plain):
    a, meta = load_dqn_agent(ckpt_ln)
    assert describe_network(a.online).use_layer_norm and meta["config_matches_weights"]
    assert a.epsilon == 0.0
    b, _ = load_dqn_agent(ckpt_plain)
    assert not describe_network(b.online).use_layer_norm

    saved = torch.load(ckpt_ln, map_location="cpu", weights_only=False)["online"]
    for k, v in a.online.state_dict().items():  # trained weights, not fresh init
        torch.testing.assert_close(v, saved[k])


# ------------------------------------------------------------------ analysis backend


def test_analyze_matches_real_agents(ckpt_ln):
    ex = PlacementExplorer(ckpt_ln)
    board, cur, nxt = _midgame_state()
    out = ex.analyze(board_to_str(board), cur, nxt)

    env = make_env(board, cur, nxt)
    placements = env.legal_placements()
    assert out["n_legal"] == len(placements) == len(out["placements"]) > 5

    q = ex.agent.q_values(placement_feature_batch(env.engine, placements))
    np.testing.assert_allclose([p["q"] for p in out["placements"]], q, rtol=1e-6)
    assert out["dqn_choice"] == ex.agent.select_action(env, explore=False) == int(np.argmax(q))

    heur = HeuristicAgent(seed=0)
    scores = [heur.evaluate_placement(env, i) for i in range(len(placements))]
    np.testing.assert_allclose([p["heuristic_score"] for p in out["placements"]], scores)
    assert set(out["heuristic_choices"]) == set(np.flatnonzero(np.isclose(scores, max(scores))))
    assert out["agree"] == (out["dqn_choice"] in out["heuristic_choices"])

    for i, p in enumerate(out["placements"]):
        nb, lines, _ = env.engine.simulate_placement(placements[i])
        assert p["result_board"] == board_to_str(nb)
        assert p["lines_cleared"] == lines
        # all 4 piece cells survive unless they were on a cleared row
        assert 0 <= len(p["placed_after_clear"]) <= 4
        if lines == 0:
            assert len(p["placed_after_clear"]) == 4
        for r, c in p["placed_after_clear"]:
            assert nb[r, c] == int(placements[i].piece) + 1

    ranks = sorted(p["q_rank"] for p in out["placements"])
    assert ranks == list(range(1, len(ranks) + 1))
    assert out["placements"][out["dqn_choice"]]["q_rank"] == 1


def test_stage_activations_are_consistent(ckpt_ln):
    ex = PlacementExplorer(ckpt_ln)
    board, cur, nxt = _midgame_state()
    out = ex.analyze(board_to_str(board), cur, nxt)
    names = [s["name"] for s in out["stages"]]
    assert names == ["input", "layernorm", "hidden1", "hidden2", "hidden3", "output"]
    n = out["n_legal"]
    dims = [len(s["values"][0]) for s in out["stages"]]
    assert dims == [232, 232, 256, 256, 128, 1]
    assert all(len(s["values"]) == n for s in out["stages"])
    np.testing.assert_allclose(
        [v[0] for v in out["stages"][-1]["values"]], [p["q"] for p in out["placements"]], rtol=1e-6
    )
    # Q reconstructed from exported weights == PyTorch Q (what the browser recomputes)
    w = ex.weights()["layers"][-1]
    import base64

    W = np.frombuffer(base64.b64decode(w["W"]), dtype="<f4").reshape(w["out"], w["in"])
    b = np.frombuffer(base64.b64decode(w["b"]), dtype="<f4")
    h3 = np.array(out["stages"][4]["values"], dtype=np.float64)
    np.testing.assert_allclose((h3 @ W.T.astype(np.float64) + b)[:, 0],
                               [p["q"] for p in out["placements"]], atol=2e-3)


def test_game_over_board_has_no_placements(ckpt_ln):
    ex = PlacementExplorer(ckpt_ln)
    full = np.ones((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.int8)
    full[:, 0] = 0  # no complete rows, but stacked to the ceiling
    out = ex.analyze(board_to_str(full), int(PieceType.T), int(PieceType.I))
    assert out["n_legal"] == 0 and out["dqn_choice"] is None and out["game_over"]


def test_step_and_sample_and_disagreement(ckpt_ln):
    ex = PlacementExplorer(ckpt_ln)
    s = ex.sample_state("heuristic", seed=3, n_pieces=10)
    assert s["pieces_placed"] == 10 and len(s["board"]) == 200
    out = ex.analyze(s["board"], s["current"], s["next"])
    nxt = ex.step(s["board"], s["current"], s["next"], out["dqn_choice"], seed=0)
    env = make_env(parse_board(s["board"]), s["current"], s["next"])
    env.step(out["dqn_choice"])
    assert nxt["board"] == board_to_str(env.engine.board)
    assert nxt["current"] == s["next"]
    d = ex.next_disagreement(s["board"], s["current"], s["next"], seed=1, max_steps=80)
    if d["found"]:
        o = ex.analyze(d["state"]["board"], d["state"]["current"], d["state"]["next"])
        assert o["agree"] is False


def test_parse_board_validation():
    with pytest.raises(ValueError):
        parse_board("0" * 199)
    with pytest.raises(ValueError):
        parse_board("9" + "0" * 199)
    assert parse_board("0" * 200).shape == (BOARD_HEIGHT, BOARD_WIDTH)


# ------------------------------------------------------------------ HTTP layer


def test_model_path_must_stay_inside_models_dir():
    for bad in ("../../etc/passwd", "pyproject.toml", "/etc/hosts"):
        with pytest.raises(ValueError):
            explorer_mod._resolve_model(bad)


def test_http_api_smoke(ckpt_ln):
    ex = PlacementExplorer(ckpt_ln)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), explorer_mod.make_handler(explorer_mod._State(ex)))
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{port}"
    try:
        html = urllib.request.urlopen(base + "/").read().decode()
        assert "Tetris DQN Explorer" in html
        info = json.load(urllib.request.urlopen(base + "/api/info"))
        assert info["network"]["input_dim"] == 232 and len(info["feature_names"]) == 232
        wts = json.load(urllib.request.urlopen(base + "/api/weights"))
        assert [(l["in"], l["out"]) for l in wts["layers"]] == [(232, 256), (256, 256), (256, 128), (128, 1)]

        def post(path, body):
            req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(req))

        out = post("/api/analyze", {"board": "0" * 200, "current": 2, "next": 0})
        assert out["n_legal"] > 0 and out["dqn_choice"] is not None
        with pytest.raises(urllib.error.HTTPError) as e:
            post("/api/analyze", {"board": "x", "current": 2, "next": 0})
        assert e.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as e:
            post("/api/load", {"path": "../pyproject.toml"})
        assert e.value.code == 400
    finally:
        httpd.shutdown()
        httpd.server_close()


# ------------------------------------------------------------------ diagram


def test_architecture_diagram_exports_svg_and_png(tmp_path):
    from tetris_rl.visualization.architecture_diagram import draw_architecture

    info = describe_network(QNetwork(232, (256, 256, 128), use_layer_norm=True))
    svg, png = tmp_path / "a.svg", tmp_path / "a.png"
    draw_architecture(info, "test", svg, png, dpi=40)
    assert svg.read_text()[:200].lstrip().startswith(("<?xml", "<svg"))
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    # also renders without LayerNorm
    draw_architecture(describe_network(QNetwork(232, (64, 32))), None, tmp_path / "b.svg", None)
