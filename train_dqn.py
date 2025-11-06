# train_dqn.py
import os
import json
import argparse
import importlib
import numpy as np

from agents.dqn_agent import DQNAgent, DQNConfig
from environments.environment import CityModel
from update_rules.update_rules import DefaultUpdateRules, DefaultUpdateRulesParameters


def _import_object(path: str):
    """
    'pkg.module:ClassName' -> the object
    """
    if ":" not in path:
        raise ValueError("Use form 'module.path:ClassName'")
    mod_path, obj_name = path.split(":", 1)
    mod = importlib.import_module(mod_path)
    return getattr(mod, obj_name)


def _resolve_agent_class(agent_path: str = None):
    """
    If user passes --agent pkg:Class use that, else default to agents.agent:CityPlanner
    """
    if agent_path:
        return _import_object(agent_path)
    try:
        mod = importlib.import_module("agents.agent")
        return getattr(mod, "CityPlanner")
    except Exception as e:
        raise RuntimeError(
            "Could not auto-detect an agent class. Pass one explicitly with "
            "--agent module.path:ClassName (e.g., --agent agents.agent:CityPlanner)"
        ) from e


def _load_update_rules(json_path: str) -> DefaultUpdateRules:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    fields = DefaultUpdateRulesParameters.__annotations__.keys()
    filtered = {k: raw[k] for k in fields if k in raw}

    coerced = {}
    for k in fields:
        if k not in filtered:
            continue
        v = filtered[k]
        ann = DefaultUpdateRulesParameters.__annotations__[k]
        try:
            if ann in (int, float):
                coerced[k] = ann(v)
            else:
                coerced[k] = v
        except Exception:
            coerced[k] = v

    params = DefaultUpdateRulesParameters(**coerced)
    rules = DefaultUpdateRules()
    rules.set_parameters(params)

    loaded = list(coerced.keys())
    ignored = [k for k in raw.keys() if k not in loaded]
    print(f"[RULES] Loaded {len(loaded)} fields from JSON. Ignored unknown keys: {ignored}")
    return rules


def _resolve_hooks(model):
    apply_action = getattr(model, "apply_action", None) or getattr(model, "_wrapped_step", None)
    advance      = getattr(model, "advance", None)
    get_observe  = getattr(model, "get_observation", None) or getattr(model, "get_observe", None)
    reward_fn    = getattr(model, "compute_reward", None)
    done_fn      = getattr(model, "is_done", None)
    valid_fn     = getattr(model, "valid_actions", None)

    def _name(f): return getattr(f, "__name__", str(f)) if f else None
    print("[HOOKS] apply_action:", _name(apply_action))
    print("[HOOKS] advance     :", _name(advance))
    print("[HOOKS] get_observe :", _name(get_observe))
    print("[HOOKS] reward      :", _name(reward_fn))
    print("[HOOKS] is_done     :", _name(done_fn))
    print("[HOOKS] valid_actions:", _name(valid_fn))

    class Hooks: ...
    h = Hooks()
    h.apply_action  = apply_action
    h.advance       = advance
    h.get_observe   = get_observe
    h.reward        = reward_fn
    h.is_done       = done_fn
    h.valid_actions = valid_fn
    return h


def _obs_size(obs) -> int:
    arr = np.asarray(obs, dtype=np.float32)
    return int(arr.size)


def train(
    episodes: int,
    max_steps_per_ep: int,
    save_every: int,
    rules_json: str,
    out_dir: str,
    width: int,
    height: int,
    agent_path: str = None,
    seed: int = 42,
):
    os.makedirs(out_dir, exist_ok=True)

    rules = _load_update_rules(rules_json)
    agent_cls = _resolve_agent_class(agent_path)

    env = CityModel(agent_class=agent_cls, width=width, height=height, update_rules=rules, seed=seed)
    hooks = _resolve_hooks(env)

    obs0 = hooks.get_observe() if hooks.get_observe else env.get_observation()
    obs_size = _obs_size(obs0)

    if hooks.valid_actions:
        valid_list = list(hooks.valid_actions())
        n_actions = len(valid_list)
        valid_idx = np.array(valid_list, dtype=np.int64)
    else:
        n_actions = width * height * len(getattr(env, "ACTION_TILE_TYPES", [1]))
        valid_idx = None
        print(f"[WARN] Could not infer number of actions from valid_actions(); defaulting to {n_actions}.")

    print(f"[BOOT] grid={width}x{height}  obs_size={obs_size}  n_actions={n_actions}")

    dqn = DQNAgent(obs_size=obs_size, n_actions=n_actions, cfg=DQNConfig(), seed=seed, save_dir=out_dir)
    dqn.reset()

    global_step = 0
    for ep in range(1, episodes + 1):
        # Re-create env if you want clean slate per episode; for now, continue stateful build.
        # If reset desired, you could add a reset() to CityModel and reconstruct masks/state here.

        obs = hooks.get_observe() if hooks.get_observe else env.get_observation()
        ep_return = 0.0

        for t in range(1, max_steps_per_ep + 1):
            action = dqn.act(obs, valid_idx)
            if hooks.apply_action is None:
                raise RuntimeError("Model.step() takes no action; please implement model.apply_action(action).")
            hooks.apply_action(action)

            next_obs = hooks.get_observe() if hooks.get_observe else env.get_observation()
            reward = hooks.reward(action) if hooks.reward else 0.0
            done = bool(hooks.is_done()) if hooks.is_done else False

            dqn.learn((obs, action, reward, next_obs, done))
            obs = next_obs
            ep_return += float(reward)
            global_step += 1

            if done:
                break

        if ep % save_every == 0:
            path = dqn.save(tag=f"ep{ep:04d}")
            print(f"[SAVE] {path}")

        print(f"[EP {ep:04d}] steps={t:03d} return={ep_return:.3f}")

    final_path = dqn.save(tag="final")
    print(f"[DONE] Saved final model to {final_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--max-steps-per-ep", type=int, default=400)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--rules-json", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="data/outputs/dqn")
    p.add_argument("--width", type=int, default=20)
    p.add_argument("--height", type=int, default=20)
    p.add_argument("--agent", type=str, default=None, help="module.path:ClassName (e.g., agents.agent:CityPlanner)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        episodes=args.episodes,
        max_steps_per_ep=args.max_steps_per_ep,
        save_every=args.save_every,
        rules_json=args.rules_json,
        out_dir=args.out_dir,
        width=args.width,
        height=args.height,
        agent_path=args.agent,
        seed=args.seed,
    )
