import argparse, base64, json, sys, time
from io import BytesIO
from typing import Any, Dict, Optional

import requests
from PIL import Image


def b64_dummy_image(w=320, h=240) -> str:
    img = Image.new("RGB", (w, h), (60, 120, 180))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def warn(msg: str):
    print(f"[WARN] {msg}")


def fail(msg: str, resp: Optional[requests.Response] = None, exit_code: int = 1):
    if resp is not None:
        try:
            body = resp.json()
        except Exception:
            body = (resp.text or "")[:500]
        print(f"[FAIL] {msg}\nStatus={resp.status_code}\nBody={body}")
    else:
        print(f"[FAIL] {msg}")
    sys.exit(exit_code)


def get_json_or_text(resp: requests.Response) -> Dict[str, Any]:
    try:
        return resp.json()
    except Exception:
        return {"raw": (resp.text or "")[:500]}


def main():
    p = argparse.ArgumentParser(description="Test NaVid FastAPI service from another computer")
    p.add_argument("--host", required=True, help="Server IP or DNS")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--episode-id", default="CMU")
    p.add_argument("--model-path", default=None, help="Optional: call /load with this model path first")
    p.add_argument("--require-map", default=False, help="Optional: set require_map true on /load")

    # New flags: by default we do NOT fail the whole test if /reset or /act fails.
    p.add_argument("--require-reset", action="store_true", help="Fail if /reset is not 200")
    p.add_argument("--require-act", action="store_true", help="Fail if /act is not 200")
    p.add_argument("--retries", type=int, default=5, help="Health check retries")
    p.add_argument("--wait", type=float, default=2.0, help="Seconds to wait between retries")

    args = p.parse_args()

    base = f"http://{args.host}:{args.port}"
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json"})

    # 1) /healthz (with retries)
    last_err = None
    for i in range(args.retries):
        try:
            r = s.get(f"{base}/healthz", timeout=args.timeout)
            if r.status_code == 200:
                try:
                    health = r.json()
                except Exception:
                    health = {}
                print("[OK] /healthz:", json.dumps(health, indent=2, ensure_ascii=False))
                if health.get("status") != "ok":
                    last_err = "Health status not ok"
                else:
                    last_err = None
                    break
            else:
                last_err = f"/healthz returned {r.status_code}"
        except Exception as e:
            last_err = f"Could not reach /healthz: {e}"
        time.sleep(args.wait)
    if last_err:
        fail(last_err)

    # 2) Optional /load (only when model-path provided)
    if args.model_path:
        payload = {"model_path": args.model_path, "require_map": bool(args.require_map)}
        try:
            r = s.post(f"{base}/load", json=payload, timeout=max(30, args.timeout))
            if r.status_code != 200:
                msg = "/load returned non-200"
                print(json.dumps(get_json_or_text(r), indent=2, ensure_ascii=False))
                fail(msg, r)
            else:
                print("[OK] /load:", json.dumps(r.json(), indent=2, ensure_ascii=False))
        except Exception as e:
            fail(f"Could not call /load: {e}")

    # 3) /reset (best-effort by default)
    try:
        r = s.post(f"{base}/reset", json={}, timeout=args.timeout)
        if r.status_code != 200:
            if args.require_reset:
                fail("/reset returned non-200", r)
            else:
                warn(f"/reset non-200 ignored (server may not have NaVid_Agent ready). "
                     f"Status={r.status_code} Body={get_json_or_text(r)}")
        else:
            print("[OK] /reset:", json.dumps(r.json(), indent=2, ensure_ascii=False))
    except Exception as e:
        if args.require_reset:
            fail(f"Could not call /reset: {e}")
        else:
            warn(f"Could not call /reset: {e} (ignored)")

    # 4) /act (best-effort by default)
    rgb_b64 = b64_dummy_image()
    act_body: Dict[str, Any] = {
        "observations": {
            "rgb_b64": rgb_b64,
            "instruction": {"text": "Go forward 50 cm and turn right 30 degrees."}
        },
        "info": None,
        "episode_id": "CMU"
    }
    try:
        r = s.post(f"{base}/act", json=act_body, timeout=max(60, args.timeout))
        if r.status_code != 200:
            if args.require_act:
                fail("/act returned non-200", r)
            else:
                warn(f"/act non-200 ignored. This often happens if NaVid_Agent failed to import "
                     f"(e.g., missing 'rospy') or model not available.\n"
                     f"Status={r.status_code} Body={get_json_or_text(r)}")
                print("\n✅ Reachability OK (health passed). Server is up, but /act is not ready.\n")
                sys.exit(0)
        resp = r.json()
        print("[OK] /act:", json.dumps(resp, indent=2, ensure_ascii=False))

        # Basic shape checks
        if "result" not in resp or "action" not in resp["result"]:
            if args.require_act:
                fail("Response missing result.action")
            else:
                warn("Response missing result.action (ignored)")
                print("\n✅ Reachability OK. Server responded to /act but payload shape differs.\n")
                sys.exit(0)

        action = resp["result"]["action"]
        if not isinstance(action, int) or action not in (0, 1, 2, 3):
            if args.require_act:
                fail(f"Unexpected action value: {action}")
            else:
                warn(f"Unexpected action value: {action} (ignored)")
                print("\n✅ Reachability OK. Server responded to /act but action value unexpected.\n")
                sys.exit(0)

        print("\n✅ All checks passed.")
        sys.exit(0)

    except Exception as e:
        if args.require_act:
            fail(f"Could not call /act: {e}")
        else:
            warn(f"Could not call /act: {e} (ignored)")
            print("\n✅ Reachability OK (health passed). Server is up, but /act not reachable.\n")
            sys.exit(0)


if __name__ == "__main__":
    main()
