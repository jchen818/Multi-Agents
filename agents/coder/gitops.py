"""Everything the coding agent does with git and GitHub. Plain code: the model never chooses a ref, a remote or a URL.

GitHub does not enforce branch rules on a private repo on the free plan, and a fine-grained token cannot be limited
to some branches. So this module is the guard:
  * the remote is always https://github.com/<repo from config/coder.yaml>.git, the token rides in an HTTP header
    passed through the environment (never in a URL, argv or .git/config);
  * it only ever pushes refs/heads/agent/<slug> to the same name, never with --force;
  * it reads main's commit on GitHub before and after every push; if main moved, it writes state/coder/ALERT and stops;
  * pull requests go to the base branch and are merged by a person.
"""
import base64
import getpass
import json
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("MULTI_AGENTS_ROOT") or Path(__file__).resolve().parents[2])
STATE = ROOT / "state" / "coder"
SECRETS = ROOT / "secrets" / "github.yaml"
API = "https://api.github.com"
BRANCH_RE = re.compile(r"agent/[a-z0-9][a-z0-9-]{2,70}")
AUTHOR = ("Multi-Agents coder", "coder@multi-agents.invalid")


class PushRefused(Exception):
    pass


def load_token(path: Path = SECRETS) -> str:
    if not path.exists():
        raise SystemExit(f"no {path}: create it with 'repo:' and 'token:' (fine-grained, this repo only)")
    if path.stat().st_mode & 0o077:
        raise SystemExit(f"{path} is readable by others: chmod 600 {path}")
    return yaml.safe_load(path.read_text())["token"].strip()


def slug(task: str, now=None) -> str:
    words = re.findall(r"[a-z0-9]+", task.lower())[:6] or ["task"]
    stamp = time.strftime("%Y%m%d-%H%M", time.localtime(now))
    return f"{'-'.join(words)[:50].strip('-')}-{stamp}"


class GitHub:
    """The few REST calls we need. `opener` is replaceable in tests."""

    def __init__(self, repo: str, token: str, opener=None):
        self.repo, self.token = repo, token
        self.opener = opener or urllib.request.urlopen

    def call(self, method: str, path: str, body=None) -> dict:
        req = urllib.request.Request(f"{API}/repos/{self.repo}{path}", method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Authorization": f"Bearer {self.token}",
                                              "Accept": "application/vnd.github+json",
                                              "X-GitHub-Api-Version": "2022-11-28",
                                              "User-Agent": "multi-agents-coder"})
        with self.opener(req, timeout=30) as r:
            return json.loads(r.read() or b"{}")

    def branch_sha(self, branch: str) -> str:
        return self.call("GET", f"/git/ref/heads/{branch}")["object"]["sha"]

    def open_pr(self, head: str, base: str, title: str, body: str, draft: bool) -> dict:
        return self.call("POST", "/pulls", {"title": title, "head": head, "base": base, "body": body, "draft": draft})


class Repo:
    """A clone of the configured repo under state/coder/repo, with one worktree per task."""

    def __init__(self, cfg: dict, token: str, github: GitHub = None, home: Path = STATE, url: str = None):
        self.cfg, self.home = cfg, Path(home)
        self.repo = cfg["repo"]
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", self.repo):
            raise SystemExit(f"bad repo name {self.repo!r}")
        self.url = url or f"https://github.com/{self.repo}.git"
        self.clone = self.home / "repo"
        self.github = github or GitHub(self.repo, token)
        hdr = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        self.env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_COUNT": "1",
                    "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
                    "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {hdr}"}

    def git(self, *args, cwd=None, check=True) -> str:
        r = subprocess.run(["git", "-C", str(cwd or self.clone), *args], env=self.env, capture_output=True,
                           text=True, timeout=300)
        if check and r.returncode:
            raise RuntimeError(f"git {args[0]} failed: {r.stderr.strip()[:500]}")
        return r.stdout.strip()

    def sync(self) -> str:
        """Fresh copy of the base branch. Returns its commit."""
        base = self.cfg.get("base", "main")
        if not (self.clone / ".git").exists():
            self.home.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "-q", "--no-tags", self.url, str(self.clone)], env=self.env, check=True,
                           capture_output=True, timeout=600)
        self.git("remote", "set-url", "origin", self.url)
        self.git("fetch", "-q", "--prune", "origin", f"+refs/heads/{base}:refs/remotes/origin/{base}")
        self.git("worktree", "prune")
        return self.git("rev-parse", f"origin/{base}")

    def new_worktree(self, task: str) -> tuple:
        """(branch, path): a new agent/<slug> branch at origin/<base>, checked out in its own directory."""
        name = slug(task)
        branch = f"{self.cfg.get('branch_prefix', 'agent/')}{name}"
        check_branch(branch)
        path = self.home / "work" / name
        self.git("worktree", "add", "-q", "-b", branch, str(path), f"origin/{self.cfg.get('base', 'main')}")
        return branch, path

    def drop(self, path: Path, branch: str):
        """Remove a local worktree and its local branch (nothing on GitHub changes)."""
        check_branch(branch)
        self.git("worktree", "remove", "--force", str(path), check=False)
        self.git("branch", "-D", branch, check=False)

    def commit(self, path: Path, message: str) -> str:
        self.git("add", "-A", cwd=path)
        if not self.git("status", "--porcelain", cwd=path):
            raise PushRefused("nothing to commit")
        who = getpass.getuser()
        self.git("-c", f"user.name={AUTHOR[0]}", "-c", f"user.email={AUTHOR[1]}", "commit", "-q", "-m",
                 f"{message.strip()}\n\nRequested-by: {who}", cwd=path)
        return self.git("rev-parse", "HEAD", cwd=path)

    def push(self, path: Path, branch: str) -> str:
        """Push the task branch (and only it). Verifies main on GitHub did not move."""
        check_branch(branch)
        base = self.cfg.get("base", "main")
        if branch == base:
            raise PushRefused("refusing to push the base branch")
        head = self.git("rev-parse", "HEAD", cwd=path)
        before = self.github.branch_sha(base)
        self.git("push", "-q", "origin", f"refs/heads/{branch}:refs/heads/{branch}", cwd=path)
        after = self.github.branch_sha(base)
        if after != before:
            alert(f"{base} on GitHub moved from {before} to {after} during the push of {branch}. "
                  f"The agent only pushes {branch}; check who changed {base}.", self.home)
            raise PushRefused(f"{base} moved during the push; stopped (see state/coder/ALERT)")
        remote = self.github.branch_sha(branch)
        if remote != head:
            raise PushRefused(f"{branch} on GitHub is {remote}, expected {head}")
        return head


def check_branch(branch: str):
    if not BRANCH_RE.fullmatch(branch or ""):
        raise PushRefused(f"branch {branch!r} is not agent/<slug>; the agent pushes nothing else")


def alert(text: str, home: Path = STATE):
    home.mkdir(parents=True, exist_ok=True)
    with open(home / "ALERT", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}\n")
    print(f"\n!!! ALERT: {text}\n")
