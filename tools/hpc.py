#!/usr/bin/env python
"""HPC (HKUST-GZ hpc2login) remote control helper.

Usage:
  python hpc.py run "command"          # run a shell command on the login node
  python hpc.py put  <local> <remote>  # upload
  python hpc.py get  <remote> <local>  # download
  python hpc.py pushkey                # install local pubkey (password login, once)

Auth: prefers the local key ~/.ssh/hpc2gz_ed25519.
If HPC_PW env var is set, password/keyboard-interactive is used as fallback.
"""
import os
import sys
import pathlib

import paramiko

HOST = os.environ.get("HPC_HOST", "hpc2login.hpc.hkust-gz.edu.cn")
PORT = int(os.environ.get("HPC_PORT", "22"))
USER = os.environ.get("HPC_USER", "xyu608")
KEY = os.environ.get("HPC_KEY", str(pathlib.Path.home() / ".ssh" / "hpc2gz_ed25519"))
PUBKEY = KEY + ".pub"


def connect(force_password=False):
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kw = dict(hostname=HOST, port=PORT, username=USER, timeout=30,
              banner_timeout=60, auth_timeout=60, allow_agent=False,
              look_for_keys=False)
    pw = os.environ.get("HPC_PW")
    errors = []
    if not force_password and os.path.exists(KEY):
        try:
            cli.connect(key_filename=KEY, **kw)
            return cli, "key"
        except Exception as e:  # noqa: BLE001
            errors.append(f"key: {e}")
    if pw:
        try:
            cli.connect(password=pw, **kw)
            return cli, "password"
        except Exception as e:  # noqa: BLE001
            errors.append(f"password: {e}")
        try:
            tr = paramiko.Transport((HOST, PORT))
            tr.connect(username=USER, password=pw)
            cli._transport = tr
            return cli, "password-transport"
        except Exception as e:  # noqa: BLE001
            errors.append(f"transport: {e}")
    raise SystemExit("SSH auth failed:\n  " + "\n  ".join(errors))


def run(cli, cmd, timeout=900):
    stdin, stdout, stderr = cli.exec_command(cmd, timeout=timeout, get_pty=False)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    rc = stdout.channel.recv_exit_status()
    return rc, out, err


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    action = sys.argv[1]

    if action == "pushkey":
        with open(PUBKEY) as fh:
            pub = fh.read().strip()
        cli, how = connect(force_password=not os.environ.get("HPC_NO_PW"))
        print(f"[auth] {how}")
        cmd = (
            "umask 077; mkdir -p ~/.ssh; touch ~/.ssh/authorized_keys; "
            f"grep -qF '{pub}' ~/.ssh/authorized_keys || echo '{pub}' >> ~/.ssh/authorized_keys; "
            "chmod 600 ~/.ssh/authorized_keys ~/.ssh 2>/dev/null; "
            "echo INSTALLED; grep -c . ~/.ssh/authorized_keys"
        )
        rc, out, err = run(cli, cmd)
        print(out, err, f"rc={rc}")
        cli.close()
        return

    cli, how = connect()
    try:
        if action == "run":
            cmd = sys.argv[2] if len(sys.argv) > 2 else sys.stdin.read()
            rc, out, err = run(cli, cmd)
            sys.stdout.write(out)
            if err:
                sys.stderr.write("\n--- stderr ---\n" + err)
            print(f"\n[exit={rc}]")
            sys.exit(rc)
        elif action == "put":
            sftp = cli.open_sftp()
            local = os.path.normpath(os.path.abspath(sys.argv[2]))
            sftp.put(local, sys.argv[3])
            sftp.close()
            print("uploaded", sys.argv[3])
        elif action == "get":
            sftp = cli.open_sftp()
            local = os.path.normpath(os.path.abspath(sys.argv[3]))
            sftp.get(sys.argv[2], local)
            sftp.close()
            print("downloaded", local)
        else:
            raise SystemExit(__doc__)
    finally:
        cli.close()


if __name__ == "__main__":
    main()
