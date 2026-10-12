#!/usr/bin/env python3
"""把 TextWorld 自带的 libdownward.so 的 GLIBC 需求降到 glibc 2.17。

## 问题

`fast_downward_textworld==20.6.4`（TextWorld 的 PDDL 规划器）里的 `libdownward.so`
是在 glibc >= 2.32 的系统上编译的。ALFWorld 的 `.tw-pddl` 游戏走的是
`textworld.envs.pddl.Env`，其 `__init__` 第一件事就是 `fast_downward.load_lib()`，
于是直接死在：

    OSError: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.32' not found
             (required by .../fast_downward/libdownward.so)

实测：

    二期 A800 集群（登录节点 mgmt-3 / 计算节点 gpu3-9）：Ubuntu 20.04，glibc 2.31
    三期 ACD 集群（ACD-Login-1 / ACD1-63）        ：Rocky 8，    glibc 2.28

两边都跑不起来，**不是登录节点的问题**，计算节点一样。

## 做法（完全不改动任何外部环境）

只在自己的目录里造一份副本，然后在 Python 侧把 `fast_downward.interface.DOWNWARD_LIB_PATH`
指向副本即可（`load_lib()` 每次调用都会读这个模块全局变量）。

1. 复制 `libdownward.so`；
2. 把 `.gnu.version_r` 里对 `libc.so.6` 的 `GLIBC_2.32` 需求改成 `GLIBC_2.17`：
   - 原地改写 `.dynstr` 里的版本名字符串（两个字符串等长，偏移不动）；
   - **同时**把对应 Vernaux 条目的 `vna_hash` 改成 `elf_hash("GLIBC_2.17")`。
     只改字符串不改 hash 是没用的：glibc 的 `match_symbol()` 先比 `vd_hash`
     再看名字，hash 不匹配一律报 `version 'GLIBC_2.17' not found`（踩过）。
3. libdownward.so 是 C++ 编的，还用到 `__libc_single_threaded`（glibc 2.32 才有的变量）。
   我们编一个一行的垫片 `libglibc_single_threaded_shim.so` 提供它，
   在 `import fast_downward` 之前用 `ctypes.CDLL(..., mode=RTLD_GLOBAL)` 预加载，
   这样后面 `dlopen(libdownward.so)` 就能从全局域解析到这个符号——
   **不需要 LD_PRELOAD**（LD_PRELOAD 也没用，它救不了 "version not found"，
   因为版本需求是记在具体的 libc.so.6 头上的）。
4. 剩下的 `GLIBCXX_3.4.29` 由 conda 环境自带的 `$CONDA_PREFIX/lib/libstdc++.so.6` 提供，
   运行时把 `$CONDA_PREFIX/lib` 放进 `LD_LIBRARY_PATH` 即可（见 `bootstrap_alfworld_env.sh`）。

垫片里的变量取 **0**（"可能有线程"）而不是 1，走保守加锁路径，避免任何竞态风险；
Fast Downward 本身是单线程的，两种取值都能跑，取 0 只是更稳。

## 用法

    python patch_libdownward.py --src <原 libdownward.so> --out-dir <输出目录>
    # 默认 src = $CONDA_PREFIX(agentenv-alfworld)/lib/python3.9/site-packages/fast_downward/libdownward.so

输出：

    <out-dir>/libdownward.so                          已打补丁
    <out-dir>/libglibc_single_threaded_shim.so        + .c 源码
    <out-dir>/PATCH_INFO.txt                          记录来源与校验和
"""
from __future__ import annotations

import argparse
import hashlib
import os
import struct
import subprocess
import sys

# ---------------------------------------------------------------- ELF helpers

ELF_HASH_MASK = 0xFFFFFFFF


def elf_hash(name: bytes) -> int:
    """标准 ELF 字符串 hash（与 libc 的 verdef 表一致）。"""
    h = 0
    for b in name:
        h = ((h << 4) + b) & ELF_HASH_MASK
        g = h & 0xF0000000
        if g:
            h ^= g >> 24
        h &= ~g & ELF_HASH_MASK
    return h & ELF_HASH_MASK


def read_sections(data: bytearray) -> dict:
    """返回 {section_name: {offset, size, link}}，只支持 ELF64 小端。"""
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise SystemExit("只支持 ELF64 小端（x86_64）的 .so")
    (e_shoff,) = struct.unpack_from("<Q", data, 0x28)
    (e_shentsize,) = struct.unpack_from("<H", data, 0x3A)
    (e_shnum,) = struct.unpack_from("<H", data, 0x3C)
    (e_shstrndx,) = struct.unpack_from("<H", data, 0x3E)

    def sh(i):
        off = e_shoff + i * e_shentsize
        name, typ, flags, addr, offset, size, link, info, align, entsize = struct.unpack_from(
            "<IIQQQQIIQQ", data, off
        )
        return dict(name=name, type=typ, offset=offset, size=size, link=link)

    shstr = sh(e_shstrndx)
    out = {}
    for i in range(e_shnum):
        s = sh(i)
        p = shstr["offset"] + s["name"]
        end = data.index(b"\0", p)
        out[data[p:end].decode()] = s
    return out


def dynstr_at(data: bytearray, sections: dict, str_off: int) -> bytes:
    base = sections[".dynstr"]["offset"] + str_off
    end = data.index(b"\0", base)
    return bytes(data[base:end])


def patch_version_need(data: bytearray, sections: dict, old: bytes, new: bytes) -> int:
    """把 libc.so.6 的 verneed 版本需求 old -> new，同步修正 vna_hash。返回改动条数。"""
    if len(old) != len(new):
        raise SystemExit("为保持 .dynstr 偏移不变，新旧版本名必须等长")
    vn = sections[".gnu.version_r"]
    base = vn["offset"]
    end = base + vn["size"]
    changed = 0

    voff = 0
    while voff < vn["size"]:
        (_vn_version, vn_cnt, _vn_file, vn_aux, vn_next) = struct.unpack_from("<HHIII", data, base + voff)
        aux = base + voff + vn_aux
        for _ in range(vn_cnt):
            vna_hash, _vna_flags, _vna_other, vna_name, vna_next = struct.unpack_from("<IHHII", data, aux)
            cur = dynstr_at(data, sections, vna_name)
            if cur == old:
                # 1) 改名字符串（原地，等长）
                so = sections[".dynstr"]["offset"] + vna_name
                data[so:so + len(new)] = new
                # 2) 改 hash —— 不改的话 glibc 仍然报 version not found
                struct.pack_into("<I", data, aux, elf_hash(new))
                changed += 1
            if vna_next == 0:
                break
            aux += vna_next
        if vn_next == 0:
            break
        voff += vn_next
    return changed


# ---------------------------------------------------------------- shim

SHIM_SRC = """\
/* 为 glibc < 2.32 提供 __libc_single_threaded。
   取 0 = "可能有线程"，走保守加锁路径；Fast Downward 单线程，跑得动且更安全。
   在 dlopen(libdownward.so) 之前用 ctypes.CDLL(..., RTLD_GLOBAL) 预加载本库。 */
char __libc_single_threaded = 0;
"""


def build_shim(out_dir: str) -> str:
    c_path = os.path.join(out_dir, "libglibc_single_threaded_shim.c")
    so_path = os.path.join(out_dir, "libglibc_single_threaded_shim.so")
    with open(c_path, "w") as fh:
        fh.write(SHIM_SRC)
    subprocess.run(
        ["gcc", "-shared", "-fPIC", "-O2", "-o", so_path, c_path],
        check=True,
    )
    return so_path


# ---------------------------------------------------------------- main


def default_src() -> str:
    for env in ("CONDA_PREFIX",):
        p = os.environ.get(env)
        if p:
            p = os.path.join(p, "lib", "python3.9", "site-packages", "fast_downward", "libdownward.so")
            if os.path.isfile(p):
                return p
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=default_src(), help="原 libdownward.so（默认取 $CONDA_PREFIX）")
    ap.add_argument("--out-dir", required=True, help="输出目录（会在里面造 patched 副本）")
    ap.add_argument("--old-version", default="GLIBC_2.32")
    ap.add_argument("--new-version", default="GLIBC_2.17")
    args = ap.parse_args()

    if not args.src or not os.path.isfile(args.src):
        return int(bool(sys.stderr.write(f"找不到源文件：{args.src!r}，请用 --src 指定\n"))) or 1

    os.makedirs(args.out_dir, exist_ok=True)
    dst = os.path.join(args.out_dir, "libdownward.so")

    data = bytearray(open(args.src, "rb").read())
    sha_src = hashlib.sha256(data).hexdigest()
    sections = read_sections(data)

    needed = {".gnu.version_r", ".dynstr"}
    missing = needed - set(sections)
    if missing:
        raise SystemExit(f"缺少 section：{missing}")

    n = patch_version_need(data, sections, args.old_version.encode(), args.new_version.encode())
    if n == 0:
        raise SystemExit(f"没找到 {args.old_version} 的 verneed 条目，源文件可能已经不是这个版本")

    with open(dst, "wb") as fh:
        fh.write(data)

    shim = build_shim(args.out_dir)

    info = [
        f"src            = {args.src}",
        f"sha256(src)    = {sha_src}",
        f"dst            = {dst}",
        f"sha256(dst)    = {hashlib.sha256(bytes(data)).hexdigest()}",
        f"patched entries= {n}  ({args.old_version} -> {args.new_version})",
        f"shim           = {shim}",
    ]
    with open(os.path.join(args.out_dir, "PATCH_INFO.txt"), "w") as fh:
        fh.write("\n".join(info) + "\n")
    print("\n".join(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
