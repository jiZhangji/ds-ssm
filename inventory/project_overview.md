# Project Overview

## Server Summary

| Server | Host | Branch | HEAD | Dirty lines | Experiments | Weights | Logs/results | Py | YAML | output_dir size |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `server_1080ti` | `172.26.193.188` | `(none)` | `b26cbda230df` | `43` | `5` | `54` | `86` | `176` | `133` | `1.4G	output_dir` |
| `server_3090_a` | `172.26.150.36` | `main` | `b26cbda230df` | `34` | `6` | `68` | `76` | `179` | `116` | `1.2G	output_dir` |
| `server_3090_b` | `172.26.133.16` | `main` | `b26cbda230df` | `38` | `6` | `86` | `99` | `179` | `130` | `1.6G	output_dir` |

## High-Level Findings

- The committed Git base is identical across all three servers.
- The meaningful research state is in uncommitted tracked modifications and untracked files.
- `output_dir` is non-empty on all servers; 3090-A and 3090-B have six top-level output directories, 1080Ti has five.
- Weight artifacts are numerous, especially on 3090-B; no weights or experiment outputs were downloaded.
- The codebase includes SAR/SSDD dataset prep tools, custom Mamba-YOLO model YAMLs, DS-SSM/PFIM experiment modules, and modified training/model task logic.
