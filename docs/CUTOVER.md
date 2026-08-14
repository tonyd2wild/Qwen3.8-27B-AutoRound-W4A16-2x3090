# Safe cutover and rollback

Use unique container names for the old and new lanes. Never remove the old container until the new model passes verification.

## Preserve an existing FP8 lane

Resolve the exact source container first:

```bash
docker inspect qwen38-fp8
docker update --restart=no qwen38-fp8
docker stop qwen38-fp8
docker rename qwen38-fp8 qwen38-fp8-220k-stable-backup
```

Then launch the AutoRound container:

```bash
./scripts/preflight.sh
./scripts/run.sh
./scripts/wait-ready.sh
python3 scripts/verify.py --benchmark-tokens 384
python3 scripts/benchmark.py --runs 3 --tokens 384
./scripts/status.sh
```

Do not run two containers on the same GPUs or host port at once.

## Roll back

If the AutoRound lane fails verification:

```bash
docker update --restart=no qwen38-autoround-int4
docker stop qwen38-autoround-int4
docker rename qwen38-autoround-int4 qwen38-autoround-int4-failed
docker rename qwen38-fp8-220k-stable-backup qwen38-fp8
docker update --restart=unless-stopped qwen38-fp8
docker start qwen38-fp8
```

Confirm health and model identity after either cutover direction. Adjust names if your `.env` uses different values.
