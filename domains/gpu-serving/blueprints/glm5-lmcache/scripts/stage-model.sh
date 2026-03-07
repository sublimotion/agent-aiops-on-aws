#!/usr/bin/env bash
# Downloads GLM-5-FP8 model weights to S3 file-by-file.
# S3 auto-syncs to FSx via data repository association.
#
# Usage: ./scripts/stage-model.sh <s3-bucket-name>

set -euo pipefail
BUCKET="${1:?Usage: $0 <s3-bucket-name>}"
MODEL_ID="${MODEL_ID:-zai-org/GLM-5-FP8}"
S3_KEY="${S3_KEY:-GLM-5-FP8}"
S3_PREFIX="s3://${BUCKET}/${S3_KEY}"
REGION="${AWS_DEFAULT_REGION:-us-east-2}"

echo "=== Staging ${MODEL_ID} to ${S3_PREFIX} ==="
echo "Region: ${REGION}"
echo ""

python3 - "$MODEL_ID" "$BUCKET" "$REGION" <<'PYEOF'
import sys, os, tempfile, subprocess, time
from huggingface_hub import HfApi

model_id = sys.argv[1]
bucket = sys.argv[2]
region = sys.argv[3]
s3_prefix = os.environ.get("S3_KEY", "GLM-5-FP8")

api = HfApi()
files = sorted(api.list_repo_files(model_id))

print(f"Checking existing files in s3://{bucket}/{s3_prefix}/...")
result = subprocess.run(
    ["aws", "s3", "ls", f"s3://{bucket}/{s3_prefix}/", "--recursive", "--region", region],
    capture_output=True, text=True
)
existing = set()
if result.returncode == 0:
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            parts = line.split()
            if len(parts) >= 4:
                key = parts[3].replace(f"{s3_prefix}/", "", 1)
                size = int(parts[2])
                existing.add((key, size))

print(f"Found {len(existing)} existing files in S3")

total_bytes = 0
uploaded = 0
skipped = 0

for i, fname in enumerate(files):
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = os.path.join(tmpdir, os.path.basename(fname))
        t0 = time.time()
        print(f"\n[{i+1}/{len(files)}] {fname}")
        downloaded = api.hf_hub_download(
            model_id, fname,
            local_dir=tmpdir,
            local_dir_use_symlinks=False
        )

        fsize = os.path.getsize(downloaded)
        if any(k == fname and s == fsize for k, s in existing):
            print(f"  SKIP (already in S3, {fsize/1e9:.2f} GB)")
            skipped += 1
            continue

        dl_time = time.time() - t0
        speed = fsize / dl_time / 1e6 if dl_time > 0 else 0
        print(f"  Downloaded {fsize/1e9:.2f} GB in {dl_time:.0f}s ({speed:.0f} MB/s)")

        t1 = time.time()
        s3_key = f"s3://{bucket}/{s3_prefix}/{fname}"
        subprocess.run(
            ["aws", "s3", "cp", downloaded, s3_key, "--region", region, "--no-progress"],
            check=True
        )
        ul_time = time.time() - t1
        speed = fsize / ul_time / 1e6 if ul_time > 0 else 0
        print(f"  Uploaded to S3 in {ul_time:.0f}s ({speed:.0f} MB/s)")

        total_bytes += fsize
        uploaded += 1

print(f"\n=== Complete ===")
print(f"Uploaded: {uploaded} files ({total_bytes/1e9:.1f} GB)")
print(f"Skipped: {skipped} files (already in S3)")
print(f"Total files: {len(files)}")
print(f"FSx DRA will auto-import to /models/{s3_prefix}/")
PYEOF

echo ""
echo "Model staged to ${S3_PREFIX}"
echo "FSx data repository association will auto-import to /models/${S3_KEY}/"
