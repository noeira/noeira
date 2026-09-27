#!/bin/bash
# Promote a pixel DAgger student run to a policy ROLE of so101-tower, so the
# real-arm deploy (examples/so101/pixel_student_deploy_real.mojo) takes it as
# `--role <role>`.
#
#   tools/project/promote_pixel_student.sh RUN_ID [ROLE] [NOTE]
#   DEFINES="-D DAGGER_PX_32" tools/project/promote_pixel_student.sh ...   # a 32x32 student
#
# 1. an older run kept its weights at <run>/student.ckpt: link them as
#    checkpoints/last.ckpt (the layout project-promote reads);
# 2. write checkpoints/norm.json — the student's manifest — from a build of
#    the run's shape (pixel_student_manifest.mojo refuses a mismatch);
# 3. project-promote <run> last --as <role>: policies/<role>.ckpt +
#    policies/<role>.norm.json + policies/<role>.kv (sha256, provenance).
set -euo pipefail
RID="${1:?usage: promote_pixel_student.sh RUN_ID [ROLE] [NOTE]}"
ROLE="${2:-pixel_lift}"
NOTE="${3:-pixel DAgger student}"
if [ -d "runs/$RID" ]; then
  RUN="runs/$RID"
else
  RUN=$(ls -d projects/*/runs/"$RID" 2>/dev/null | head -1)
fi
[ -n "${RUN:-}" ] && [ -f "$RUN/run.kv" ] || { echo "no run $RID under runs/ or projects/*/runs/" >&2; exit 1; }
mkdir -p "$RUN/checkpoints"
if [ ! -f "$RUN/checkpoints/last.ckpt" ]; then
  [ -f "$RUN/student.ckpt" ] || { echo "no $RUN/checkpoints/last.ckpt nor $RUN/student.ckpt" >&2; exit 1; }
  ln "$RUN/student.ckpt" "$RUN/checkpoints/last.ckpt" 2>/dev/null || cp "$RUN/student.ckpt" "$RUN/checkpoints/last.ckpt"
  echo "linked $RUN/student.ckpt -> checkpoints/last.ckpt"
fi
pixi run mojo run -I . ${DEFINES:-} tools/project/pixel_student_manifest.mojo "$RUN"
pixi run project-promote "$RID" last --as "$ROLE" --note "$NOTE"
