#!/usr/bin/env bash
# Run from an activated SONIC environment. All outputs go to logs_eval/.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  printf 'Usage: bash %s {metrics|play|video} RUN_DIR [CHECKPOINT]\n' "$0" >&2
  exit 2
fi

mode="$1"
run_dir="$2"
checkpoint_name="${3:-model_step_018000.pt}"
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$repo_dir"
source_checkpoint="$run_dir/$checkpoint_name"
if [[ ! -f "$source_checkpoint" || ! -f "$run_dir/config.yaml" ]]; then
  printf 'Missing checkpoint or saved config: %s\n' "$source_checkpoint" >&2
  exit 2
fi

checkpoint_tag="${checkpoint_name%.pt}"
run_name="$(basename -- "$run_dir")"
output_root="${ELF3_EVAL_OUTPUT_DIR:-logs_eval/elf3_policy_comparison_20261109}"
output_dir="$output_root/$run_name/$checkpoint_tag/$mode"
# eval_agent_trl creates exported/ and can copy last.pt next to its input.
# Stage symlinks so those writes never go into the source training directory.
input_dir="$output_root/$run_name/$checkpoint_tag/input"
mkdir -p -- "$input_dir"
ln -sfn -- "$(realpath -- "$source_checkpoint")" "$input_dir/$checkpoint_name"
ln -sfn -- "$(realpath -- "$run_dir/config.yaml")" "$input_dir/config.yaml"
checkpoint="$input_dir/$checkpoint_name"
common=(
  "+checkpoint=$checkpoint"
  '++use_encoder=g1'
  '++seed=0'
  '++use_wandb=false'
  "++eval_name=$run_name-$checkpoint_tag-$mode"
)

case "$mode" in
  metrics)
    options=(
      '+headless=True'
      "++num_envs=${ELF3_EVAL_NUM_ENVS:-64}"
      '+callbacks=im_eval_elf3'
      '++eval_callbacks=im_eval'
      '++run_eval_loop=False'
      "++eval_output_dir=$output_dir"
      "++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=${ELF3_EVAL_MAX_MOTIONS:-512}"
      '++manager_env.terminations.anchor_pos.params.threshold=0.25'
      '++manager_env.terminations.anchor_pos.params.threshold_adaptive=false'
      '++manager_env.terminations.anchor_ori_full.params.threshold=1.0'
      '++manager_env.terminations.ee_body_pos.params.threshold=0.25'
      '++manager_env.terminations.ee_body_pos.params.threshold_adaptive=false'
      '++manager_env.terminations.foot_pos_xyz=null'
    )
    ;;
  play)
    options=(
      '+headless=False'
      '++num_envs=4'
      '++eval_callbacks=[]'
      '++run_eval_loop=True'
      '++max_render_steps=3000'
      '++manager_env.config.env_spacing=3.0'
      '++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=4'
    )
    ;;
  video)
    options=(
      '+headless=True'
      '++num_envs=4'
      '+callbacks=im_eval_elf3'
      '++eval_callbacks=im_eval'
      '++run_eval_loop=False'
      '++manager_env.config.render_results=True'
      '++manager_env.config.max_render_envs=4'
      "++manager_env.config.save_rendering_dir=$output_dir"
      '++manager_env.config.env_spacing=10.0'
      '++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=4'
      '~manager_env/recorders=empty'
      '+manager_env/recorders=render'
    )
    ;;
  *)
    printf 'Unknown mode: %s\n' "$mode" >&2
    exit 2
    ;;
esac

# Arguments are passed as separate array elements; no eval or shell parsing.
exec python gear_sonic/eval_agent_trl.py "${common[@]}" "${options[@]}"
