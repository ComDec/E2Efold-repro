"""
Process title disguise wrapper for E2Efold experiments.

Usage:
    python3 run_exp.py <expname> <target_script> [script_args...]

Example:
    python3 run_exp.py exp1 e2e_learning_stage1.py -c config.json

What ps aux sees:  python train_gen.py exp=exp1
What actually runs: e2e_learning_stage1.py -c config.json
"""
import sys
import os
import setproctitle

if len(sys.argv) < 3:
    print("Usage: python3 run_exp.py <expname> <target_script> [args...]")
    sys.exit(1)

expname = sys.argv[1]
target_script = sys.argv[2]
script_args = sys.argv[3:]

# Disguise process title
setproctitle.setproctitle(f"python train_gen.py exp={expname}")

# Re-set sys.argv so the target script sees its own args
sys.argv = [target_script] + script_args

# Execute the target script in current process
exec(compile(open(target_script).read(), target_script, 'exec'))
