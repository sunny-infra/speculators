python scripts/prepare_data.py \
  --model /mnt/sfs_turbo/models/ZhipuAI/GLM-5.2/ \
  --trust-remote-code \
  --data /mnt/sfs_turbo/yc02324691/codes/speculators-sunny/data/data_jsonl \
  --output /mnt/sfs_turbo/yc02324691/codes/speculators-sunny/data/glm52-perfect/prepare_data_60w \
  --seq-length 4096 \
  --max-samples 600000 \
  --minimum-valid-tokens 48 \
  --assistant-pattern '<\|assistant\|>((?:(?!<\|user\|>|<\|assistant\|>).)*)'
