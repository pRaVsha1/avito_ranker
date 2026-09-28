"""Необязательная однократная загрузка фиксированной open-source модели."""

import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_ID = 'intfloat/multilingual-e5-small'
REVISION = '614241f622f53c4eeff9890bdc4f31cfecc418b3'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts-dir', default='artifacts')
    args = parser.parse_args()
    print(
        snapshot_download(
            MODEL_ID,
            revision=REVISION,
            local_dir=Path(args.artifacts_dir) / 'e5-small',
            allow_patterns=[
                'config.json',
                'model.safetensors',
                'tokenizer.json',
                'tokenizer_config.json',
                'special_tokens_map.json',
                'sentencepiece.bpe.model',
            ],
        )
    )
