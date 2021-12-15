"""
what does it mean by "fetch"?: https://www.quora.com/What-does-fetch-means-in-computer-science
Korpora 에서도 fetch 라고 표현한다: https://github.com/ko-nlp/Korpora
"""
import wandb
import yaml
from typing import Tuple, List
from Korpora import KoreanParallelKOENNewsKorpus
from tokenizers import Tokenizer
from cleanformer.paths import CONFIG_YAML, KORPORA_DIR, tokenizer_dir, transformer_dir
from cleanformer.models import Transformer


def fetch_kor2eng() -> Tuple[List[Tuple[str, str]],
                             List[Tuple[str, str]],
                             List[Tuple[str, str]]]:
    # download the data
    korpus = KoreanParallelKOENNewsKorpus(root_dir=str(KORPORA_DIR))
    kor2eng_train = list(zip(korpus.train.texts, korpus.train.pairs))
    kor2eng_val = list(zip(korpus.dev.texts, korpus.dev.pairs))
    kor2eng_test = list(zip(korpus.test.texts, korpus.test.pairs))
    return kor2eng_train, kor2eng_val, kor2eng_test


def fetch_tokenizer(entity: str, ver: str) -> Tokenizer:
    artifact = wandb.Api().artifact(f"{entity}/cleanformer/tokenizer:{ver}", type="other")
    artifact_path = tokenizer_dir(ver)
    json_path = artifact_path / "tokenizer.json"
    artifact.download(root=str(artifact_path))
    tokenizer = Tokenizer.from_file(str(json_path))
    # just manually register the special tokens
    tokenizer.pad_token = artifact.metadata['pad']
    tokenizer.pad_token_id = artifact.metadata['pad_id']
    tokenizer.unk_token = artifact.metadata['unk']
    tokenizer.unk_token_id = artifact.metadata['unk_id']
    tokenizer.bos_token = artifact.metadata['bos']
    tokenizer.bos_token_id = artifact.metadata['bos_id']
    tokenizer.eos_token = artifact.metadata['eos']
    tokenizer.eos_token_id = artifact.metadata['eos_id']
    return tokenizer


def fetch_transformer(entity: str, ver: str) -> Transformer:
    artifact_path = transformer_dir(ver)
    ckpt_path = artifact_path / "transformer.ckpt"
    wandb.Api().artifact(f"{entity}/cleanformer/transformer:{ver}", type="model") \
               .download(root=str(artifact_path))
    transformer = Transformer.load_from_checkpoint(str(ckpt_path))
    return transformer


# --- fetchers for fetching local files --- #
def fetch_config() -> dict:
    """
    just load the config file from local
    """
    with open(str(CONFIG_YAML), 'r') as fh:
        return yaml.safe_load(fh)
