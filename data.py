import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset
from transformer_lens import HookedTransformer
from datasets import load_dataset
from einops import rearrange


class StreamingActivationDataset(IterableDataset):
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        dataset_conf: str,
        batch_size: int,
        n_tokens: int,
        device: str,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.batch_size = batch_size
        self.n_tokens = n_tokens
        self.dtype = dtype

        # each rank loads its own copy - maybe good maybe not
        self.model = HookedTransformer.from_pretrained(
            model_name, device=device, dtype=dtype
        )
        self.model.cfg.n_ctx = self.model.cfg.n_ctx // 32
        self.n_layers = self.model.cfg.n_layers
        self.d_model = self.model.cfg.d_model

        dataset = load_dataset(
            dataset_name, dataset_conf, split="train", streaming=True
        ).shard(num_shards=dist.get_world_size(), index=dist.get_rank())

        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    def __len__(self):
        return self.n_tokens // self.batch_size

    def __iter__(self):
        tokens_processed = 0

        for batch in self.dataloader:
            if tokens_processed >= self.n_tokens:
                break

            toks = self.model.to_tokens(batch["text"], truncate=True)
            n_toks = toks.shape[0] * toks.shape[1]

            with torch.no_grad():
                _, cache = self.model.run_with_cache(
                    toks,
                    names_filter=lambda n: n.endswith("normalized")
                    or n.endswith("mlp_out"),
                )

            pre = torch.stack(
                [cache["normalized", i, "ln2"] for i in range(self.n_layers)]
            )
            post = torch.stack([cache["mlp_out", i] for i in range(self.n_layers)])
            pre = rearrange(pre, "l b t d -> (b t) l d").to(self.dtype)
            post = rearrange(post, "l b t d -> (b t) l d").to(self.dtype)

            tokens_processed += n_toks

            yield pre, post
