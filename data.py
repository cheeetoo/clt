import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformer_lens import HookedTransformer
from datasets import load_dataset
from einops import rearrange


class StreamingActivationDataset:
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        dataset_conf: str,
        batch_size: int,
        n_tokens: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.batch_size = batch_size
        self.n_tokens = n_tokens // dist.get_world_size()
        self.dtype = dtype
        self.device = device

        # each rank loads its own copy - maybe good maybe not
        self.model = HookedTransformer.from_pretrained(
            model_name, device="cpu", dtype=dtype
        ).to(device)
        self.model.eval()

        self.model.cfg.n_ctx = seq_len
        self.n_layers: int = self.model.cfg.n_layers  # type: ignore
        self.d_model: int = self.model.cfg.d_model  # type: ignore

        dataset = (
            load_dataset(dataset_name, dataset_conf, split="train", streaming=True)
            .shard(num_shards=dist.get_world_size(), index=dist.get_rank())
            .filter(lambda ex: ex["text"].strip() != "")
        )

        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=4,
        )

    def __len__(self):
        return self.n_tokens // self.batch_size

    def _get_acts(self, toks):
        with torch.autocast("cuda", dtype=self.dtype):
            with torch.inference_mode():
                _, cache = self.model.run_with_cache(
                    toks,
                    names_filter=lambda n: n.endswith("normalized")
                    or n.endswith("mlp_out"),
                )

            pre = torch.stack(
                [cache["normalized", i, "ln2"] for i in range(self.n_layers)]
            )
            post = torch.stack([cache["mlp_out", i] for i in range(self.n_layers)])
            pre = rearrange(pre, "l b t d -> (b t) l d")
            post = rearrange(post, "l b t d -> (b t) l d")

        return pre, post

    def __iter__(self):
        tokens_processed = 0

        while tokens_processed < self.n_tokens:
            for batch in self.dataloader:
                if tokens_processed >= self.n_tokens:
                    break

                toks = self.model.to_tokens(batch["text"], truncate=True)
                toks = toks.to(self.device, non_blocking=True)
                b, s = toks.shape
                if s < self.model.cfg.n_ctx:
                    pad = self.model.cfg.n_ctx - s
                    pad_id = self.model.tokenizer.pad_token_id
                    toks = F.pad(toks, (0, pad), value=pad_id)

                tokens_processed += b * s

                pre, post = self._get_acts(toks)

                yield pre, post
