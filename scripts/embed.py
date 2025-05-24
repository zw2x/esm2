import os
from pathlib import Path
import click
import dotenv

dotenv.load_dotenv()
from omegaconf import OmegaConf
import torch
from esm2.logging_utils import get_logger

log = get_logger(__name__)


@click.group()
def cli_main():
    """CLI for extracting embeddings from a model."""
    pass


@cli_main.command("run")
@click.option(
    "--model_cfg",
    type=str,
    required=True,
    help="Path to model configuration file.",
    default=Path(os.environ["ESM2_CONFIG_DIR"]) / "worker/esm2.yaml",
)
def run_esm2_worker(model_cfg):
    from esm2.esm2_worker import ESM2Worker

    model_cfg = OmegaConf.load(model_cfg)
    log.info(f"Loaded model configuration from {model_cfg}")

    worker = ESM2Worker(model_cfg)
    model = worker.model
    log.info(model)
    model.eval()
    model.cuda()

    data = [
        ("protein1", ["MKTV", "RQGAC"]),
        ("protein2", ["KALTRA"]),
        ("protein3", ["KAAISQQ"]),
    ]
    chain_index = [
        torch.cat([torch.full((len(s),), k) for k, s in enumerate(seqs)]) for _, seqs in data
    ]

    def _pad(t):
        max_len = max(len(s) for s in chain_index)
        seq_len = len(t)
        if seq_len >= max_len:
            return t[:max_len]
        return torch.cat([t, torch.full((max_len - seq_len,), 0)])

    chain_index = torch.stack([_pad(t) for t in chain_index], dim=0)

    seqs = [(desc, "".join(seqs)) for (desc, seqs) in data]
    tokens = worker.encode_seqs(seqs)
    mask = worker.trim_node((tokens != worker.pad_id) * (tokens != worker.eos_id))
    tokens = tokens.cuda()
    chain_out = worker.embed(
        {"esm_tokens": tokens, "chain_index": chain_index.cuda(), "mask": mask.cuda()},
        split_chains=True,
    )
    out = worker.embed(
        {"esm_tokens": tokens, "chain_index": chain_index.cuda(), "mask": mask.cuda()},
        split_chains=False,
    )
    print(torch.norm(chain_out["esm_node_acts"] - out["esm_node_acts"], dim=-1))


if __name__ == "__main__":
    cli_main()
