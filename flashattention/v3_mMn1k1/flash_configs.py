#!/usr/bin/env python3

import os
import re


OUTPUT = os.environ.get("FLASH_FWD_OUTPUT", "configs.cmake")
MAX_SMEM_KB = int(os.environ.get("FLASH_FWD_MAX_SMEM_KB", "100"))
ONLY_CONFIGS = os.environ.get("FLASH_FWD_ONLY_CONFIGS", "")

HEAD_DIM_VALUES = (64, 128)
HEAD_DIM_TO_BM = {64: (128, 256), 128: (64, 128)}
HEAD_DIM_TO_BN = {64: (32, 64,), 128: (32, 64)}
NWARPS = (8, )

STAGE = 2

WARP_SIZE = 32
PRODUCER_WARPS = 1
ELEMENT_BYTES = 2
MAX_THREADS_PER_BLOCK = 1024
BLOCK_TILE_UNIT = 16
MIN_BLOCK_TILE_AREA = 4
CONFIG_NAME_RE = re.compile(
    r"^hd(?P<head_dim>\d+)_bm(?P<block_m>\d+)_bn(?P<block_n>\d+)_w(?P<nwarps>\d+)_s(?P<stage>\d+)$"
)


def align_up(value, alignment):
    return ((value + alignment - 1) // alignment) * alignment


def pipeline_storage_bytes(stage):
    array_bytes = stage * 8
    offset = align_up(array_bytes, 16)
    return align_up(offset + array_bytes, 16)


def smem_bytes(head_dim, block_m, block_n, stage):
    q_size = block_m * head_dim
    k_size = block_n * head_dim
    v_size = k_size
    o_size = q_size
    qk_size = k_size + (q_size if q_size >= k_size else k_size)
    qkv_size = qk_size + v_size
    qkvo_size = max(qkv_size, o_size)
    offset = 0
    offset = align_up(offset, 128)
    offset += qkvo_size * ELEMENT_BYTES
    offset = align_up(offset, 16)
    offset += pipeline_storage_bytes(stage)
    offset = align_up(offset, 16)
    offset += pipeline_storage_bytes(1)
    return align_up(offset, 128)


def config_name(head_dim, block_m, block_n, nwarps, stage):
    return f"hd{head_dim}_bm{block_m}_bn{block_n}_w{nwarps}_s{stage}"


def config_values(value):
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


def block_shapes(head_dim):
    block_ms = config_values(HEAD_DIM_TO_BM[head_dim])
    block_ns = config_values(HEAD_DIM_TO_BN[head_dim])
    return ((block_m, block_n) for block_m in block_ms for block_n in block_ns)


def nwarp_values():
    return config_values(NWARPS)


def is_valid_config(head_dim, block_m, block_n, nwarps, stage, max_smem):
    if head_dim <= 0 or block_m <= 0 or block_n <= 0 or nwarps <= 0 or stage <= 0:
        return False
    if (block_m // BLOCK_TILE_UNIT) * (block_n // BLOCK_TILE_UNIT) < MIN_BLOCK_TILE_AREA:
        return False
    if nwarps not in nwarp_values():
        return False
    if head_dim % 32 != 0:
        return False
    if stage != STAGE:
        return False
    if block_m % (nwarps * BLOCK_TILE_UNIT) != 0:
        return False
    if block_n % BLOCK_TILE_UNIT != 0:
        return False
    if (PRODUCER_WARPS + nwarps) * WARP_SIZE > MAX_THREADS_PER_BLOCK:
        return False
    if block_n * head_dim * ELEMENT_BYTES <= 0:
        return False
    if smem_bytes(head_dim, block_m, block_n, stage) > max_smem:
        return False
    return True


def split_config_names(value):
    return [name for name in re.split(r"[,;\s]+", value.strip()) if name]


def make_config(head_dim, block_m, block_n, nwarps, stage):
    return (
        head_dim,
        block_m,
        block_n,
        nwarps,
        stage,
        smem_bytes(head_dim, block_m, block_n, stage),
        config_name(head_dim, block_m, block_n, nwarps, stage),
    )


def parse_config_name(name, max_smem):
    match = CONFIG_NAME_RE.match(name)
    if not match:
        raise ValueError(
            f"invalid flash_fwd config name '{name}', expected "
            "hd<head_dim>_bm<block_m>_bn<block_n>_w<nwarps>_s<stage>"
        )
    values = {key: int(value) for key, value in match.groupdict().items()}
    if not is_valid_config(
        values["head_dim"],
        values["block_m"],
        values["block_n"],
        values["nwarps"],
        values["stage"],
        max_smem,
    ):
        raise ValueError(
            f"invalid flash_fwd config '{name}' for max shared memory {max_smem // 1024} KiB"
        )
    return make_config(**values)


def append_unique_config(configs, cfg):
    name = cfg[-1]
    if not any(existing[-1] == name for existing in configs):
        configs.append(cfg)


def default_configs(max_smem):
    configs = []
    for head_dim in HEAD_DIM_VALUES:
        for block_m, block_n in block_shapes(head_dim):
            for nwarps in nwarp_values():
                if not is_valid_config(head_dim, block_m, block_n, nwarps, STAGE, max_smem):
                    continue
                configs.append(make_config(head_dim, block_m, block_n, nwarps, STAGE))
    return configs


def generate_configs(max_smem_kb, only_config_names=()):
    max_smem = max_smem_kb * 1024
    if only_config_names:
        configs = []
        for name in only_config_names:
            append_unique_config(configs, parse_config_name(name, max_smem))
        return configs

    return default_configs(max_smem)


def main():
    configs = generate_configs(
        MAX_SMEM_KB,
        split_config_names(ONLY_CONFIGS),
    )
    if not configs:
        raise RuntimeError("no valid flash_fwd configs generated")

    default_name = "hd64_bm128_bn32_w8_s2"
    default_config = next((cfg for cfg in configs if cfg[-1] == default_name), configs[0])

    output_dir = os.path.dirname(OUTPUT)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write("set(FLASH_FWD_CONFIGS\n")
        for cfg in configs:
            f.write(f'    "{"|".join(str(x) for x in cfg)}"\n')
        f.write(")\n")
        f.write(f'set(FLASH_FWD_DEFAULT_CONFIG "{"|".join(str(x) for x in default_config)}")\n')

    print(f"Generated {len(configs)} configs -> {OUTPUT}")


if __name__ == "__main__":
    main()
