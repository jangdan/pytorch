from dataclasses import dataclass

import torch._inductor.config as config

from ..utils import ensure_cute_available


if ensure_cute_available():
    import cutlass.utils as utils  # type: ignore[import-not-found]


@dataclass(frozen=True)
class CuTeGemmConfig:
    TILE_M: int = 128
    TILE_N: int = 192
    CLUSTER_M: int = 2
    CLUSTER_N: int = 1
    USE_2_CTA: bool = False
    TENSORMAP_UPDATE_MODE: utils.TensorMapUpdateMode = utils.TensorMapUpdateMode.SMEM


def get_exhaustive_groupgemm_configs() -> list[CuTeGemmConfig]:
    """
    Returns the exhaustive configuration set for the Blackwell CuTeDSL Grouped GEMM kernel.
    For information regarding valid config sets, see:
    https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/grouped_gemm.py
    """

    # Tile_n is always the same regardless of 2cta
    tile_n_vals = [32, 64, 96, 128, 160, 192, 224, 256]

    # Valid clusters
    clusters_no_2cta = [
        (1, 1),
        (1, 2),
        (1, 4),
        (1, 8),
        (1, 16),
        (2, 1),
        (2, 2),
        (2, 4),
        (2, 8),
        (4, 1),
        (4, 2),
        (4, 4),
        (8, 1),
        (8, 2),
        (16, 1),
    ]
    clusters_2cta = [
        (2, 1),
        (2, 2),
        (2, 4),
        (2, 8),
        (4, 1),
        (4, 2),
        (4, 4),
        (8, 1),
        (8, 2),
        (16, 1),
    ]

    configs: list[CuTeGemmConfig] = []

    # Non-2cta configs
    for tensormap_update_mode in [
        utils.TensorMapUpdateMode.SMEM,
        utils.TensorMapUpdateMode.GMEM,
    ]:
        for tile_m in [64, 128]:
            for tile_n in tile_n_vals:
                for cluster_m, cluster_n in clusters_no_2cta:
                    configs.append(
                        CuTeGemmConfig(
                            tile_m,
                            tile_n,
                            cluster_m,
                            cluster_n,
                            USE_2_CTA=False,
                            TENSORMAP_UPDATE_MODE=tensormap_update_mode,
                        )
                    )

    for tensormap_update_mode in [
        utils.TensorMapUpdateMode.SMEM,
        utils.TensorMapUpdateMode.GMEM,
    ]:
        for tile_m in [128, 256]:
            for tile_n in tile_n_vals:
                for cluster_m, cluster_n in clusters_2cta:
                    configs.append(
                        CuTeGemmConfig(
                            tile_m,
                            tile_n,
                            cluster_m,
                            cluster_n,
                            USE_2_CTA=True,
                            TENSORMAP_UPDATE_MODE=tensormap_update_mode,
                        )
                    )

    return configs


def get_default_groupgemm_configs() -> list[CuTeGemmConfig]:
    """
    Returns the default configuration set for the Blackwell CuTeDSL Grouped GEMM kernel.
    """

    return [
        CuTeGemmConfig(
            TILE_M=128,
            TILE_N=256,
            CLUSTER_M=2,
            CLUSTER_N=1,
            USE_2_CTA=False,
            TENSORMAP_UPDATE_MODE=utils.TensorMapUpdateMode.SMEM,
        ),
        CuTeGemmConfig(
            TILE_M=256,
            TILE_N=160,
            CLUSTER_M=2,
            CLUSTER_N=1,
            USE_2_CTA=True,
            TENSORMAP_UPDATE_MODE=utils.TensorMapUpdateMode.GMEM,
        ),
        CuTeGemmConfig(
            TILE_M=256,
            TILE_N=256,
            CLUSTER_M=2,
            CLUSTER_N=1,
            USE_2_CTA=True,
            TENSORMAP_UPDATE_MODE=utils.TensorMapUpdateMode.GMEM,
        ),
        CuTeGemmConfig(
            TILE_M=64,
            TILE_N=32,
            CLUSTER_M=1,
            CLUSTER_N=1,
            USE_2_CTA=False,
            TENSORMAP_UPDATE_MODE=utils.TensorMapUpdateMode.GMEM,
        ),
        CuTeGemmConfig(
            TILE_M=64,
            TILE_N=256,
            CLUSTER_M=1,
            CLUSTER_N=2,
            USE_2_CTA=False,
            TENSORMAP_UPDATE_MODE=utils.TensorMapUpdateMode.SMEM,
        ),
        CuTeGemmConfig(
            TILE_M=128,
            TILE_N=256,
            CLUSTER_M=1,
            CLUSTER_N=2,
            USE_2_CTA=False,
            TENSORMAP_UPDATE_MODE=utils.TensorMapUpdateMode.SMEM,
        ),
        CuTeGemmConfig(
            TILE_M=256,
            TILE_N=256,
            CLUSTER_M=2,
            CLUSTER_N=2,
            USE_2_CTA=True,
            TENSORMAP_UPDATE_MODE=utils.TensorMapUpdateMode.GMEM,
        ),
        CuTeGemmConfig(
            TILE_M=128,
            TILE_N=256,
            CLUSTER_M=1,
            CLUSTER_N=2,
            USE_2_CTA=False,
            TENSORMAP_UPDATE_MODE=utils.TensorMapUpdateMode.GMEM,
        ),
        CuTeGemmConfig(
            TILE_M=64,
            TILE_N=32,
            CLUSTER_M=1,
            CLUSTER_N=1,
            USE_2_CTA=False,
            TENSORMAP_UPDATE_MODE=utils.TensorMapUpdateMode.SMEM,
        ),
        CuTeGemmConfig(
            TILE_M=256,
            TILE_N=256,
            CLUSTER_M=2,
            CLUSTER_N=1,
            USE_2_CTA=True,
            TENSORMAP_UPDATE_MODE=utils.TensorMapUpdateMode.SMEM,
        ),
        CuTeGemmConfig(
            TILE_M=128,
            TILE_N=256,
            CLUSTER_M=1,
            CLUSTER_N=1,
            USE_2_CTA=False,
            TENSORMAP_UPDATE_MODE=utils.TensorMapUpdateMode.GMEM,
        ),
        CuTeGemmConfig(
            TILE_M=256,
            TILE_N=256,
            CLUSTER_M=8,
            CLUSTER_N=1,
            USE_2_CTA=True,
            TENSORMAP_UPDATE_MODE=utils.TensorMapUpdateMode.GMEM,
        ),
        CuTeGemmConfig(
            TILE_M=64,
            TILE_N=32,
            CLUSTER_M=1,
            CLUSTER_N=2,
            USE_2_CTA=False,
            TENSORMAP_UPDATE_MODE=utils.TensorMapUpdateMode.SMEM,
        ),
        CuTeGemmConfig(
            TILE_M=256,
            TILE_N=192,
            CLUSTER_M=2,
            CLUSTER_N=1,
            USE_2_CTA=True,
            TENSORMAP_UPDATE_MODE=utils.TensorMapUpdateMode.GMEM,
        ),
        CuTeGemmConfig(
            TILE_M=256,
            TILE_N=256,
            CLUSTER_M=2,
            CLUSTER_N=2,
            USE_2_CTA=True,
            TENSORMAP_UPDATE_MODE=utils.TensorMapUpdateMode.SMEM,
        ),
        CuTeGemmConfig(
            TILE_M=128,
            TILE_N=96,
            CLUSTER_M=1,
            CLUSTER_N=2,
            USE_2_CTA=False,
            TENSORMAP_UPDATE_MODE=utils.TensorMapUpdateMode.SMEM,
        ),
        CuTeGemmConfig(
            TILE_M=64,
            TILE_N=192,
            CLUSTER_M=1,
            CLUSTER_N=1,
            USE_2_CTA=False,
            TENSORMAP_UPDATE_MODE=utils.TensorMapUpdateMode.SMEM,
        ),
        CuTeGemmConfig(
            TILE_M=64,
            TILE_N=64,
            CLUSTER_M=1,
            CLUSTER_N=1,
            USE_2_CTA=False,
            TENSORMAP_UPDATE_MODE=utils.TensorMapUpdateMode.GMEM,
        ),
        CuTeGemmConfig(
            TILE_M=64,
            TILE_N=192,
            CLUSTER_M=1,
            CLUSTER_N=1,
            USE_2_CTA=False,
            TENSORMAP_UPDATE_MODE=utils.TensorMapUpdateMode.GMEM,
        ),
        CuTeGemmConfig(
            TILE_M=128,
            TILE_N=64,
            CLUSTER_M=1,
            CLUSTER_N=1,
            USE_2_CTA=False,
            TENSORMAP_UPDATE_MODE=utils.TensorMapUpdateMode.GMEM,
        ),
        CuTeGemmConfig(
            TILE_M=64,
            TILE_N=160,
            CLUSTER_M=1,
            CLUSTER_N=1,
            USE_2_CTA=False,
            TENSORMAP_UPDATE_MODE=utils.TensorMapUpdateMode.GMEM,
        ),
        CuTeGemmConfig(
            TILE_M=64,
            TILE_N=256,
            CLUSTER_M=1,
            CLUSTER_N=1,
            USE_2_CTA=False,
            TENSORMAP_UPDATE_MODE=utils.TensorMapUpdateMode.GMEM,
        ),
    ]


def get_groupgemm_configs() -> list[CuTeGemmConfig]:
    """
    Returns the configuration set for the Blackwell CuTeDSL Grouped GEMM kernel.

    Note: CuTeDSL autotuning is still experimental — enabling it may trigger kernel launch failures
    or unstable results. By default, autotuning is disabled and we return only
    a single baseline config.
    """
    if (
        config.cutedsl_enable_autotuning
        and config.max_autotune_gemm_search_space == "EXHAUSTIVE"
    ):
        return get_exhaustive_groupgemm_configs()
    elif config.cutedsl_enable_autotuning:
        return get_default_groupgemm_configs()
    else:
        return [get_default_groupgemm_configs()[0]]
