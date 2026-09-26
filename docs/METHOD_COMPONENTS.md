# Method Components

## Overview

The implementation separates semantic preference-distribution modeling from caption decoding. Public interfaces use the same component names as the method description.

## Conditional Transport

`recdiffusion.transport.HistoryConditionedRFM` implements Riemannian flow matching conditioned only on the observed-history pathway.

`recdiffusion.transport.DualViewConditionedRFM` separately encodes observed history and collaborative evidence into memories `M_H` and `M_E`. Each flow block uses separate cross-attention modules and a learned softmax gate to combine both views.

Both transports predict tangent velocities on the unit sphere and use spherical Heun integration to obtain semantic endpoints.

## Conditional Source Calibration

`recdiffusion.source_calibration.ConditionalSourceCalibration` implements the bounded source map `A_psi(r; C_u)`. It predicts a tangent displacement and applies the spherical exponential map. The output layer is initialized to zero, making the initial map an exact identity.

The full method freezes the dual-view conditioned transport while optimizing conditional source calibration through the numerical solver.

## Content Decoding

`recdiffusion.caption_decoder.EmbeddingConditionedCaptionDecoder` implements the deterministic map from a sampled semantic representation to a caption. It receives the sampled representation rather than the user's history or collaborative evidence. The same frozen decoder is shared across methods during evaluation.

## Variants

| Variant | Conditional transport | Collaborative evidence | Source calibration | Caption decoder |
|---|---|---:|---:|---|
| History-conditioned Riemannian flow matching | History-conditioned | No | No | Shared and frozen |
| Ours without dual-view conditioning | History-conditioned | No | Yes | Shared and frozen |
| Ours without conditional source calibration | Dual-view conditioned | Yes | No | Shared and frozen |
| Ours | Dual-view conditioned | Yes | Yes | Shared and frozen |

