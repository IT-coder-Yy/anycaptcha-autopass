"""内外双盘接缝旋转配准。"""
from common.vision import Prediction


def predict(background, piece):
    from rotate_dual.measure import measure_rotation
    if piece.ndim != 3 or piece.shape[2] != 4:
        return Prediction("rotate_dual", False, reason="需要含透明通道的内盘图；不能从矩形尺寸猜半径")
    result = measure_rotation(background[:, :, :3], piece)
    return Prediction("rotate_dual", result.deg is not None and result.conf >= .75, result.conf,
                      value=result.deg, reason=result.note,
                      detail={"radius": result.radius}, elapsed_ms=result.ms)
