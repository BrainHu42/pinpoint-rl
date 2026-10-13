__all__ = ["AttentionInterpolateGeoModel", "build_model", "predict"]


def __getattr__(name: str):
    if name in __all__:
        from pinpoint.attention_reranker.inference import (
            AttentionInterpolateGeoModel,
            build_model,
            predict,
        )

        exports = {
            "AttentionInterpolateGeoModel": AttentionInterpolateGeoModel,
            "build_model": build_model,
            "predict": predict,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
