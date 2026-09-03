def temporal_composite(raw_stack, bands, mask=None, reducer="median", dim="time", compute=True):
    """Select bands, optionally mask, and reduce by a specified reducer over time."""
    subset = raw_stack.sel(band=list(bands))
    if mask is not None:
        subset = subset.where(mask)
    if reducer == "median":
        composite = subset.median(dim=dim)
    elif reducer == "mean":
        composite = subset.mean(dim=dim)
    else:
        raise ValueError(f"Unsupported reducer: {reducer}")
    return composite.compute() if compute else composite

def percentile_stretch(image, low_q=0.02, high_q=0.98, clip_range=(0, 1), compute=True):
    """Apply percentile contrast stretch for visualization."""
    low = image.quantile(low_q)
    high = image.quantile(high_q)
    stretched = (image - low) / (high - low)
    stretched = stretched.clip(*clip_range)
    return stretched.compute() if compute else stretched