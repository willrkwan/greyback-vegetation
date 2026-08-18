def compute_normalized_difference(raw_stack, num_band, den_band, mask=None, reducer="median", dim="time", compute=True):
    """Compute a normalized-difference index such as NDVI."""
    subset = raw_stack
    if mask is not None:
        subset = subset.where(mask)

    numerator = subset.sel(band=num_band)
    denominator = subset.sel(band=den_band)
    index = (numerator - denominator) / (numerator + denominator)

    if reducer == "median":
        index = index.median(dim=dim)
    elif reducer == "mean":
        index = index.mean(dim=dim)
    elif reducer is not None:
        raise ValueError(f"Unsupported reducer: {reducer}")

    return index.compute() if compute else index