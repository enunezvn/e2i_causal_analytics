import resource, time, sys, json
rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
t0 = time.perf_counter()
try:
    import doubleml
    err = None
except Exception as e:
    err = repr(e)
dt = time.perf_counter() - t0
rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(json.dumps({"import_error": err, "import_s": round(dt, 3), "rss_before_mib": round(rss0, 1), "rss_after_mib": round(rss1, 1), "rss_delta_mib": round(rss1 - rss0, 1), "doubleml": getattr(doubleml, "__version__", None) if not err else None, "plotly_loaded": "plotly" in sys.modules}))
