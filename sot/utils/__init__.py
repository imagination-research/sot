def _print_to_streams(streams, text, **print_kwargs):
    for stream in streams:
        print(text, file=stream, **print_kwargs)
