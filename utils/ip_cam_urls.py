"""Armado de URLs RTSP/RTMP/SNAP para camara IP (Reolink-style)."""
from urllib.parse import quote

# Perfiles RTMP Reolink standalone (doc oficial): path, channel, stream id.
_REOLINK_RTMP_PROFILES = {
    "MAIN": ("bcs/channel0_main.bcs", 0, 0),
    "EXT": ("bcs/channel0_ext.bcs", 0, 0),
    "SUB": ("bcs/channel0_sub.bcs", 0, 1),
}


def resolve_rtmp_profile(stream_profile: str) -> tuple[str, int, int]:
    """Resuelve MAIN | EXT | SUB a (stream_path, channel, stream_id)."""
    key = (stream_profile or "").upper().strip()
    try:
        return _REOLINK_RTMP_PROFILES[key]
    except KeyError as exc:
        valid = ", ".join(sorted(_REOLINK_RTMP_PROFILES))
        raise ValueError(
            f"perfil RTMP desconocido {stream_profile!r}; usar {valid}"
        ) from exc


def build_rtsp_url(host, user, password, port, stream_path):
    user_q = quote(user or "", safe="")
    pass_q = quote(password or "", safe="")
    return f"rtsp://{user_q}:{pass_q}@{host}:{port}/{stream_path}"


def build_rtmp_url(host, user, password, port, stream_profile):
    """URL RTMP Reolink; ``stream_profile`` es MAIN, EXT o SUB."""
    stream_path, channel, stream_id = resolve_rtmp_profile(stream_profile)
    user_q = quote(user or "", safe="")
    pass_q = quote(password or "", safe="")
    port_s = str(port).strip() if port else "1935"
    if port_s and port_s != "1935":
        authority = f"{host}:{port_s}"
    else:
        authority = host
    return (
        f"rtmp://{authority}/{stream_path}"
        f"?channel={channel}&stream={stream_id}&user={user_q}&password={pass_q}"
    )


def build_snap_url(host, user, password, res_query):
    user_q = quote(user or "", safe="")
    pass_q = quote(password or "", safe="")
    return (
        f"http://{host}/cgi-bin/api.cgi?cmd=Snap&channel=0&rs=aaa"
        f"&user={user_q}&password={pass_q}&{res_query}"
    )
