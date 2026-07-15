"""Armado de URLs RTSP/SNAP para camara IP (Reolink-style)."""
from urllib.parse import quote


def build_rtsp_url(host, user, password, port, stream_path):
    user_q = quote(user or "", safe="")
    pass_q = quote(password or "", safe="")
    return f"rtsp://{user_q}:{pass_q}@{host}:{port}/{stream_path}"


def build_snap_url(host, user, password, res_query):
    user_q = quote(user or "", safe="")
    pass_q = quote(password or "", safe="")
    return (
        f"http://{host}/cgi-bin/api.cgi?cmd=Snap&channel=0&rs=aaa"
        f"&user={user_q}&password={pass_q}&{res_query}"
    )
