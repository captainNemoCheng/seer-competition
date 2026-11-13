"""Serve the pilot on loopback behind TLS with a private bearer URL path."""
import hmac
import os
from pathlib import Path
import secrets
import socket
from policy_space.runtime.server import PolicySpaceServer, load_policy_model

ROOT = Path('/workspace/x2real-training')


class PathProtectedServer(PolicySpaceServer):
    def __init__(self, *args, access_path, **kwargs):
        super().__init__(*args, **kwargs)
        self.access_path = access_path

    def _handle_client(self, ws):
        if not hmac.compare_digest(ws.request.path, self.access_path):
            ws.close(code=1008, reason='Unauthorized service path')
            return
        return super()._handle_client(ws)


def main():
    private = ROOT/'.service'
    private.mkdir(mode=0o700, exist_ok=True)
    pathfile = private/'access-path'
    if not pathfile.exists():
        with os.fdopen(os.open(pathfile, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600), 'w') as f:
            f.write('/'+secrets.token_urlsafe(32))
    access_path = pathfile.read_text().strip()
    if not access_path.startswith('/') or len(access_path)<32:
        raise ValueError('Invalid service access path')
    sock = socket.create_server(('127.0.0.1',8002))
    model, metadata = load_policy_model(ROOT/'deploy.yaml')
    server = PathProtectedServer(model, host='127.0.0.1',port=8002,metadata=metadata,
                                 sock=sock,access_path=access_path,max_connections=4,max_sessions=1)
    print('SERVICE_READY path stored in private .service/access-path',flush=True)
    try:
        server.serve_forever()
    finally:
        model.close()


if __name__ == '__main__':
    main()
