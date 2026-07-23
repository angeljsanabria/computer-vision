# native — extensiones nativas

## my_rga (RGA RK3568)

| Ruta | Descripcion |
|------|-------------|
| `my_rga/` | Codigo fuente del paquete pip |
| `wheels/` | Wheels generados (`*.whl` en .gitignore) |
| `build_my_rga_wsl.sh` | Cross-compile en WSL → `wheels/` |
| `my_rga_smoke.py` | Smoke test tras `pip install` en placa |
| `my_rga/deploy/README.md` | Guia instalacion en anpr-core |

Build:

```bash
chmod +x native/build_my_rga_wsl.sh
./native/build_my_rga_wsl.sh
```
