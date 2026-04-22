# Desarrollo en macOS

## Que hacer en el Mac

- Editar codigo, manifiestos y documentacion en este repo.
- **Docker Desktop** con emulacion **linux/amd64** para construir y ejecutar el contenedor de [docker/](docker/) (RKNN-Toolkit2 oficialmente orientado a **Linux x86_64**).

## Docker Desktop (recomendado)

1. Instalar [Docker Desktop para Mac](https://www.docker.com/products/docker-desktop/).
2. Al construir la imagen, forzar plataforma amd64 si tu Mac es Apple Silicon:

```bash
cd edge_iot_myd
docker build --platform linux/amd64 -f docker/Dockerfile -t rknn-toolkit2-export .
```

3. Ejecutar export de modelos dentro del contenedor (ver [docker/README.md](docker/README.md)).

## Sin Docker

- Usar una **VM Linux x86_64** (UTM, Parallels, VMware) con Ubuntu 22.04 e instalar RKNN-Toolkit2 siguiendo el mismo README del directorio `docker/`, o un PC Linux dedicado.

La GPU del Mac **no** es un requisito para exportar `.rknn`.
