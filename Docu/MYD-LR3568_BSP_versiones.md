# MYD-LR3568 / MYD-LR3568-GK-B: versiones de BSP y alineacion RKNN

Este documento complementa los PDF en esta carpeta con una **checklist** para alinear el PC de export (RKNN-Toolkit2) con la **SOM/placa** y el **runtime** en dispositivo.

## Datos fijados desde documentacion MYIR (repo local)

| Campo | Valor (segun PDF) | Fuente |
|-------|-------------------|--------|
| SoC | Rockchip RK3568 | `MYD-LR3568 Development Board Overview.pdf` |
| Variante IPC Box | RK3568J, quad A55 (1.4 GHz; opcional 1.8 GHz overdrive segun MYIR) | `MYD-LR3568-GK-B IPC Box Overview.pdf` |
| NPU | Hasta 1.0 TOPS | Ambos PDF |
| GPU | Mali-G52 2EE | Ambos PDF |
| RAM / almacenamiento tipico | 2 GB/4 GB LPDDR4; 16 GB/32 GB eMMC | PDF |
| OS declarado | Linux y Debian; en placa de desarrollo se cita Linux 5.10 y Debian 11 en una seccion; IPC Box cita SDK Linux/Debian | PDF |

## Campos que debes completar desde tu imagen/SDK MYIR

Rellena cuando tengas la placa o el paquete de descarga del fabricante:

| Campo | Valor (rellenar) | Notas |
|-------|------------------|--------|
| Nombre archivo imagen SD / imagen eMMC | | Ej. `debian-xxx.img` |
| Version Debian/Ubuntu en la placa | | `cat /etc/os-release` |
| Version kernel | | `uname -r` |
| Version driver NPU / modulo `rknpu` | | Segun BSP MYIR o `dmesg` |
| Version **librknnrt** en el dispositivo | | Paquete o `.so` incluido en BSP |
| Version **RKNN-Toolkit2** usada al exportar `.rknn` | | Debe ser **compatible** con runtime de la tabla anterior |

## Regla de alineacion

- **RKNN-Toolkit2** (host x86_64 Linux, p. ej. en Docker) y **RKNN Runtime / RKNN Lite** (dispositivo aarch64) deben corresponder a la **misma generacion** de SDK que documente Rockchip/MYIR para RK3568.
- Si mezclas versiones arbitrarias, suele fallar la carga del modelo o aparecen errores de operadores.

## Referencias utiles (externas)

- Repositorio toolkit (export / PC): [airockchip/rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2)
- Runtime en dispositivo: paquetes `rknn-toolkit-lite2` / librerias del BSP del fabricante.

## Estado

- Checklist base creada: **completar** filas de la segunda tabla al validar hardware real.
