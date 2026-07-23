#ifndef MY_RGA_IMAGE_UTILS_PORT_H_
#define MY_RGA_IMAGE_UTILS_PORT_H_

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

int my_rga_get_image_size(image_buffer_t* image);

int my_rga_convert_image(
    image_buffer_t* src_img,
    image_buffer_t* dst_img,
    image_rect_t* src_box,
    image_rect_t* dst_box,
    unsigned char color,
    int* used_rga);

int my_rga_letterbox_bgr(
    image_buffer_t* src_image,
    image_buffer_t* dst_image,
    letterbox_t* letterbox,
    unsigned char fill_color,
    int* used_rga);

int my_rga_bgr_to_rgb(
    image_buffer_t* src_bgr,
    image_buffer_t* dst_rgb,
    int* used_rga);

#ifdef __cplusplus
}
#endif

#endif
