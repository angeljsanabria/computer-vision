#include "image_utils_port.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "im2d.h"
#include "rga.h"

static int crop_and_scale_image_c(
    int channel,
    unsigned char* src,
    int src_width,
    int src_height,
    int crop_x,
    int crop_y,
    int crop_width,
    int crop_height,
    unsigned char* dst,
    int dst_width,
    int dst_height,
    int dst_box_x,
    int dst_box_y,
    int dst_box_width,
    int dst_box_height)
{
    if (dst == NULL) {
        return -1;
    }

    float x_ratio = (float)crop_width / (float)dst_box_width;
    float y_ratio = (float)crop_height / (float)dst_box_height;

    for (int dst_y = dst_box_y; dst_y < dst_box_y + dst_box_height; dst_y++) {
        for (int dst_x = dst_box_x; dst_x < dst_box_x + dst_box_width; dst_x++) {
            int dst_x_offset = dst_x - dst_box_x;
            int dst_y_offset = dst_y - dst_box_y;

            int src_x = (int)(dst_x_offset * x_ratio) + crop_x;
            int src_y = (int)(dst_y_offset * y_ratio) + crop_y;

            float x_diff = (dst_x_offset * x_ratio) - (src_x - crop_x);
            float y_diff = (dst_y_offset * y_ratio) - (src_y - crop_y);

            int index1 = src_y * src_width * channel + src_x * channel;
            int index2 = index1 + src_width * channel;
            if (src_y == src_height - 1) {
                index2 = index1 - src_width * channel;
            }
            int index3 = index1 + 1 * channel;
            int index4 = index2 + 1 * channel;
            if (src_x == src_width - 1) {
                index3 = index1 - 1 * channel;
                index4 = index2 - 1 * channel;
            }

            for (int c = 0; c < channel; c++) {
                unsigned char a = src[index1 + c];
                unsigned char b = src[index3 + c];
                unsigned char cpx = src[index2 + c];
                unsigned char d = src[index4 + c];

                unsigned char pixel = (unsigned char)(
                    a * (1 - x_diff) * (1 - y_diff) +
                    b * x_diff * (1 - y_diff) +
                    cpx * y_diff * (1 - x_diff) +
                    d * x_diff * y_diff);

                dst[(dst_y * dst_width + dst_x) * channel + c] = pixel;
            }
        }
    }

    return 0;
}

static int get_rga_fmt(image_format_t fmt)
{
    switch (fmt) {
    case IMAGE_FORMAT_RGB888:
        return RK_FORMAT_RGB_888;
    case IMAGE_FORMAT_BGR888:
        return RK_FORMAT_BGR_888;
    case IMAGE_FORMAT_RGBA8888:
        return RK_FORMAT_RGBA_8888;
    case IMAGE_FORMAT_YUV420SP_NV12:
        return RK_FORMAT_YCbCr_420_SP;
    case IMAGE_FORMAT_YUV420SP_NV21:
        return RK_FORMAT_YCrCb_420_SP;
    default:
        return -1;
    }
}

int my_rga_get_image_size(image_buffer_t* image)
{
    if (image == NULL) {
        return 0;
    }
    switch (image->format) {
    case IMAGE_FORMAT_GRAY8:
        return image->width * image->height;
    case IMAGE_FORMAT_RGB888:
    case IMAGE_FORMAT_BGR888:
        return image->width * image->height * 3;
    case IMAGE_FORMAT_RGBA8888:
        return image->width * image->height * 4;
    case IMAGE_FORMAT_YUV420SP_NV12:
    case IMAGE_FORMAT_YUV420SP_NV21:
        return image->width * image->height * 3 / 2;
    default:
        return 0;
    }
}

static int convert_image_cpu(
    image_buffer_t* src,
    image_buffer_t* dst,
    image_rect_t* src_box,
    image_rect_t* dst_box,
    unsigned char color)
{
    if (dst->virt_addr == NULL || src->virt_addr == NULL) {
        return -1;
    }
    if (src->format != dst->format) {
        return -1;
    }

    int src_box_x = 0;
    int src_box_y = 0;
    int src_box_w = src->width;
    int src_box_h = src->height;
    if (src_box != NULL) {
        src_box_x = src_box->left;
        src_box_y = src_box->top;
        src_box_w = src_box->right - src_box->left + 1;
        src_box_h = src_box->bottom - src_box->top + 1;
    }

    int dst_box_x = 0;
    int dst_box_y = 0;
    int dst_box_w = dst->width;
    int dst_box_h = dst->height;
    if (dst_box != NULL) {
        dst_box_x = dst_box->left;
        dst_box_y = dst_box->top;
        dst_box_w = dst_box->right - dst_box->left + 1;
        dst_box_h = dst_box->bottom - dst_box->top + 1;
    }

    if (dst_box_w != dst->width || dst_box_h != dst->height) {
        int dst_size = my_rga_get_image_size(dst);
        memset(dst->virt_addr, color, (size_t)dst_size);
    }

    if (src->format == IMAGE_FORMAT_RGB888 || src->format == IMAGE_FORMAT_BGR888) {
        return crop_and_scale_image_c(
            3,
            src->virt_addr,
            src->width,
            src->height,
            src_box_x,
            src_box_y,
            src_box_w,
            src_box_h,
            dst->virt_addr,
            dst->width,
            dst->height,
            dst_box_x,
            dst_box_y,
            dst_box_w,
            dst_box_h);
    }

    return -1;
}

static int convert_image_rga(
    image_buffer_t* src_img,
    image_buffer_t* dst_img,
    image_rect_t* src_box,
    image_rect_t* dst_box,
    unsigned char color)
{
    int ret = 0;
    int srcWidth = src_img->width;
    int srcHeight = src_img->height;
    void* src = src_img->virt_addr;
    int src_fd = src_img->fd;
    int srcFmt = get_rga_fmt(src_img->format);

    int dstWidth = dst_img->width;
    int dstHeight = dst_img->height;
    void* dst = dst_img->virt_addr;
    int dst_fd = dst_img->fd;
    int dstFmt = get_rga_fmt(dst_img->format);

    int use_handle = 0;
#if defined(LIBRGA_IM2D_HANDLE)
    use_handle = 1;
#endif

    IM_STATUS ret_rga = IM_STATUS_NOERROR;
    im_rect srect;
    im_rect drect;
    im_rect prect;
    memset(&prect, 0, sizeof(prect));

    if (src_box != NULL) {
        srect.x = src_box->left;
        srect.y = src_box->top;
        srect.width = src_box->right - src_box->left + 1;
        srect.height = src_box->bottom - src_box->top + 1;
    } else {
        srect.x = 0;
        srect.y = 0;
        srect.width = srcWidth;
        srect.height = srcHeight;
    }

    if (dst_box != NULL) {
        drect.x = dst_box->left;
        drect.y = dst_box->top;
        drect.width = dst_box->right - dst_box->left + 1;
        drect.height = dst_box->bottom - dst_box->top + 1;
    } else {
        drect.x = 0;
        drect.y = 0;
        drect.width = dstWidth;
        drect.height = dstHeight;
    }

    rga_buffer_t rga_buf_src;
    rga_buffer_t rga_buf_dst;
    rga_buffer_t pat;
    rga_buffer_handle_t rga_handle_src = 0;
    rga_buffer_handle_t rga_handle_dst = 0;
    memset(&pat, 0, sizeof(rga_buffer_t));

    im_handle_param_t in_param;
    in_param.width = srcWidth;
    in_param.height = srcHeight;
    in_param.format = srcFmt;

    im_handle_param_t dst_param;
    dst_param.width = dstWidth;
    dst_param.height = dstHeight;
    dst_param.format = dstFmt;

    if (use_handle) {
        if (src_fd > 0) {
            rga_handle_src = importbuffer_fd(src_fd, &in_param);
        } else {
            rga_handle_src = importbuffer_virtualaddr(src, &in_param);
        }
        if (rga_handle_src <= 0) {
            return -1;
        }
        rga_buf_src = wrapbuffer_handle(
            rga_handle_src, srcWidth, srcHeight, srcFmt, srcWidth, srcHeight);
    } else {
        if (src_fd > 0) {
            rga_buf_src = wrapbuffer_fd(src_fd, srcWidth, srcHeight, srcFmt, srcWidth, srcHeight);
        } else {
            rga_buf_src = wrapbuffer_virtualaddr(src, srcWidth, srcHeight, srcFmt, srcWidth, srcHeight);
        }
    }

    if (use_handle) {
        if (dst_fd > 0) {
            rga_handle_dst = importbuffer_fd(dst_fd, &dst_param);
        } else {
            rga_handle_dst = importbuffer_virtualaddr(dst, &dst_param);
        }
        if (rga_handle_dst <= 0) {
            ret = -1;
            goto err;
        }
        rga_buf_dst = wrapbuffer_handle(
            rga_handle_dst, dstWidth, dstHeight, dstFmt, dstWidth, dstHeight);
    } else {
        if (dst_fd > 0) {
            rga_buf_dst = wrapbuffer_fd(dst_fd, dstWidth, dstHeight, dstFmt, dstWidth, dstHeight);
        } else {
            rga_buf_dst = wrapbuffer_virtualaddr(dst, dstWidth, dstHeight, dstFmt, dstWidth, dstHeight);
        }
    }

    if (drect.width != dstWidth || drect.height != dstHeight) {
        im_rect dst_whole_rect = {0, 0, dstWidth, dstHeight};
        int imcolor = color | (color << 8) | (color << 16) | (color << 24);
        ret_rga = imfill(rga_buf_dst, dst_whole_rect, imcolor);
        if (ret_rga <= 0 && dst != NULL) {
            size_t dst_size = (size_t)my_rga_get_image_size(dst_img);
            memset(dst, color, dst_size);
        }
    }

    ret_rga = improcess(rga_buf_src, rga_buf_dst, pat, srect, drect, prect, 0);
    if (ret_rga <= 0) {
        ret = -1;
    }

err:
    if (rga_handle_src > 0) {
        releasebuffer_handle(rga_handle_src);
    }
    if (rga_handle_dst > 0) {
        releasebuffer_handle(rga_handle_dst);
    }
    return ret;
}

int my_rga_convert_image(
    image_buffer_t* src_img,
    image_buffer_t* dst_img,
    image_rect_t* src_box,
    image_rect_t* dst_box,
    unsigned char color,
    int* used_rga)
{
    int ret;
    if (used_rga != NULL) {
        *used_rga = 0;
    }

    if (src_img->width % 16 == 0 && dst_img->width % 16 == 0) {
        ret = convert_image_rga(src_img, dst_img, src_box, dst_box, color);
        if (ret == 0) {
            if (used_rga != NULL) {
                *used_rga = 1;
            }
            return 0;
        }
    }

    ret = convert_image_cpu(src_img, dst_img, src_box, dst_box, color);
    return ret;
}

int my_rga_letterbox_bgr(
    image_buffer_t* src_image,
    image_buffer_t* dst_image,
    letterbox_t* letterbox,
    unsigned char fill_color,
    int* used_rga)
{
    int src_w = src_image->width;
    int src_h = src_image->height;
    int dst_w = dst_image->width;
    int dst_h = dst_image->height;

    float scale_w = (float)dst_w / (float)src_w;
    float scale_h = (float)dst_h / (float)src_h;
    float scale = scale_w < scale_h ? scale_w : scale_h;

    int resize_w = (int)(src_w * scale);
    int resize_h = (int)(src_h * scale);
    if (resize_w < 1) {
        resize_w = 1;
    }
    if (resize_h < 1) {
        resize_h = 1;
    }

    int offset_x = (dst_w - resize_w) / 2;
    int offset_y = (dst_h - resize_h) / 2;

    image_rect_t src_box;
    src_box.left = 0;
    src_box.top = 0;
    src_box.right = src_w - 1;
    src_box.bottom = src_h - 1;

    image_rect_t dst_box;
    dst_box.left = offset_x;
    dst_box.top = offset_y;
    dst_box.right = offset_x + resize_w - 1;
    dst_box.bottom = offset_y + resize_h - 1;

    if (letterbox != NULL) {
        letterbox->scale = scale;
        letterbox->x_pad = offset_x;
        letterbox->y_pad = offset_y;
    }

    return my_rga_convert_image(
        src_image, dst_image, &src_box, &dst_box, fill_color, used_rga);
}

int my_rga_bgr_to_rgb(
    image_buffer_t* src_bgr,
    image_buffer_t* dst_rgb,
    int* used_rga)
{
    if (src_bgr == NULL || dst_rgb == NULL || src_bgr->virt_addr == NULL || dst_rgb->virt_addr == NULL) {
        return -1;
    }
    if (used_rga != NULL) {
        *used_rga = 0;
    }

    int srcWidth = src_bgr->width;
    int srcHeight = src_bgr->height;
    int dstWidth = dst_rgb->width;
    int dstHeight = dst_rgb->height;

    if (srcWidth != dstWidth || srcHeight != dstHeight) {
        return -1;
    }

    if (srcWidth % 16 != 0) {
        for (int i = 0; i < srcWidth * srcHeight; i++) {
            dst_rgb->virt_addr[i * 3 + 0] = src_bgr->virt_addr[i * 3 + 2];
            dst_rgb->virt_addr[i * 3 + 1] = src_bgr->virt_addr[i * 3 + 1];
            dst_rgb->virt_addr[i * 3 + 2] = src_bgr->virt_addr[i * 3 + 0];
        }
        return 0;
    }

    rga_buffer_t src_img = wrapbuffer_virtualaddr(
        src_bgr->virt_addr, srcWidth, srcHeight, RK_FORMAT_BGR_888);
    rga_buffer_t dst_img = wrapbuffer_virtualaddr(
        dst_rgb->virt_addr, dstWidth, dstHeight, RK_FORMAT_RGB_888);

    IM_STATUS ret = imcvtcolor(
        src_img,
        dst_img,
        RK_FORMAT_BGR_888,
        RK_FORMAT_RGB_888,
        IM_COLOR_SPACE_DEFAULT,
        1);
    if (ret == IM_STATUS_SUCCESS) {
        if (used_rga != NULL) {
            *used_rga = 1;
        }
        return 0;
    }

    for (int i = 0; i < srcWidth * srcHeight; i++) {
        dst_rgb->virt_addr[i * 3 + 0] = src_bgr->virt_addr[i * 3 + 2];
        dst_rgb->virt_addr[i * 3 + 1] = src_bgr->virt_addr[i * 3 + 1];
        dst_rgb->virt_addr[i * 3 + 2] = src_bgr->virt_addr[i * 3 + 0];
    }
    return 0;
}
