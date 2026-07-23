#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <tuple>
#include <vector>

#include "image_utils_port.h"

namespace py = pybind11;

static py::array_t<uint8_t> ensure_bgr_contiguous(const py::array& arr)
{
    if (arr.ndim() != 3 || arr.shape(2) != 3) {
        throw std::invalid_argument("expected uint8 array with shape (H, W, 3)");
    }
    if (arr.dtype().kind() != 'u' || arr.dtype().itemsize() != 1) {
        throw std::invalid_argument("expected uint8 dtype");
    }
    return py::array_t<uint8_t, py::array::c_style | py::array::forcecast>(arr);
}

static py::tuple resize_bgr_impl(const py::array& input, int out_w, int out_h)
{
    py::array_t<uint8_t> src_arr = ensure_bgr_contiguous(input);
    auto src_info = src_arr.request();
    int src_h = static_cast<int>(src_info.shape[0]);
    int src_w = static_cast<int>(src_info.shape[1]);

    py::array_t<uint8_t> dst_arr({out_h, out_w, 3});
    auto dst_info = dst_arr.request();

    image_buffer_t src_img{};
    src_img.width = src_w;
    src_img.height = src_h;
    src_img.format = IMAGE_FORMAT_BGR888;
    src_img.virt_addr = static_cast<unsigned char*>(src_info.ptr);

    image_buffer_t dst_img{};
    dst_img.width = out_w;
    dst_img.height = out_h;
    dst_img.format = IMAGE_FORMAT_BGR888;
    dst_img.virt_addr = static_cast<unsigned char*>(dst_info.ptr);

    int used_rga = 0;
    int ret = my_rga_convert_image(&src_img, &dst_img, nullptr, nullptr, 0, &used_rga);
    if (ret != 0) {
        throw std::runtime_error("my_rga_convert_image failed");
    }
    return py::make_tuple(dst_arr, used_rga != 0);
}

static py::tuple letterbox_bgr_impl(
    const py::array& input,
    int canvas_w,
    int canvas_h,
    uint8_t fill_value)
{
    py::array_t<uint8_t> src_arr = ensure_bgr_contiguous(input);
    auto src_info = src_arr.request();
    int src_h = static_cast<int>(src_info.shape[0]);
    int src_w = static_cast<int>(src_info.shape[1]);

    py::array_t<uint8_t> dst_arr({canvas_h, canvas_w, 3});
    auto dst_info = dst_arr.request();

    image_buffer_t src_img{};
    src_img.width = src_w;
    src_img.height = src_h;
    src_img.format = IMAGE_FORMAT_BGR888;
    src_img.virt_addr = static_cast<unsigned char*>(src_info.ptr);

    image_buffer_t dst_img{};
    dst_img.width = canvas_w;
    dst_img.height = canvas_h;
    dst_img.format = IMAGE_FORMAT_BGR888;
    dst_img.virt_addr = static_cast<unsigned char*>(dst_info.ptr);

    letterbox_t lb{};
    int used_rga = 0;
    int ret = my_rga_letterbox_bgr(&src_img, &dst_img, &lb, fill_value, &used_rga);
    if (ret != 0) {
        throw std::runtime_error("my_rga_letterbox_bgr failed");
    }

    return py::make_tuple(dst_arr, lb.scale, lb.x_pad, lb.y_pad, used_rga != 0);
}

static py::tuple bgr_to_rgb_impl(const py::array& input)
{
    py::array_t<uint8_t> src_arr = ensure_bgr_contiguous(input);
    auto src_info = src_arr.request();
    int src_h = static_cast<int>(src_info.shape[0]);
    int src_w = static_cast<int>(src_info.shape[1]);

    py::array_t<uint8_t> dst_arr({src_h, src_w, 3});
    auto dst_info = dst_arr.request();

    image_buffer_t src_img{};
    src_img.width = src_w;
    src_img.height = src_h;
    src_img.format = IMAGE_FORMAT_BGR888;
    src_img.virt_addr = static_cast<unsigned char*>(src_info.ptr);

    image_buffer_t dst_img{};
    dst_img.width = src_w;
    dst_img.height = src_h;
    dst_img.format = IMAGE_FORMAT_RGB888;
    dst_img.virt_addr = static_cast<unsigned char*>(dst_info.ptr);

    int used_rga = 0;
    int ret = my_rga_bgr_to_rgb(&src_img, &dst_img, &used_rga);
    if (ret != 0) {
        throw std::runtime_error("my_rga_bgr_to_rgb failed");
    }

    return py::make_tuple(dst_arr, used_rga != 0);
}

PYBIND11_MODULE(my_rga, m)
{
    m.doc() = "my_rga: Rockchip RGA resize, letterbox, BGR->RGB (RK3568 aarch64)";

    m.def(
        "resize_bgr",
        [](const py::array& input, int out_w, int out_h) {
            return resize_bgr_impl(input, out_w, out_h);
        },
        py::arg("input"),
        py::arg("out_w"),
        py::arg("out_h"));

    m.def(
        "letterbox_bgr",
        [](const py::array& input, int canvas_w, int canvas_h, uint8_t fill_value) {
            return letterbox_bgr_impl(input, canvas_w, canvas_h, fill_value);
        },
        py::arg("input"),
        py::arg("canvas_w"),
        py::arg("canvas_h"),
        py::arg("fill_value") = 0);

    m.def(
        "bgr_to_rgb",
        [](const py::array& input) { return bgr_to_rgb_impl(input); },
        py::arg("input"));
}
