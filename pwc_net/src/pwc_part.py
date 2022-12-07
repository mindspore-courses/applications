import mindspore
import mindspore.nn as nn
import mindspore.ops as P


class FeatureExtractor(nn.Cell):
    def __init__(self, num_channels):
        super(FeatureExtractor, self).__init__()
        self.num_channels = num_channels

        self.layers = nn.CellList()

        for _, (in_channel, out_channel) in enumerate(
            zip(num_channels[:-1], num_channels[1:])
        ):
            self.layers.append(
                nn.SequentialCell(
                    [
                        nn.Conv2d(
                            in_channel,
                            out_channel,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            has_bias=True,
                            pad_mode="pad",
                        ),
                        nn.LeakyReLU(0.1),
                        nn.Conv2d(
                            out_channel,
                            out_channel,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            has_bias=True,
                            pad_mode="pad",
                        ),
                        nn.LeakyReLU(0.1),
                    ]
                )
            )

    def construct(self, x):
        feature_pyramid = []
        for layer in self.layers:
            x = layer(x)
            feature_pyramid.append(x)
        feature_pyramid = feature_pyramid[::-1]
        return feature_pyramid


def get_warping_grid(x):
    b, h, w, _ = P.Shape()(x)
    bg, hg, wg = P.Meshgrid(indexing="ij")((nn.Range(b)(), nn.Range(h)(), nn.Range(w)()))
    return bg, hg, wg


def nearest_warping(x, flow):
    bg, hg, wg = get_warping_grid(x)
    flow = flow.astype("Int32")

    warped_gx = P.Add()(wg, flow[:, :, :, 0])
    warped_gy = P.Add()(hg, flow[:, :, :, 1])
    _, h, w, _ = P.shape()(x)
    warped_gx = P.clip_by_value(warped_gx, 0, w - 1)
    warped_gy = P.clip_by_value(warped_gy, 0, h - 1)
    warped_indices = P.Stack(3)([bg, warped_gy, warped_gx])

    warped_x = P.GatherNd()(x, warped_indices)
    return warped_x


def bilinear_warp(x, flow):
    _, h, w, _ = P.Shape()(x)
    bg, hg, wg = get_warping_grid(x)
    bg = bg.astype("float32")
    hg = hg.astype("float32")
    wg = wg.astype("float32")

    fx = flow[..., 0]
    fy = flow[..., 1]
    fx_0, fx_1 = P.Floor()(fx), P.Ceil()(fx)
    fy_0, fy_1 = P.Floor()(fy), P.Ceil()(fy)

    h_lim = h - 1
    w_lim = w - 1

    gx_0 = P.clip_by_value(wg + fx_0, 0, w_lim)
    gx_1 = P.clip_by_value(wg + fx_1, 0, w_lim)
    gy_0 = P.clip_by_value(hg + fy_0, 0, h_lim)
    gy_1 = P.clip_by_value(hg + fy_1, 0, h_lim)

    g_00 = P.Stack(3)([bg, gy_0, gx_0]).astype("Int32")
    g_01 = P.Stack(3)([bg, gy_0, gx_1]).astype("Int32")
    g_10 = P.Stack(3)([bg, gy_1, gx_0]).astype("Int32")
    g_11 = P.Stack(3)([bg, gy_1, gx_1]).astype("Int32")

    I_00 = P.GatherNd()(x, g_00)
    I_01 = P.GatherNd()(x, g_01)
    I_10 = P.GatherNd()(x, g_10)
    I_11 = P.GatherNd()(x, g_11)

    w_00 = P.Sub()(gx_1, fx) * P.Sub()(gy_1, fy)
    w_01 = P.Sub()(fx, gx_0) * P.Sub()(gy_1, fy)
    w_10 = P.Sub()(gx_1, fx) * P.Sub()(fy, gy_0)
    w_11 = P.Sub()(fx, gx_0) * P.Sub()(fy, gy_0)

    warped_x = P.AddN()(
        [
            P.Mul()(I_00, w_00[..., None]),
            P.Mul()(I_01, w_01[..., None]),
            P.Mul()(I_10, w_10[..., None]),
            P.Mul()(I_11, w_11[..., None]),
        ]
    )

    return warped_x


class WarpingLayer(nn.Cell):
    def __init__(self, warp_type="nearest"):
        super(WarpingLayer, self).__init__()
        self.warp_type = warp_type

    def construct(self, x, flow):
        x = P.Transpose()(x, (0, 2, 3, 1))
        flow = P.Transpose()(flow, (0, 2, 3, 1))
        if self.warp_type == "nearest":
            warped_x = nearest_warping(x, flow)
        elif self.warp_type == "bilinear":
            warped_x = bilinear_warp(x, flow)
        else:
            raise NotImplementedError
        warped_x = P.Transpose()(warped_x, (0, 3, 1, 2))
        return warped_x


class FlowEstimator(nn.Cell):
    def __init__(self, in_channel):
        super(FlowEstimator, self).__init__()

        self.layer = nn.SequentialCell(
            [
                nn.Conv2d(
                    in_channel,
                    128,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    has_bias=True,
                    pad_mode="pad",
                ),
                nn.LeakyReLU(0.1),
                nn.Conv2d(
                    128,
                    128,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    has_bias=True,
                    pad_mode="pad",
                ),
                nn.LeakyReLU(0.1),
                nn.Conv2d(
                    128,
                    96,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    has_bias=True,
                    pad_mode="pad",
                ),
                nn.LeakyReLU(0.1),
                nn.Conv2d(
                    96,
                    64,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    has_bias=True,
                    pad_mode="pad",
                ),
                nn.LeakyReLU(0.1),
                nn.Conv2d(
                    64,
                    32,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    has_bias=True,
                    pad_mode="pad",
                ),
            ]
        )
        self.predict_flow = nn.Conv2d(
            32, 2, kernel_size=3, stride=1, padding=1, has_bias=True, pad_mode="pad"
        )

    def construct(self, x):
        intermediate_feat = self.layer(x)
        flow = self.predict_flow(intermediate_feat)
        return intermediate_feat, flow


class DenseFlowEstimator(nn.Cell):
    def __init__(self, in_channel):
        super(DenseFlowEstimator, self).__init__()
        self.layer_1 = nn.SequentialCell(
            [
                nn.Conv2d(
                    in_channel,
                    128,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    has_bias=True,
                    pad_mode="pad",
                ),
                nn.LeakyReLU(0.1),
            ]
        )
        self.layer_2 = nn.SequentialCell(
            [
                nn.Conv2d(
                    in_channel + 128,
                    128,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    has_bias=True,
                    pad_mode="pad",
                ),
                nn.LeakyReLU(0.1),
            ]
        )
        self.layer_3 = nn.SequentialCell(
            [
                nn.Conv2d(
                    in_channel + 128 + 128,
                    96,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    has_bias=True,
                    pad_mode="pad",
                ),
                nn.LeakyReLU(0.1),
            ]
        )
        self.layer_4 = nn.SequentialCell(
            [
                nn.Conv2d(
                    in_channel + 128 + 128 + 96,
                    64,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    has_bias=True,
                    pad_mode="pad",
                ),
                nn.LeakyReLU(0.1),
            ]
        )
        self.layer_5 = nn.SequentialCell(
            [
                nn.Conv2d(
                    in_channel + 128 + 128 + 96 + 64,
                    32,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    has_bias=True,
                    pad_mode="pad",
                ),
            ]
        )
        self.predict_flow = nn.Conv2d(
            in_channel + 128 + 128 + 96 + 64 + 32,
            2,
            kernel_size=3,
            stride=1,
            padding=1,
            has_bias=True,
            pad_mode="pad",
        )

    def construct(self, x):
        x1 = P.Concat(1)([x, self.layer_1(x)])
        x2 = P.Concat(1)([x1, self.layer_2(x1)])
        x3 = P.Concat(1)([x2, self.layer_3(x2)])
        x4 = P.Concat(1)([x3, self.layer_4(x3)])
        x5 = P.Concat(1)([x4, self.layer_5(x4)])
        flow = self.predict_flow(x5)
        return x5, flow


class ContextNetwork(nn.Cell):
    def __init__(self, in_channel):
        super(ContextNetwork, self).__init__()

        self.layer = nn.SequentialCell(
            [
                nn.Conv2d(
                    in_channel,
                    128,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    has_bias=True,
                    pad_mode="pad",
                ),
                nn.LeakyReLU(0.1),
                nn.Conv2d(
                    128,
                    128,
                    kernel_size=3,
                    stride=1,
                    dilation=2,
                    padding=2,
                    has_bias=True,
                    pad_mode="pad",
                ),
                nn.LeakyReLU(0.1),
                nn.Conv2d(
                    128,
                    128,
                    kernel_size=3,
                    stride=1,
                    dilation=4,
                    padding=4,
                    has_bias=True,
                    pad_mode="pad",
                ),
                nn.LeakyReLU(0.1),
                nn.Conv2d(
                    128,
                    96,
                    kernel_size=3,
                    stride=1,
                    dilation=8,
                    padding=8,
                    has_bias=True,
                    pad_mode="pad",
                ),
                nn.LeakyReLU(0.1),
                nn.Conv2d(
                    96,
                    64,
                    kernel_size=3,
                    stride=1,
                    dilation=16,
                    padding=16,
                    has_bias=True,
                    pad_mode="pad",
                ),
                nn.LeakyReLU(0.1),
                nn.Conv2d(
                    64,
                    32,
                    kernel_size=3,
                    stride=1,
                    dilation=1,
                    padding=1,
                    has_bias=True,
                    pad_mode="pad",
                ),
                nn.LeakyReLU(0.1),
                nn.Conv2d(
                    32,
                    2,
                    kernel_size=3,
                    stride=1,
                    dilation=1,
                    padding=1,
                    has_bias=True,
                    pad_mode="pad",
                ),
            ]
        )

    def construct(self, x):
        return self.layer(x)
