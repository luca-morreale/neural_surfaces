
import torch


class Interpolation():

    def init_interpolation(self):
        if torch.cuda.is_available():
            self.dtype = torch.cuda.FloatTensor
            self.dtype_long = torch.cuda.LongTensor
        else:
            self.dtype = torch.FloatTensor
            self.dtype_long = torch.LongTensor

    def get_interpolation_coordinates(self, uvs, image_feat):
        ## discretize the uv coordiantes based on image size

        H = image_feat.size(-1)

        coords = uvs + 1.0 # push uvs into positive space
        x = coords[..., 0] * H / 2.0
        y = coords[..., 1] * H / 2.0
        x0 = x.type(self.dtype_long) # relative index on height
        y0 = y.type(self.dtype_long)

        x1 = x0 + 1
        y1 = y0 + 1

        # bound indicies within image size
        x0 = torch.clamp(x0, 0, H-2)
        x1 = torch.clamp(x1, 1, H-1)
        y0 = torch.clamp(y0, 0, H-2)
        y1 = torch.clamp(y1, 1, H-1)

        return x, y, x0, x1, y0, y1

    def get_interpolation_features(self, x0, x1, y0, y1, image_feat):
        ## get features to interpolate
        Ia = image_feat[:, y0, x0 ]
        Ib = image_feat[:, y1, x0 ]
        Ic = image_feat[:, y0, x1 ]
        Id = image_feat[:, y1, x1 ]

        return Ia, Ib, Ic, Id

    def get_interpolation_weights(self, x, y, x0, x1, y0, y1):
        ## compute bilinear interpolation weights
        wa = (x1.type(self.dtype) - x) * (y1.type(self.dtype) - y)
        wb = (x1.type(self.dtype) - x) * (y - y0.type(self.dtype))
        wc = (x - x0.type(self.dtype)) * (y1.type(self.dtype) - y)
        wd = (x - x0.type(self.dtype)) * (y - y0.type(self.dtype))
        return wa, wb, wc, wd


    def interpolate(self, image_feat, uvs):
        image_feat = image_feat.squeeze(0)
        x, y, x0, x1, y0, y1 = self.get_interpolation_coordinates(uvs, image_feat)
        Ia, Ib, Ic, Id = self.get_interpolation_features(x0, x1, y0, y1, image_feat)
        wa, wb, wc, wd = self.get_interpolation_weights(x, y, x0, x1, y0, y1)

        interpolated_feat = wa * Ia + wb * Ib + wc * Ic + wd * Id
        return interpolated_feat.t()



class BatchInterpolation(Interpolation):
    ## this is the same as Interpolation
    ## except works on a batch of feature images
    ## at the same time (ie multiple patches)

    def get_interpolation_features(self, x0, x1, y0, y1, image_feat):

        batch_index = torch.arange(x0.size(0)).unsqueeze(1).repeat(1, x0.size(1)).view(-1)

        Ia = image_feat[batch_index, ..., y0.view(-1), x0.view(-1) ].transpose(1,0)
        Ib = image_feat[batch_index, ..., y1.view(-1), x0.view(-1) ].transpose(1,0)
        Ic = image_feat[batch_index, ..., y0.view(-1), x1.view(-1) ].transpose(1,0)
        Id = image_feat[batch_index, ..., y1.view(-1), x1.view(-1) ].transpose(1,0)

        return Ia, Ib, Ic, Id


    def interpolate(self, image_feat, uvs):
        # image_feat = image_feat.squeeze(0)
        x, y, x0, x1, y0, y1 = self.get_interpolation_coordinates(uvs, image_feat)
        Ia, Ib, Ic, Id = self.get_interpolation_features(x0, x1, y0, y1, image_feat)
        wa, wb, wc, wd = self.get_interpolation_weights(x.squeeze(), y.squeeze(), x0.squeeze(), x1.squeeze(), y0.squeeze(), y1.squeeze())

        interpolated_feat = wa.view(-1) * Ia + wb.view(-1) * Ib + wc.view(-1) * Ic + wd.view(-1) * Id
        final_feats = interpolated_feat.t().reshape(uvs.size(0), uvs.size(1), -1)

        return final_feats
