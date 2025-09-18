from torch import Tensor


class Gaussians:

    def __init__(
        self,
        means=None,
        covariances=None,
        harmonics=None,
        opacities=None,
        scales=None,
        rotations=None,
        **kwargs
    ):
        self.means: Tensor = means
        self.covariances: Tensor = covariances
        self.harmonics: Tensor = harmonics
        self.opacities: Tensor = opacities
        self.scales: Tensor = scales
        self.rotations: Tensor = rotations
        for key, value in kwargs.items():
            setattr(self, key, value)

    def detach_cpu_copy(self):
        # get all attributes of the class, including dynamic added
        all_fields = vars(self)
        copy_gaussians = Gaussians()
        # iterate over all fields
        for field_name, field_value in all_fields.items():
            # check if the field is a tensor
            if isinstance(field_value, Tensor):
                # detach and copy the tensor
                copy_gaussians.__setattr__(field_name, field_value.detach().cpu())
            else:
                # if not tensor, just copy the value
                copy_gaussians.__setattr__(field_name, field_value)
        return copy_gaussians
