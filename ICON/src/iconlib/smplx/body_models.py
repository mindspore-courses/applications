"""smpl model"""
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
from typing import Optional, Dict, Union
import os.path as osp
import os
import pickle
from collections import namedtuple
import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor
import numpy as np

from .utils import Struct, Array, to_np, to_tensor, SMPLOutput
from .vertex_ids import vertex_ids as VERTEX_IDS
from .vertex_joint_selector import VertexJointSelector
from .libs import lbs


ModelOutput = namedtuple(
    "ModelOutput",
    [
        "vertices",
        "joints",
        "full_pose",
        "betas",
        "global_orient",
        "body_pose",
        "expression",
        "left_hand_pose",
        "right_hand_pose",
        "jaw_pose",
    ],
)
ModelOutput.__new__.__defaults__ = (None,) * len(ModelOutput._fields)


class SMPL(nn.Cell):
    """smpl model"""

    NUM_JOINTS = 23
    NUM_BODY_JOINTS = 23
    SHAPE_SPACE_DIM = 300

    def __init__(
            self,
            model_path: str,
            kid_template_path: str = "",
            data_struct: Optional[Struct] = None,
            create_betas: bool = True,
            betas: Optional[Tensor] = None,
            num_betas: int = 10,
            create_global_orient: bool = True,
            global_orient: Optional[Tensor] = None,
            create_body_pose: bool = True,
            body_pose: Optional[Tensor] = None,
            create_transl: bool = True,
            transl: Optional[Tensor] = None,
            dtype=ms.float32,
            batch_size: int = 1,
            joint_mapper=None,
            gender: str = "neutral",
            age: str = "adult",
            vertex_ids: Dict[str, int] = None,
            v_template: Optional[Union[Tensor, Array]] = None,
            v_personal: Optional[Union[Tensor, Array]] = None,
            **kwargs,
    ) -> None:
        """ SMPL model constructor

            Parameters
            ----------
            model_path: str
                The path to the folder or to the file where the model
                parameters are stored
            data_struct: Strct
                A struct object. If given, then the parameters of the model are
                read from the object. Otherwise, the model tries to read the
                parameters from the given `model_path`. (default = None)
            create_global_orient: bool, optional
                Flag for creating a member variable for the global orientation
                of the body. (default = True)
            global_orient: mindspore.tensor, optional, Bx3
                The default value for the global orientation variable.
                (default = None)
            create_body_pose: bool, optional
                Flag for creating a member variable for the pose of the body.
                (default = True)
            body_pose: mindspore.tensor, optional, Bx(Body Joints * 3)
                The default value for the body pose variable.
                (default = None)
            num_betas: int, optional
                Number of shape components to use
                (default = 10).
            create_betas: bool, optional
                Flag for creating a member variable for the shape space
                (default = True).
            betas: mindspore.tensor, optional, Bx10
                The default value for the shape member variable.
                (default = None)
            create_transl: bool, optional
                Flag for creating a member variable for the translation
                of the body. (default = True)
            transl: mindspore.tensor, optional, Bx3
                The default value for the transl variable.
                (default = None)
            dtype: mindspore.dtype, optional
                The data type for the created variables
            batch_size: int, optional
                The batch size used for creating the member variables
            joint_mapper: object, optional
                An object that re-maps the joints. Useful if one wants to
                re-order the SMPL joints to some other convention (e.g. MSCOCO)
                (default = None)
            gender: str, optional
                Which gender to load
            vertex_ids: dict, optional
                A dictionary containing the indices of the extra vertices that
                will be selected
        """

        self.gender = gender
        self.age = age

        if data_struct is None:
            if osp.isdir(model_path):
                model_fn = "SMPL_{}.{ext}".format(gender.upper(), ext="pkl")
                smpl_path = os.path.join(model_path, model_fn)
            else:
                smpl_path = model_path
            assert osp.exists(smpl_path), "Path {} does not exist!".format(smpl_path)

            with open(smpl_path, "rb") as smpl_file:
                data_struct = Struct(**pickle.load(smpl_file, encoding="latin1"))

        super(SMPL, self).__init__()
        self.batch_size = batch_size
        shapedirs = data_struct.shapedirs
        if shapedirs.shape[-1] < self.SHAPE_SPACE_DIM:
            # print(f'WARNING: You are using a {self.name()} model, with only'
            #       ' 10 shape coefficients.')
            num_betas = min(num_betas, 10)
        else:
            num_betas = min(num_betas, self.SHAPE_SPACE_DIM)

        if self.age == "kid":
            v_template_smil = np.load(kid_template_path)
            v_template_smil -= np.mean(v_template_smil, axis=0)
            v_template_diff = np.expand_dims(
                v_template_smil - data_struct.v_template, axis=2
            )
            shapedirs = np.concatenate(
                (shapedirs[:, :, :num_betas], v_template_diff), axis=2
            )
            num_betas = num_betas + 1

        self._num_betas = num_betas
        shapedirs = shapedirs[:, :, :num_betas]
        shapedirs = np.load("shapedirs.npy")
        # The shape components
        self.shapedirs = ms.Parameter(
            to_tensor(shapedirs, dtype=dtype), requires_grad=False
        )

        if vertex_ids is None:
            # SMPL and SMPL-H share the same topology, so any extra joints can
            # be drawn from the same place
            vertex_ids = VERTEX_IDS["smplh"]

        self.dtype = dtype

        self.joint_mapper = joint_mapper

        self.vertex_joint_selector = VertexJointSelector(
            vertex_ids=vertex_ids, **kwargs
        )

        self.faces = data_struct.f
        self.faces_tensor = ms.Parameter(
            to_tensor(to_np(self.faces, dtype=np.int64), dtype=ms.int64),
            requires_grad=False,
        )

        if create_betas:
            if betas is None:
                zeros = ms.ops.Zeros()
                default_betas = zeros((batch_size, self.num_betas), dtype)
            else:
                if isinstance((betas), Tensor):
                    default_betas = ms.ops.stop_gradient(betas.copy())
                else:
                    default_betas = Tensor(betas, dtype=dtype)

            self.betas = ms.Parameter(default_betas, requires_grad=True)

        # The tensor that contains the global rotation of the model
        # It is separated from the pose of the joints in case we wish to
        # optimize only over one of them
        zeros = ms.ops.Zeros()
        if create_global_orient:
            if global_orient is None:
                default_global_orient = zeros((batch_size, 3), dtype)
            else:
                if isinstance((global_orient), Tensor):
                    default_global_orient = ms.ops.stop_gradient(global_orient.copy())
                else:
                    default_global_orient = Tensor(global_orient, dtype=dtype)

            self.global_orient = ms.Parameter(default_global_orient, requires_grad=True)

        if create_body_pose:
            if body_pose is None:
                default_body_pose = zeros((batch_size, self.NUM_BODY_JOINTS * 3), dtype)
            else:
                if isinstance((body_pose), Tensor):
                    default_body_pose = ms.ops.stop_gradient(body_pose.copy())
                else:
                    default_body_pose = Tensor(body_pose, dtype=dtype)

            self.body_pose = ms.Parameter(default_body_pose, requires_grad=True)

        if create_transl:
            if transl is None:
                default_transl = zeros((batch_size, 3), dtype)
            else:
                default_transl = Tensor(transl, dtype=dtype)

            self.transl = ms.Parameter(default_transl, requires_grad=True)

        if v_template is None:
            v_template = data_struct.v_template

        if not isinstance((v_template), Tensor):
            v_template = to_tensor(to_np(v_template), dtype=dtype)

        if v_personal is not None:
            v_personal = to_tensor(to_np(v_personal), dtype=dtype)
            v_template += v_personal

        # The vertices of the template model
        v_template = np.load("v_template.npy")
        self.v_template = ms.Parameter(Tensor(v_template), requires_grad=False)

        j_regressor = to_tensor(to_np(data_struct.J_regressor), dtype=dtype)
        j_regressor = np.load("J_regressor.npy")
        self.j_regressor = ms.Parameter(Tensor(j_regressor), requires_grad=False)

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*3 x 207
        num_pose_basis = data_struct.posedirs.shape[-1]
        # 207 x 20670
        posedirs = np.reshape(data_struct.posedirs, [-1, num_pose_basis]).T
        posedirs = np.load("posedirs.npy")
        self.posedirs = ms.Parameter(
            to_tensor(to_np(posedirs), dtype=dtype), requires_grad=False
        )

        # indices of parents for each joints
        parents = to_tensor(to_np(data_struct.kintree_table[0]), dtype=ms.int64)
        parents[0] = -1
        # parents = np.load("parents.npy")
        self.parents = ms.Parameter(parents, requires_grad=False)
        lbs_weights = np.load("lbs_weights.npy")
        self.lbs_weights = ms.Parameter(
            to_tensor(lbs_weights, dtype=dtype), requires_grad=False
        )

    @property
    def num_betas(self):
        """num betas"""
        return self._num_betas

    @property
    def num_expression_coeffs(self):
        """num_expression_coeffs"""
        return 0

    def name(self) -> str:
        """name"""
        return "SMPL"

    def reset_params(self, **params_dict) -> None:
        """reset_params"""
        for param_name, param in self.named_parameters():
            if param_name in params_dict:
                param[:] = ms.ops.stop_gradient(Tensor(params_dict[param_name]))
            else:
                param.fill_(0)

    def get_num_verts(self) -> int:
        """get_num_verts"""
        return self.v_template.shape[0]

    def get_num_faces(self) -> int:
        """get_num_faces"""
        return self.faces.shape[0]

    def extra_repr(self) -> str:
        """extra_repr"""
        msg = [
            f"Gender: {self.gender.upper()}",
            f"Number of joints: {self.J_regressor.shape[0]}",
            f"Betas: {self.num_betas}",
        ]
        return "\n".join(msg)

    def construct(
            self,
            betas: Optional[Tensor] = None,
            body_pose: Optional[Tensor] = None,
            global_orient: Optional[Tensor] = None,
            transl: Optional[Tensor] = None,
            return_verts=True,
            return_full_pose: bool = False,
            pose2rot: bool = True,
    ) -> SMPLOutput:
        """ Forward pass for the SMPL model

            Parameters
            ----------
            global_orient: mindspore.tensor, optional, shape Bx3
                If given, ignore the member variable and use it as the global
                rotation of the body. Useful if someone wishes to predicts this
                with an external model. (default=None)
            betas: mindspore.tensor, optional, shape BxN_b
                If given, ignore the member variable `betas` and use it
                instead. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            body_pose: mindspore.tensor, optional, shape Bx(J*3)
                If given, ignore the member variable `body_pose` and use it
                instead. For example, it can used if someone predicts the
                pose of the body joints are predicted from some external model.
                It should be a tensor that contains joint rotations in
                axis-angle format. (default=None)
            transl: mindspore.tensor, optional, shape Bx3
                If given, ignore the member variable `transl` and use it
                instead. For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            return_verts: bool, optional
                Return the vertices. (default=True)
            return_full_pose: bool, optional
                Returns the full axis-angle pose vector (default=False)

            Returns
            -------
        """
        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = (
            global_orient if global_orient is not None else self.global_orient
        )
        body_pose = body_pose if body_pose is not None else self.body_pose
        betas = betas if betas is not None else self.betas

        apply_trans = transl is not None or hasattr(self, "transl")
        if transl is None and hasattr(self, "transl"):
            transl = self.transl

        body_pose = Tensor(body_pose.asnumpy())
        global_orient = Tensor(global_orient.asnumpy())
        full_pose = ms.ops.concat([global_orient, body_pose], axis=1)

        batch_size = max(betas.shape[0], global_orient.shape[0], body_pose.shape[0])

        if betas.shape[0] != batch_size:
            num_repeats = int(batch_size / betas.shape[0])
            betas = ms.ops.broadcast_to(betas, (num_repeats, -1))

        vertices, joints = lbs(
            betas,
            full_pose,
            self.v_template,
            self.shapedirs,
            self.posedirs,
            self.j_regressor,
            self.parents,
            self.lbs_weights,
            pose2rot=pose2rot,
        )

        joints = self.vertex_joint_selector.construct(vertices, joints)
        # Map the joints to the current dataset
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if apply_trans:
            expand_dims = ms.ops.ExpandDims()
            joints += expand_dims(transl, 1)
            vertices += expand_dims(transl, 1)

        output = SMPLOutput(
            vertices=vertices if return_verts else None,
            global_orient=global_orient,
            body_pose=body_pose,
            joints=joints,
            betas=betas,
            full_pose=full_pose if return_full_pose else None,
        )

        return output
