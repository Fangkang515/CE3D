import torch


def get_gradient_loss(
    video_frames,
    video_frames_dx,
    video_frames_dy,
    jif_current,
    model_F_mapping,
    model_F_atlas,
    rgb_output_pred,
    device,
    resx,
    number_of_frames,
):
    xplus1yt_foreground = torch.cat(
        (
            (jif_current[0, :] + 1) / (resx / 2) - 1,
            jif_current[1, :] / (resx / 2) - 1,
            jif_current[2, :] / (number_of_frames / 2.0) - 1,
        ),
        dim=1,
    ).to(device)

    xyplus1t_foreground = torch.cat(
        (
            (jif_current[0, :]) / (resx / 2) - 1,
            (jif_current[1, :] + 1) / (resx / 2) - 1,
            jif_current[2, :] / (number_of_frames / 2.0) - 1,
        ),
        dim=1,
    ).to(device)
    rgb_dx_gt = (
        video_frames_dx[jif_current[1, :], jif_current[0, :], :, jif_current[2, :]]
        .squeeze(1)
        .to(device)
    )
    rgb_dy_gt = (
        video_frames_dy[jif_current[1, :], jif_current[0, :], :, jif_current[2, :]]
        .squeeze(1)
        .to(device)
    )

    mapping_outs_xplus1 = model_F_mapping(xplus1yt_foreground)
    uv_foreground1_xplus1yt = mapping_outs_xplus1[..., :2]
    uv_foreground2_xplus1yt = mapping_outs_xplus1[..., 2:4]
    alphaxplus1 = 0.5 * (mapping_outs_xplus1[..., -1:] + 1.0)
    alphaxplus1 = alphaxplus1 * 0.99
    alphaxplus1 = alphaxplus1 + 0.001

    mapping_outs_yplus1 = model_F_mapping(xyplus1t_foreground)
    uv_foreground1_xyplus1t = mapping_outs_yplus1[..., :2]
    uv_foreground2_xyplus1t = mapping_outs_yplus1[..., 2:4]
    alphayplus1 = 0.5 * (mapping_outs_yplus1[..., -1:] + 1.0)
    alphayplus1 = alphayplus1 * 0.99
    alphayplus1 = alphayplus1 + 0.001

    def merge_uv_to_atlas(uv_foreground1, uv_foreground2):
        split_idx = uv_foreground1.shape[0]
        uv_datas = torch.cat(
            [uv_foreground1 * 0.25 + 0.75, uv_foreground2 * 0.25 + 0.25], dim=0
        )
        rgb_outputs = (model_F_atlas(uv_datas) + 1) * 0.5
        rgb_output1, rgb_output2 = (
            rgb_outputs[:split_idx, ...],
            rgb_outputs[split_idx:, ...],
        )
        return rgb_output1, rgb_output2

    # hash function of model_F_atlas's: inputs should be in range [0, 1]
    rgb_output1_xplus1yt = (
        model_F_atlas(uv_foreground1_xplus1yt * 0.25 + 0.75) + 1.0
    ) * 0.5
    rgb_output2_xplus1yt = (
        model_F_atlas(uv_foreground2_xplus1yt * 0.25 + 0.25) + 1.0
    ) * 0.5
    # rgb_output1_xplus1yt, rgb_output2_xplus1yt = merge_uv_to_atlas(uv_foreground1_xplus1yt, uv_foreground2_xplus1yt)

    rgb_output1_xyplus1t = (
        model_F_atlas(uv_foreground1_xyplus1t * 0.25 + 0.75) + 1.0
    ) * 0.5
    rgb_output2_xyplus1t = (
        model_F_atlas(uv_foreground2_xyplus1t * 0.25 + 0.25) + 1.0
    ) * 0.5
    # rgb_output1_xyplus1t, rgb_output2_xyplus1t = merge_uv_to_atlas(uv_foreground1_xyplus1t, uv_foreground2_xyplus1t)

    # Reconstructed RGB values:
    rgb_output_pred_xyplus1t = (
        rgb_output1_xyplus1t * alphayplus1 + rgb_output2_xyplus1t * (1.0 - alphayplus1)
    )
    rgb_output_pred_xplus1yt = (
        rgb_output1_xplus1yt * alphaxplus1 + rgb_output2_xplus1yt * (1.0 - alphaxplus1)
    )

    # Use reconstructed RGB values for computing derivatives:
    rgb_dx_output = rgb_output_pred_xplus1yt - rgb_output_pred
    rgb_dy_output = rgb_output_pred_xyplus1t - rgb_output_pred
    gradient_loss = torch.mean(
        (rgb_dx_gt - rgb_dx_output).norm(dim=1) ** 2
        + (rgb_dy_gt - rgb_dy_output).norm(dim=1) ** 2
    )

    return gradient_loss  # , rgb_loss


def get_rigidity_loss(
    jif_foreground,
    derivative_amount,
    resx,
    number_of_frames,
    model_F_mapping,
    uv_foreground,
    device,
    uv_mapping_scale=1.0,
    return_all=False,
    is_foreground=True,
):
    is_patch = (
        torch.cat((jif_foreground[1, :] - derivative_amount, jif_foreground[1, :]))
        / (resx / 2)
        - 1
    )
    js_patch = (
        torch.cat((jif_foreground[0, :], jif_foreground[0, :] - derivative_amount))
        / (resx / 2)
        - 1
    )
    fs_patch = (
        torch.cat((jif_foreground[2, :], jif_foreground[2, :]))
        / (number_of_frames / 2.0)
        - 1
    )
    xyt_p = torch.cat((js_patch, is_patch, fs_patch), dim=1).to(device)
    if is_foreground:
        uv_p = model_F_mapping(xyt_p)[..., :2]
    else:
        uv_p = model_F_mapping(xyt_p)[..., 2:4]
    u_p = uv_p[:, 0].view(
        2, -1
    )  # u_p[0,:]= u(x,y-derivative_amount,t).  u_p[1,:]= u(x-derivative_amount,y,t)
    v_p = uv_p[:, 1].view(
        2, -1
    )  # v_p[0,:]= u(x,y-derivative_amount,t).  v_p[1,:]= v(x-derivative_amount,y,t)

    u_p_d_ = (
        uv_foreground[:, 0].unsqueeze(0) - u_p
    )  # u_p_d_[0,:]=u(x,y,t)-u(x,y-derivative_amount,t)   u_p_d_[1,:]= u(x,y,t)-u(x-derivative_amount,y,t).
    v_p_d_ = (
        uv_foreground[:, 1].unsqueeze(0) - v_p
    )  # v_p_d_[0,:]=u(x,y,t)-v(x,y-derivative_amount,t).  v_p_d_[1,:]= u(x,y,t)-v(x-derivative_amount,y,t).

    # to match units: 1 in uv coordinates is resx/2 in image space.
    du_dx = u_p_d_[1, :] * resx / 2
    du_dy = u_p_d_[0, :] * resx / 2
    dv_dy = v_p_d_[0, :] * resx / 2
    dv_dx = v_p_d_[1, :] * resx / 2

    jacobians = torch.cat(
        (
            torch.cat(
                (du_dx.unsqueeze(-1).unsqueeze(-1), du_dy.unsqueeze(-1).unsqueeze(-1)),
                dim=2,
            ),
            torch.cat(
                (dv_dx.unsqueeze(-1).unsqueeze(-1), dv_dy.unsqueeze(-1).unsqueeze(-1)),
                dim=2,
            ),
        ),
        dim=1,
    )
    jacobians = jacobians / uv_mapping_scale
    jacobians = jacobians / derivative_amount

    JtJ = torch.matmul(jacobians.transpose(1, 2), jacobians)

    a = JtJ[:, 0, 0] + 0.001
    b = JtJ[:, 0, 1]
    c = JtJ[:, 1, 0]
    d = JtJ[:, 1, 1] + 0.001

    JTJinv = torch.zeros_like(jacobians).to(device)
    JTJinv[:, 0, 0] = d
    JTJinv[:, 0, 1] = -b
    JTJinv[:, 1, 0] = -c
    JTJinv[:, 1, 1] = a
    JTJinv = JTJinv / ((a * d - b * c).unsqueeze(-1).unsqueeze(-1))

    rigidity_loss = (JtJ**2).sum(1).sum(1).sqrt() + (JTJinv**2).sum(1).sum(1).sqrt()

    if return_all:
        return rigidity_loss
    else:
        return rigidity_loss.mean()


def get_optical_flow_loss_all(
    jif_foreground,
    uv_foreground,
    resx,
    number_of_frames,
    model_F_mapping,
    optical_flows,
    optical_flows_mask,
    uv_mapping_scale,
    device,
    alpha=1.0,
    is_foreground=True,
):
    xyt_foreground_forward_should_match, relevant_batch_indices_forward = (
        get_corresponding_flow_matches_all(
            jif_foreground, optical_flows_mask, optical_flows, resx, number_of_frames
        )
    )
    if is_foreground:
        uv_foreground_forward_should_match = model_F_mapping(
            xyt_foreground_forward_should_match.to(device)
        )[..., :2]
    else:
        uv_foreground_forward_should_match = model_F_mapping(
            xyt_foreground_forward_should_match.to(device)
        )[..., 2:4]

    errors = (uv_foreground_forward_should_match - uv_foreground).norm(dim=1)
    errors[relevant_batch_indices_forward == False] = 0
    errors = errors * (alpha.squeeze())

    return errors * resx / (2 * uv_mapping_scale)


def get_optical_flow_loss(
    jif_foreground,
    uv_foreground,
    optical_flows_reverse,
    optical_flows_reverse_mask,
    resx,
    number_of_frames,
    model_F_mapping,
    optical_flows,
    optical_flows_mask,
    uv_mapping_scale,
    device,
    use_alpha=False,
    alpha=1.0,
    is_foreground=True,
):
    # Forward flow:
    (
        uv_foreground_forward_relevant,
        xyt_foreground_forward_should_match,
        relevant_batch_indices_forward,
    ) = get_corresponding_flow_matches(
        jif_foreground,
        optical_flows_mask,
        optical_flows,
        resx,
        number_of_frames,
        True,
        uv_foreground,
    )
    if is_foreground:
        uv_foreground_forward_should_match = model_F_mapping(
            xyt_foreground_forward_should_match.to(device)
        )[..., :2]
    else:
        uv_foreground_forward_should_match = model_F_mapping(
            xyt_foreground_forward_should_match.to(device)
        )[..., 2:4]
    loss_flow_next = (
        (uv_foreground_forward_should_match - uv_foreground_forward_relevant).norm(
            dim=1
        )
        * resx
        / (2 * uv_mapping_scale)
    )

    # Backward flow:
    (
        uv_foreground_backward_relevant,
        xyt_foreground_backward_should_match,
        relevant_batch_indices_backward,
    ) = get_corresponding_flow_matches(
        jif_foreground,
        optical_flows_reverse_mask,
        optical_flows_reverse,
        resx,
        number_of_frames,
        False,
        uv_foreground,
    )
    if is_foreground:
        uv_foreground_backward_should_match = model_F_mapping(
            xyt_foreground_backward_should_match.to(device)
        )[..., :2]
    else:
        uv_foreground_backward_should_match = model_F_mapping(
            xyt_foreground_backward_should_match.to(device)
        )[..., 2:4]
    loss_flow_prev = (
        (uv_foreground_backward_should_match - uv_foreground_backward_relevant).norm(
            dim=1
        )
        * resx
        / (2 * uv_mapping_scale)
    )

    if use_alpha:
        flow_loss = (
            loss_flow_prev * alpha[relevant_batch_indices_backward].squeeze()
        ).mean() * 0.5 + (
            loss_flow_next * alpha[relevant_batch_indices_forward].squeeze()
        ).mean() * 0.5
    else:
        flow_loss = (loss_flow_prev).mean() * 0.5 + (loss_flow_next).mean() * 0.5

    return flow_loss


# A helper function for get_optical_flow_loss to return matching points according to the optical flow
def get_corresponding_flow_matches(
    jif_foreground,
    optical_flows_mask,
    optical_flows,
    resx,
    number_of_frames,
    is_forward,
    uv_foreground,
    use_uv=True,
):
    batch_forward_mask = torch.where(
        optical_flows_mask[
            jif_foreground[1, :].squeeze(),
            jif_foreground[0, :].squeeze(),
            jif_foreground[2, :].squeeze(),
            :,
        ]
    )
    forward_frames_amount = 2 ** batch_forward_mask[1]
    relevant_batch_indices = batch_forward_mask[0]
    jif_foreground_forward_relevant = jif_foreground[:, relevant_batch_indices, 0]
    forward_flows_for_loss = optical_flows[
        jif_foreground_forward_relevant[1],
        jif_foreground_forward_relevant[0],
        :,
        jif_foreground_forward_relevant[2],
        batch_forward_mask[1],
    ]

    if is_forward:
        jif_foreground_forward_should_match = torch.stack(
            (
                jif_foreground_forward_relevant[0] + forward_flows_for_loss[:, 0],
                jif_foreground_forward_relevant[1] + forward_flows_for_loss[:, 1],
                jif_foreground_forward_relevant[2] + forward_frames_amount,
            )
        )
    else:
        jif_foreground_forward_should_match = torch.stack(
            (
                jif_foreground_forward_relevant[0] + forward_flows_for_loss[:, 0],
                jif_foreground_forward_relevant[1] + forward_flows_for_loss[:, 1],
                jif_foreground_forward_relevant[2] - forward_frames_amount,
            )
        )

    xyt_foreground_forward_should_match = torch.stack(
        (
            jif_foreground_forward_should_match[0] / (resx / 2) - 1,
            jif_foreground_forward_should_match[1] / (resx / 2) - 1,
            jif_foreground_forward_should_match[2] / (number_of_frames / 2) - 1,
        )
    ).T
    if use_uv:
        uv_foreground_forward_relevant = uv_foreground[batch_forward_mask[0]]
        return (
            uv_foreground_forward_relevant,
            xyt_foreground_forward_should_match,
            relevant_batch_indices,
        )
    else:
        return xyt_foreground_forward_should_match, relevant_batch_indices


# A helper function for get_optical_flow_loss_all to return matching points according to the optical flow
def get_corresponding_flow_matches_all(
    jif_foreground,
    optical_flows_mask,
    optical_flows,
    resx,
    number_of_frames,
    use_uv=True,
):
    jif_foreground_forward_relevant = jif_foreground

    forward_flows_for_loss = optical_flows[
        jif_foreground_forward_relevant[1],
        jif_foreground_forward_relevant[0],
        :,
        jif_foreground_forward_relevant[2],
        0,
    ].squeeze()
    forward_flows_for_loss_mask = optical_flows_mask[
        jif_foreground_forward_relevant[1],
        jif_foreground_forward_relevant[0],
        jif_foreground_forward_relevant[2],
        0,
    ].squeeze()

    jif_foreground_forward_should_match = torch.stack(
        (
            jif_foreground_forward_relevant[0].squeeze() + forward_flows_for_loss[:, 0],
            jif_foreground_forward_relevant[1].squeeze() + forward_flows_for_loss[:, 1],
            jif_foreground_forward_relevant[2].squeeze() + 1,
        )
    )

    xyt_foreground_forward_should_match = torch.stack(
        (
            jif_foreground_forward_should_match[0] / (resx / 2) - 1,
            jif_foreground_forward_should_match[1] / (resx / 2) - 1,
            jif_foreground_forward_should_match[2] / (number_of_frames / 2) - 1,
        )
    ).T
    if use_uv:
        return xyt_foreground_forward_should_match, forward_flows_for_loss_mask > 0
    else:
        return 0


def get_optical_flow_alpha_loss(
    model_F_mapping,
    jif_foreground,
    alpha,
    optical_flows_reverse,
    optical_flows_reverse_mask,
    resx,
    number_of_frames,
    optical_flows,
    optical_flows_mask,
    device,
):
    # Forward flow
    xyt_foreground_forward_should_match, relevant_batch_indices_forward = (
        get_corresponding_flow_matches(
            jif_foreground,
            optical_flows_mask,
            optical_flows,
            resx,
            number_of_frames,
            True,
            0,
            use_uv=False,
        )
    )
    alpha_foreground_forward_should_match = model_F_mapping(
        xyt_foreground_forward_should_match.to(device)
    )[..., -1:]
    alpha_foreground_forward_should_match = 0.5 * (
        alpha_foreground_forward_should_match + 1.0
    )
    alpha_foreground_forward_should_match = alpha_foreground_forward_should_match * 0.99
    alpha_foreground_forward_should_match = (
        alpha_foreground_forward_should_match + 0.001
    )
    loss_flow_alpha_next = (
        (alpha[relevant_batch_indices_forward] - alpha_foreground_forward_should_match)
        .abs()
        .mean()
    )

    # Backward loss
    xyt_foreground_backward_should_match, relevant_batch_indices_backward = (
        get_corresponding_flow_matches(
            jif_foreground,
            optical_flows_reverse_mask,
            optical_flows_reverse,
            resx,
            number_of_frames,
            False,
            0,
            use_uv=False,
        )
    )
    alpha_foreground_backward_should_match = model_F_mapping(
        xyt_foreground_backward_should_match.to(device)
    )[..., -1:]
    alpha_foreground_backward_should_match = 0.5 * (
        alpha_foreground_backward_should_match + 1.0
    )
    alpha_foreground_backward_should_match = (
        alpha_foreground_backward_should_match * 0.99
    )
    alpha_foreground_backward_should_match = (
        alpha_foreground_backward_should_match + 0.001
    )
    loss_flow_alpha_prev = (
        (
            alpha_foreground_backward_should_match
            - alpha[relevant_batch_indices_backward]
        )
        .abs()
        .mean()
    )

    return (loss_flow_alpha_next + loss_flow_alpha_prev) * 0.5


def get_optical_flow_alpha_loss_all(
    model_F_mapping,
    jif_foreground,
    alpha,
    resx,
    number_of_frames,
    optical_flows,
    optical_flows_mask,
    device,
):
    xyt_foreground_forward_should_match, relevant_batch_indices_forward = (
        get_corresponding_flow_matches_all(
            jif_foreground, optical_flows_mask, optical_flows, resx, number_of_frames
        )
    )
    alpha_foreground_forward_should_match = model_F_mapping(
        xyt_foreground_forward_should_match.to(device)
    )[..., -1:]
    alpha_foreground_forward_should_match = 0.5 * (
        alpha_foreground_forward_should_match + 1.0
    )
    alpha_foreground_forward_should_match = alpha_foreground_forward_should_match * 0.99
    alpha_foreground_forward_should_match = (
        alpha_foreground_forward_should_match + 0.001
    )

    loss_flow_alpha_next = (alpha - alpha_foreground_forward_should_match).abs()
    loss_flow_alpha_next[relevant_batch_indices_forward == False] = 0

    return loss_flow_alpha_next
