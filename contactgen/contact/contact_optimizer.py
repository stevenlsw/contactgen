import torch
from kornia.geometry.linalg import inverse_transformation
from manopth import rodrigues_layer
from .opt_utils import compute_uv, compute_uv_loss


def optimize_pose(model, mano_layer, obj_verts, obj_cmap, obj_partition, obj_uv,
                  w_contact=1e-1, w_pene=3.0, w_uv=1e-2, w_pose_reg=1e-2, w_shape_reg=1e-2,
                  global_iter=200, pose_iter=1000,
                  global_lr=5e-2, pose_lr=5e-3, eps=-1e-3):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    batch_size = obj_verts.shape[0]
    global_pose = torch.zeros((batch_size, 3), dtype=obj_verts.dtype, device=obj_verts.device)
    mano_trans = torch.zeros((batch_size, 3), dtype=obj_verts.dtype, device=obj_verts.device)
    
    mano_pose = torch.zeros((batch_size, mano_layer.ncomps), dtype=obj_verts.dtype, device=obj_verts.device)
    mano_shape = torch.zeros((batch_size, 10), dtype=obj_verts.dtype, device=obj_verts.device)

    mano_pose.requires_grad = False
    mano_shape.requires_grad = False
    global_pose.requires_grad = True
    mano_trans.requires_grad = True 
    hand_opt_params = [global_pose, mano_trans]
    global_optimizer = torch.optim.Adam(hand_opt_params, lr=global_lr)
    
    for it in range(global_iter):
        loss_info = ""
        loss = 0

        _, frames = mano_layer(torch.cat((torch.zeros_like(global_pose, device=global_pose.device, dtype=global_pose.dtype), mano_pose), dim=1),
                               th_betas=mano_shape, th_trans=torch.zeros_like(mano_trans, device=mano_trans.device, dtype=mano_trans.dtype))
        inv_trans = inverse_transformation(frames.reshape(-1, 4, 4)).reshape(batch_size, -1, 4, 4)
        joints = frames[:, :, :3, 3]
        inv_trans_mat = inv_trans
        root = joints[:, 0, :]

        global_rotation = rodrigues_layer.batch_rodrigues(global_pose).reshape(batch_size, 3, 3)
        query_pnts_cano = torch.matmul(obj_verts - root.unsqueeze(dim=1) - mano_trans.unsqueeze(dim=1), global_rotation) + root.unsqueeze(dim=1)
        pnts = model.transform_queries(query_pnts_cano, inv_trans_mat)
        pnts = model.add_pose_feature(pnts, root, inv_trans_mat)
        pnts = model.add_shape_feature(queries=pnts, shape_indices=None, latent_shape_code=mano_shape)
        pred, pred_p_full = model.forward(pnts)
        
        pred_p = torch.gather(pred_p_full, dim=2, index=obj_partition.unsqueeze(dim=-1)).squeeze(-1)
        loss_contact = w_contact * (torch.abs(pred_p) * obj_cmap).sum(dim=-1).mean(dim=0)
        loss += loss_contact
        loss_info += "contact loss: {:.3f} | ".format(loss_contact.item())

        _, frames = mano_layer(torch.cat((global_pose, mano_pose), dim=1), th_betas=mano_shape, th_trans=mano_trans)
        uv_pred = compute_uv(frames, obj_verts, obj_partition)
        uv_loss = w_uv * compute_uv_loss(uv_pred, obj_uv, weight=1.0+obj_cmap)
        loss += uv_loss
        loss_info += "uv loss: {:.3f}".format(uv_loss.item())

        global_optimizer.zero_grad()
        loss.backward()
        global_optimizer.step()
        print("global iter {} | ".format(it) + loss_info)

    mano_pose.requires_grad = True
    mano_shape.requires_grad = True
    global_pose.requires_grad = True
    mano_trans.requires_grad = True
    hand_opt_params = [global_pose, mano_pose, mano_shape, mano_trans]
    pose_optimizer = torch.optim.Adam(hand_opt_params, lr=pose_lr)
    
    for it in range(pose_iter):
        loss_info = ""
        loss = 0
        _, frames = mano_layer(
            torch.cat((torch.zeros_like(global_pose, device=global_pose.device, dtype=global_pose.dtype), mano_pose),
                      dim=1),
            th_betas=mano_shape,
            th_trans=torch.zeros_like(mano_trans, device=mano_trans.device, dtype=mano_trans.dtype))
        inv_trans = inverse_transformation(frames.reshape(-1, 4, 4)).reshape(batch_size, -1, 4, 4)
        joints = frames[:, :, :3, 3]
        inv_trans_mat = inv_trans
        root = joints[:, 0, :]

        global_rotation = rodrigues_layer.batch_rodrigues(global_pose).reshape(batch_size, 3, 3)
        query_pnts_cano = torch.matmul(obj_verts - root.unsqueeze(dim=1) - mano_trans.unsqueeze(dim=1), global_rotation) + root.unsqueeze(dim=1)
        pnts = model.transform_queries(query_pnts_cano, inv_trans_mat)
        pnts = model.add_pose_feature(pnts, root, inv_trans_mat)
        pnts = model.add_shape_feature(queries=pnts, shape_indices=None, latent_shape_code=mano_shape)
        pred, pred_p_full = model.forward(pnts)
        pred_p = torch.gather(pred_p_full, dim=2, index=obj_partition.unsqueeze(dim=-1)).squeeze(-1)  # (B, Q)
        loss_contact = w_contact * (torch.abs(pred_p) * obj_cmap).sum(dim=-1).mean(dim=0) 
        loss += loss_contact
        loss_info += "contact loss: {:.3f} | ".format(loss_contact.item())

        mask = pred_p_full < eps
        masked_value = pred_p_full[mask]
        if len(masked_value) > 0:
            loss_pene = w_pene * (-masked_value.sum()) / batch_size
            loss += loss_pene
            loss_info += "pene loss: {:.3f} | ".format(loss_pene.item())

        _, frames = mano_layer(torch.cat((global_pose, mano_pose), dim=1), th_betas=mano_shape, th_trans=mano_trans)
        uv_pred = compute_uv(frames, obj_verts, obj_partition)
        uv_loss = w_uv * compute_uv_loss(uv_pred, obj_uv, weight=1+obj_cmap)
        loss += uv_loss
        loss_info += "uv loss: {:.3f} | ".format(uv_loss.item())

        pose_reg_loss = w_pose_reg * (mano_pose ** 2).sum() / batch_size
        loss += pose_reg_loss
        loss_info += "pose reg loss: {:.3f} | ".format(pose_reg_loss.item())

        shape_reg_loss = w_shape_reg * (mano_shape ** 2).sum() / batch_size
        loss += shape_reg_loss
        loss_info += "shape reg loss: {:.3f}".format(shape_reg_loss.item())

        pose_optimizer.zero_grad()
        loss.backward()
        pose_optimizer.step()
        print("iter {} | ".format(it) + loss_info)

    return global_pose, mano_pose, mano_shape, mano_trans