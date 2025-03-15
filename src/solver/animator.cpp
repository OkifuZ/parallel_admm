#include "solver/animator.h"
#include "mutils/exception_handle.h"
#include <tbb/parallel_for.h>

void ScriptAnimator::reset() {
    this->m_curr_frame = -1;
}

void ScriptAnimator::read_rigid_animate(int mesh_id, const std::string& npz_path) {
    // load data
    cnpy::NpyArray pos;
    try { pos = cnpy::npz_load(npz_path, "pos"); }
    catch (const std::exception& e) { ADU::make_exception("attach_animate2mesh npz file read pos failed"); }

    cnpy::NpyArray quat;
    try { quat = cnpy::npz_load(npz_path, "quat"); }
    catch (const std::exception& e) { ADU::make_exception("attach_animate2mesh npz file read quat failed"); }

    if (pos.shape[1] != 3) ADU::make_exception("attach_animate2mesh npz pos shape wrong");
    if (quat.shape[1] != 4) ADU::make_exception("attach_animate2mesh npz quat shape wrong");
    if (pos.shape[0] != pos.shape[0]) ADU::make_exception("attach_animate2mesh npz pos&quat shape not match");

    // convert 
    ScriptedData script_data;
    script_data.pos.resize(pos.shape[0]);
    script_data.quat.resize(quat.shape[0]);
    auto pos_data = pos.data<ADU::Real>();
    auto quat_data = quat.data<ADU::Real>();
    script_data.frame_nu = pos.shape[0];
    for (int fn = 0; fn < script_data.frame_nu; fn++) {
        script_data.pos[fn] = ADU::Vecf_3{ pos_data[fn * 3 + 0], pos_data[fn * 3 + 1] , pos_data[fn * 3 + 2] };
        script_data.quat[fn] = ADU::Quatf{ quat_data[fn * 4 + 3], quat_data[fn * 4 + 0] , quat_data[fn * 4 + 1] , quat_data[fn * 4 + 2] };
    }

    script_data.type = SCR_TYPE::RIGID;
    // attach mesh
    m_script_data.push_back(script_data);
    m_mesh_id_2_script_idx[mesh_id] = m_script_data.size() - 1;
}

void ScriptAnimator::read_point_animate(int mesh_id, const std::string& npz_path) {
    cnpy::NpyArray points;
    try { points = cnpy::npz_load(npz_path, "points"); }
    catch (const std::exception& e) { ADU::make_exception("attach_animate2mesh npz file read points failed"); }
    
    if (points.shape.size() != 3 || points.shape[2] != 3) ADU::make_exception("attach_animate2mesh npz file points not right");

    ScriptedData script_data;
    int frame_size = points.shape[1] * points.shape[2];
    ADU::Matf_X3 tmp_data(points.shape[1], points.shape[2]);
    for (int fn = 0; fn < points.shape[0]; fn++) {
        Eigen::Map<ADU::Matd_X3> tmp_data_map(&(points.data<double>()[frame_size * fn]), points.shape[1], points.shape[2]);
        tmp_data = tmp_data_map.cast<ADU::Real>();
        //Eigen::Map<ADU::Matf_X3> tmp_data_map(&(points.data<ADU::Real>()[frame_size * fn]), points.shape[1], points.shape[2]);
        script_data.points_data.push_back(tmp_data);
    }

    std::cout << "points data shape:\n";
    std::cout << points.shape[0] << "  " << points.shape[1] << "  " << points.shape[2] << "\n";

    script_data.frame_nu = points.shape[0];
    script_data.type = SCR_TYPE::POINTS;
    m_script_data.push_back(script_data);
    m_mesh_id_2_script_idx[mesh_id] = m_script_data.size() - 1;
}


void ScriptAnimator::attach_animate2mesh(int mesh_id, const std::string& npz_path) {

    if (!mesh) ADU::make_exception("attach_animate2mesh empty mesh");
    if (mesh_id < 0 || mesh_id > mesh->mesh_list.size()) ADU::make_exception("attach_animate2mesh invalid mesh_id");
    if (!mesh->mesh_list[mesh_id]->is_static) ADU::make_exception("attach_animate2mesh not a static mesh!");

    // TODO: make this better
    if (npz_path.find("sequence") == std::string::npos) {
        read_rigid_animate(mesh_id, npz_path);
    }
    else {
        read_point_animate(mesh_id, npz_path);
    }
}

void ScriptAnimator::animate_rigid(int mesh_id, const ADU::Matf_X3& verts_prev, ADU::Matf_X3& verts, ADU::Matf_X3& velocity, ADU::Real dt) {

    const auto& script_data = m_script_data[m_mesh_id_2_script_idx[mesh_id]];
    const auto& mesh_obj = mesh->mesh_list[mesh_id];

    m_curr_frame += 1;

    if (m_curr_frame >= int(script_data.frame_nu)) {
        m_curr_frame = int(script_data.frame_nu);
        tbb::parallel_for(tbb::blocked_range<size_t>(mesh_obj->start_vertIdx, mesh_obj->end_vertIdx), [&](const tbb::blocked_range<size_t>& r) {
            for (int vi = r.begin(); vi < r.end(); vi++) {
                const auto& trans = script_data.pos[m_curr_frame - 1];
                const auto& rotq = script_data.quat[m_curr_frame - 1];
                verts.row(vi) = trans + rotq * mesh->verts.row(vi).transpose();
                velocity.row(vi).setZero();
            }
            });
        /*for (int vi = mesh_obj->start_vertIdx; vi < mesh_obj->end_vertIdx; vi++) {
            const auto& trans = script_data.pos[m_curr_frame - 1];
            const auto& rotq = script_data.quat[m_curr_frame - 1];
            verts.row(vi) = trans + rotq * mesh->verts.row(vi).transpose();
            velocity.row(vi).setZero();
        }*/
        return;
    }

    if (script_data.frame_nu < 3) ADU::make_exception("animate too short script!");

    /*ADU::Vecf_3 v;
    if (m_curr_frame == script_data.frame_nu - 1) v = (script_data.pos[m_curr_frame] - script_data.pos[m_curr_frame - 1]) / dt;
    else if (m_curr_frame == 0) v = (script_data.pos[m_curr_frame + 1] - script_data.pos[m_curr_frame]) / dt;
    else v = (script_data.pos[m_curr_frame + 1] - script_data.pos[m_curr_frame - 1]) / 2.0_r / dt;*/
    tbb::parallel_for(tbb::blocked_range<size_t>(mesh_obj->start_vertIdx, mesh_obj->end_vertIdx), [&](const tbb::blocked_range<size_t>& r) {
        for (int vi = r.begin(); vi < r.end(); vi++) {
            const auto& trans = script_data.pos[m_curr_frame - 1];
            const auto& rotq = script_data.quat[m_curr_frame - 1];
            verts.row(vi) = trans + rotq * mesh->verts.row(vi).transpose();
            velocity.row(vi) = (verts.row(vi) - verts_prev.row(vi)) / dt;
        }
        });
    //for (int vi = mesh_obj->start_vertIdx; vi < mesh_obj->end_vertIdx; vi++) {
    //    const auto& trans = script_data.pos[m_curr_frame - 1];
    //    const auto& rotq = script_data.quat[m_curr_frame - 1];
    //    verts.row(vi) = trans + rotq * mesh->verts.row(vi).transpose();
    //    //velocity.row(vi) = v;
    //    velocity.row(vi) = (verts.row(vi) - verts_prev.row(vi)) / dt;
    //}
}

void ScriptAnimator::animate_points(int mesh_id, const ADU::Matf_X3& verts_prev, ADU::Matf_X3& verts, ADU::Matf_X3& velocity, ADU::Real dt) {
    const auto& script_data = m_script_data[m_mesh_id_2_script_idx[mesh_id]];
    const auto& mesh_obj = mesh->mesh_list[mesh_id];

    if (m_curr_frame >= int(script_data.frame_nu)) {
        m_curr_frame = int(script_data.frame_nu);
        
        const auto& trans = mesh_obj->m_translate;
        const auto& rotq = mesh_obj->m_rotate;
        const auto scale = mesh_obj->m_scale;
        const auto& curr_points = script_data.points_data[m_curr_frame - 1];
        tbb::parallel_for(tbb::blocked_range<size_t>(mesh_obj->start_vertIdx, mesh_obj->end_vertIdx), [&](const tbb::blocked_range<size_t>& r) {
            for (int vi = r.begin(); vi < r.end(); vi++) {
                verts.row(vi) = trans + rotq * (scale * curr_points.row(vi - mesh_obj->start_vertIdx).transpose());
                velocity.row(vi).setZero();
            }
            });

        /*verts.block(mesh_obj->start_vertIdx, 0, mesh_obj->end_vertIdx - mesh_obj->start_vertIdx, 3) = script_data.points_data[m_curr_frame - 1];
        velocity.block(mesh_obj->start_vertIdx, 0, mesh_obj->end_vertIdx - mesh_obj->start_vertIdx, 3) =
            (verts.block(mesh_obj->start_vertIdx, 0, mesh_obj->end_vertIdx - mesh_obj->start_vertIdx, 3) -
                verts_prev.block(mesh_obj->start_vertIdx, 0, mesh_obj->end_vertIdx - mesh_obj->start_vertIdx, 3)) / dt;*/

        return;
    }

    const auto& trans = mesh_obj->m_translate;
    const auto& rotq = mesh_obj->m_rotate;
    const auto scale = mesh_obj->m_scale;
    const auto& curr_points = script_data.points_data[m_curr_frame];
    //std::cout << "curr points shape: " << curr_points.rows() << ", " << curr_points.cols() << "\n";
    //std::cout << mesh_obj->start_vertIdx << ", " << mesh_obj->end_vertIdx << "\n";
    
    tbb::parallel_for(tbb::blocked_range<size_t>(mesh_obj->start_vertIdx, mesh_obj->end_vertIdx), [&](const tbb::blocked_range<size_t>& r) {
        for (int vi = r.begin(); vi < r.end(); vi++) {
            verts.row(vi) = trans + rotq * (scale * curr_points.row(vi - mesh_obj->start_vertIdx).transpose());
            //verts.row(vi) = (scale * curr_points.row(vi - mesh_obj->start_vertIdx).transpose());
            //verts.row(vi) = trans + rotq * (scale * curr_points.row(0).transpose());
            velocity.row(vi) = (verts.row(vi) - verts_prev.row(vi)) / dt;
        }
        });

    /*verts.block(mesh_obj->start_vertIdx, 0, mesh_obj->end_vertIdx - mesh_obj->start_vertIdx, 3) = script_data.points_data[m_curr_frame];
    velocity.block(mesh_obj->start_vertIdx, 0, mesh_obj->end_vertIdx - mesh_obj->start_vertIdx, 3) =
        (verts.block(mesh_obj->start_vertIdx, 0, mesh_obj->end_vertIdx - mesh_obj->start_vertIdx, 3) -
            verts_prev.block(mesh_obj->start_vertIdx, 0, mesh_obj->end_vertIdx - mesh_obj->start_vertIdx, 3)) / dt;*/
}

void ScriptAnimator::animate(int mesh_id, const ADU::Matf_X3& verts_prev, ADU::Matf_X3& verts, ADU::Matf_X3& velocity, ADU::Real dt) {
    // TODO: rotate
    using namespace ADU;
    if (!mesh) ADU::make_exception("animate empty mesh");
    if (mesh_id < 0 || mesh_id > mesh->mesh_list.size()) ADU::make_exception("animate invalid mesh_id");
    if (!mesh->mesh_list[mesh_id]->is_static) ADU::make_exception("animate not a static mesh!");
    if (m_mesh_id_2_script_idx.count(mesh_id) == 0) ADU::make_exception("animate no attached script");

    const auto& script_data = m_script_data[m_mesh_id_2_script_idx[mesh_id]];

    if (script_data.type == SCR_TYPE::RIGID) {
        animate_rigid(mesh_id, verts_prev, verts, velocity, dt);
    }
    else {
        animate_points(mesh_id, verts_prev, verts, velocity, dt);
    }
}


void ScriptAnimator::animate_all(const ADU::Matf_X3& verts_prev, ADU::Matf_X3& verts, ADU::Matf_X3& velocity, ADU::Real dt) {
    m_curr_frame += 1;

    for (auto& miter : m_mesh_id_2_script_idx) {
        animate(miter.first, verts_prev, verts, velocity, dt);
    }
}