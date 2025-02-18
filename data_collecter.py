import time
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
import pandas as pd
import torch
from collections import deque
from actuator_net import ActuatorNet


class DataCollector:
    def __init__(self, model_path, save_dir):
        self.m = mujoco.MjModel.from_xml_path(model_path)
        self.d = mujoco.MjData(self.m)
        self.base_save_dir = Path(save_dir)
        self.base_save_dir.mkdir(exist_ok=True)
        
        # Set up joint info and history buffers
        self._setup_joint_indices()
        self._init_joint_buffers()

        # Current state tracking
        self._current_state = {}

        # Control patterns
        self.control_patterns = {
            'inverse_kinematics': self._inverse_kinematics_control,
        }

        # Initial trajectory parameters
        self.init_amp = 1
        self.init_freq = 1

        # Trajectory parameters
        self.sinusoidal_trajectory_params_last_updated_at = 0
        self.curr_amp_x = self.init_amp
        self.curr_amp_z = self.init_amp
        self.curr_freq = self.init_freq

        # Geom IDs
        self.floor_geom_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.trunk_geom_ids = [] 
        self.left_foot_geom_ids = []
        self.right_foot_geom_ids = []
        for geom_id in range(self.m.ngeom):
            body_id = self.m.geom_bodyid[geom_id]
            body_name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name in ["trunk"]:
                self.trunk_geom_ids.append(geom_id)
            elif body_name == "L_toe":
                self.left_foot_geom_ids.append(geom_id)
            elif body_name == "R_toe":
                self.right_foot_geom_ids.append(geom_id)

    def _init_joint_buffers(self):
        """Initialize history buffers for each joint"""
        self.joint_buffers = {}
        for j_name in self.joint_info.keys():
            self.joint_buffers[j_name] = deque(maxlen=3)

    def _setup_joint_indices(self):
        """Map joints to their MuJoCo indices"""
        joint_names = [
            "L_hip_joint", "L_hip2_joint", "L_thigh_joint", "L_calf_joint", "L_toe_joint",
            "R_hip_joint", "R_hip2_joint", "R_thigh_joint", "R_calf_joint", "R_toe_joint"
        ]
        self.joint_info = {}
        for j_name in joint_names:
            jnt_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, j_name)
            act_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, j_name)
            self.joint_info[j_name] = {
                'joint_id': jnt_id,
                'act_id': act_id,
                'qpos_adr': self.m.jnt_qposadr[jnt_id],
                'qvel_adr': self.m.jnt_dofadr[jnt_id],
            }

    def _track_positions(self, j_name, pos_err, vel):
        """Store joint state in tracking dict"""
        self._current_state[f'{j_name}_pos_err'] = pos_err
        self._current_state[f'{j_name}_vel'] = vel

    def _get_torque_for_joint(self, j_name, sensor_data):
        """Extract torque value for a joint"""
        parts = j_name.split('_')
        # sensor_name = f"{parts[0]}_{parts[1]}_torque"
        sensor_name = f"{parts[0]}_{parts[1]}_actuatorfrc"
        print(sensor_data.get(sensor_name), sensor_name)
        return sensor_data.get(sensor_name)

    def _inverse_kinematics_control(self, t):
        """Inverse kinematics foot trajectory tracking"""
        self._current_state = {}
        kp = 5
        kd = 4
        alpha = 0.5

        for side in ['L', 'R']:
            foot_site_name = f"{side}_toe_sensor"
            site_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, foot_site_name)
            current_pos = self.d.site(site_id).xpos.copy()

            # Update trajectory parameters periodically
            updateFrequency = 5
            tRounded = round(t)
            if tRounded % updateFrequency == 0 and tRounded != self.sinusoidal_trajectory_params_last_updated_at:
                self.curr_amp_x = self.init_amp * np.random.uniform(1, 1.3)
                self.curr_amp_z = self.init_amp * np.random.uniform(1, 1.3)
                self.curr_freq = self.init_freq * np.random.uniform(1, 25)/12.5
                self.sinusoidal_trajectory_params_last_updated_at = tRounded

            # Generate desired positions
            phase_offset = 0.0 if side == 'L' else np.pi
            x = self.curr_amp_x * np.sin(2 * np.pi * self.curr_freq * t + phase_offset)
            z = -0.85 + self.curr_amp_z * np.sin(2 * np.pi * self.curr_freq * t + phase_offset)
            desired_pos = np.array([x, current_pos[1], z])

            # Jacobian-based IK
            error = desired_pos - current_pos
            Jt = np.zeros((3, self.m.nv))
            mujoco.mj_jacSite(self.m, self.d, Jt, None, site_id)
            
            leg_joints = [
                f"{side}_hip_joint", f"{side}_hip2_joint",
                f"{side}_thigh_joint", f"{side}_calf_joint", f"{side}_toe_joint"
            ]
            qvel_adrs = [self.joint_info[j_name]['qvel_adr'] for j_name in leg_joints]
            J_pinv = np.linalg.pinv(Jt)
            delta_theta = alpha * np.dot(J_pinv, error)
            
            # PD control for each joint
            for i, j_name in enumerate(leg_joints):
                info = self.joint_info[j_name]
                current_angle = self.d.qpos[info['qpos_adr']]
                desired_angle = current_angle + delta_theta[i]
                joint_range = self.m.jnt_range[info['joint_id']]
                desired_angle = np.clip(desired_angle, joint_range[0], joint_range[1])
                actual_vel = self.d.qvel[info['qvel_adr']]
                pos_err = desired_angle - current_angle
                torque = kp * pos_err + kd * (-actual_vel)
                self.d.ctrl[info['act_id']] = torque
                self._track_positions(j_name, pos_err, actual_vel)

    def _get_sensor_data(self):
        """Collect all relevant sensor data"""
        sensor_data = {}
        joints = ['hip', 'hip2', 'thigh', 'calf', 'toe']
        sides = ['L', 'R']
        for side in sides:
            for joint in joints:
                sensor_name = f'{side}_{joint}_actuatorfrc'
                sensor_data[sensor_name] = self.d.sensor(sensor_name).data[0]
        sensor_data.update(self._current_state)
        return sensor_data

    def _run_simulation(self, control_callback, sample_callback, duration, visualize, drop_biped, start_condition, termination_condition):
        """Generic simulation loop to reduce code duplication."""

        # Randomize initial orientation
        if drop_biped:
            self._set_initial_orientation()

        next_sample_time = 0.0
        viewer = None
        if visualize:
            viewer = mujoco.viewer.launch_passive(self.m, self.d)
        
        start_condition_met = False
        termination_condition_met = False
        sim_time = 0.0

        try:
            while sim_time < duration and not termination_condition_met:
                control_callback(sim_time)
                mujoco.mj_step(self.m, self.d)
                sim_time = self.d.time

                # Check start condition (e.g., foot contact)
                if not start_condition_met:
                    start_condition_met = start_condition()
                    if start_condition_met:
                        next_sample_time = sim_time
                        print(f"Start condition met at {sim_time:.3f}s")

                # Check termination condition (e.g., trunk contact)
                termination_condition_met = termination_condition()

                # Sample data if conditions are met
                if sim_time >= next_sample_time and start_condition_met:
                    sample_callback()
                    next_sample_time += 0.0025

                if viewer:
                    viewer.sync()
        finally:
            if viewer:
                viewer.close()
                time.sleep(0.5) # Allow time for viewer to close

    def collect_data(self, control_type, save_dir, duration, visualize, drop_biped, start_on_foot_contact, end_on_trunk_contact):
        """Main data collection routine (refactored to use _run_simulation)"""
        joint_data = []
        self._init_joint_buffers()
        
        # Define conditions
        def start_condition():
            if not start_on_foot_contact:
                return True  # Immediate start
            for i in range(self.d.ncon):
                contact = self.d.contact[i]
                geom1, geom2 = contact.geom1, contact.geom2
                if ((geom1 == self.floor_geom_id and (geom2 in self.left_foot_geom_ids or geom2 in self.right_foot_geom_ids)) or 
                    (geom2 == self.floor_geom_id and (geom1 in self.left_foot_geom_ids or geom1 in self.right_foot_geom_ids))):
                    return True
            return False

        def termination_condition():
            if end_on_trunk_contact:
                for i in range(self.d.ncon):
                    contact = self.d.contact[i]
                    geom1, geom2 = contact.geom1, contact.geom2
                    if ((geom1 == self.floor_geom_id and geom2 in self.trunk_geom_ids) or 
                        (geom2 == self.floor_geom_id and geom1 in self.trunk_geom_ids)):
                        print(f"Trunk contact detected at {self.d.time:.3f}s")
                        return True
            return False

        # Define sampling callback
        def sample_callback():
            sensor_data = self._get_sensor_data()
            for j_name in self.joint_info.keys():
                pos_err = sensor_data.get(f'{j_name}_pos_err', 0.0)
                vel = sensor_data.get(f'{j_name}_vel', 0.0)
                torque = self._get_torque_for_joint(j_name, sensor_data)
                self.joint_buffers[j_name].append((pos_err, vel, torque))
            
            if all(len(buf) >= 3 for buf in self.joint_buffers.values()):
                input_features = []
                target_torques = []
                for j_name in self.joint_info.keys():
                    buf = self.joint_buffers[j_name]
                    input_features.extend([buf[2][0], buf[2][1], buf[1][0], buf[1][1], buf[0][0], buf[0][1]])
                    target_torques.append(buf[2][2])
                joint_data.append((input_features, target_torques))

        # Run simulation
        self._run_simulation(
            control_callback=lambda t: self.control_patterns[control_type](t),
            sample_callback=sample_callback,
            duration=duration,
            visualize=visualize,
            drop_biped=drop_biped,
            start_condition=start_condition,
            termination_condition=termination_condition
        )

        # Save data (existing code remains unchanged)
        if save_dir:
            save_dir.mkdir(exist_ok=True, parents=True)
            if joint_data:
                inputs = torch.tensor([d[0] for d in joint_data], dtype=torch.float32)
                targets = torch.tensor([d[1] for d in joint_data], dtype=torch.float32)
                file_path = save_dir / "concatenated_data.pt"
                if file_path.exists():
                    existing = torch.load(file_path, weights_only=True)
                    inputs = torch.cat([existing['inputs'], inputs], dim=0)
                    targets = torch.cat([existing['targets'], targets], dim=0)
                torch.save({'inputs': inputs, 'targets': targets}, file_path)


    def evaluate_model(self, model_path, control_type, duration, visualize, drop_biped, start_on_foot_contact, end_on_trunk_contact):
        import matplotlib
        matplotlib.use('Agg')  # Set backend to non-interactive
        import matplotlib.pyplot as plt
        import pickle
    
        """Evaluate model using DRY simulation loop"""
        model = ActuatorNet(input_dim=60, output_dim=10)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        torque_data = {j_name: {'predicted': [], 'desired': []} for j_name in self.joint_info.keys()}
        self._init_joint_buffers()

        # Define conditions
        def start_condition():
            if not start_on_foot_contact:
                return True  # Immediate start
            for i in range(self.d.ncon):
                contact = self.d.contact[i]
                geom1, geom2 = contact.geom1, contact.geom2
                if ((geom1 == self.floor_geom_id and (geom2 in self.left_foot_geom_ids or geom2 in self.right_foot_geom_ids)) or 
                    (geom2 == self.floor_geom_id and (geom1 in self.left_foot_geom_ids or geom1 in self.right_foot_geom_ids))):
                    return True
            return False

        def termination_condition():
            if end_on_trunk_contact:
                for i in range(self.d.ncon):
                    contact = self.d.contact[i]
                    geom1, geom2 = contact.geom1, contact.geom2
                    if ((geom1 == self.floor_geom_id and geom2 in self.trunk_geom_ids) or 
                        (geom2 == self.floor_geom_id and geom1 in self.trunk_geom_ids)):
                        print(f"Trunk contact detected at {self.d.time:.3f}s")
                        return True
            return False

        # Define sampling callback
        def sample_callback():
            sensor_data = self._get_sensor_data()
            for j_name in self.joint_info.keys():
                pos_err = sensor_data.get(f'{j_name}_pos_err', 0.0)
                vel = sensor_data.get(f'{j_name}_vel', 0.0)
                torque = self._get_torque_for_joint(j_name, sensor_data)
                self.joint_buffers[j_name].append((pos_err, vel, torque))
            
            if all(len(buf) >= 3 for buf in self.joint_buffers.values()):
                input_features = []
                for j_name in self.joint_info.keys():
                    buf = self.joint_buffers[j_name]
                    input_features.extend([
                        buf[2][0], buf[2][1], # Current pos error and velocity
                        buf[1][0], buf[1][1], # Previous pos error and velocity
                        buf[0][0], buf[0][1]  # Pre-previous pos error and velocity
                    ])
                input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    predicted = model(input_tensor).squeeze().tolist()
                for idx, j_name in enumerate(self.joint_info.keys()):
                    torque_data[j_name]['predicted'].append(predicted[idx])
                    torque_data[j_name]['desired'].append(self.joint_buffers[j_name][2][2])

        # Run simulation
        self._run_simulation(
            control_callback=lambda t: self.control_patterns[control_type](t),
            sample_callback=sample_callback,
            duration=duration,
            visualize=visualize,
            drop_biped=drop_biped,
            start_condition=start_condition,
            termination_condition=termination_condition
        )

        # Plot results
        torque_comparison_dir = self.base_save_dir / 'torque_comparisons'
        torque_comparison_dir.mkdir(exist_ok=True)
        for j_name in self.joint_info.keys():
            desired = torque_data[j_name]['desired']
            predicted = torque_data[j_name]['predicted']
            if desired and predicted:

                # Save the figure
                # with open('figure.pickle', 'wb') as f:
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(desired, label='Desired')
                ax.plot(predicted, label='Predicted')
                ax.set_title(f'Torque Comparison: {j_name}')
                ax.set_xlabel('Time Steps')
                ax.set_ylabel('Torque (Nm)')
                ax.legend()
                
                # Save the figure to a pickle file
                pickle_file = torque_comparison_dir / f'torque_{j_name}.pickle'
                with open(pickle_file, 'wb') as f:
                    pickle.dump(fig, f)
                
                # Optionally, save a static image for viewing later
                fig.savefig(torque_comparison_dir / f'torque_{j_name}.png')
                plt.close(fig)

        return torque_data

    def _euler_to_quaternion(self, roll, pitch, yaw):
        """Convert Euler angles (radians) to a unit quaternion."""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        return [qw, qx, qy, qz]
    

    def _set_initial_orientation(self):
        # Find root joint's qpos address
        self.root_joint_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "root")

        if self.root_joint_id == -1:
            raise ValueError("Root joint not found.")
        self.root_qpos_adr = self.m.jnt_qposadr[self.root_joint_id]

        # Find weld constraint ID
        self.world_weld_eq_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_EQUALITY, "world_root")

        # Disable weld constraint to allow free motion
        if self.world_weld_eq_id != -1:
            self.d.eq_active[self.world_weld_eq_id] = 0
        
        # Set random initial orientation (small perturbations)
        roll = np.random.uniform(-np.pi/12, np.pi/12)
        pitch = np.random.uniform(-np.pi/12, np.pi/12)
        yaw = np.random.uniform(-np.pi/12, np.pi/12)
        quat = self._euler_to_quaternion(roll, pitch, yaw)
        
        # Update position and orientation in root joint's qpos
        start_idx = self.root_qpos_adr
        drop_height = np.random.uniform(0.5, 1)
        self.d.qpos[start_idx:start_idx+3] = [0, 0, drop_height]  # x, y, z
        self.d.qpos[start_idx+3:start_idx+7] = quat  # quaternion


    def run_data_collection_suite(self, trials_per_pattern, duration, visualize, drop_biped, start_on_foot_contact, end_on_trunk_contact):
        """Orchestrate data collection for all control patterns."""
        for control_type in self.control_patterns.keys():
            method_dir = self.base_save_dir / control_type

            for trial in range(trials_per_pattern):

                print(f"\nCollecting {control_type} - Trial {trial+1}/{trials_per_pattern}")

                mujoco.mj_resetData(self.m, self.d)

                self.collect_data(
                    control_type=control_type,
                    save_dir=method_dir,
                    duration=duration,
                    visualize=visualize,
                    drop_biped=drop_biped,
                    start_on_foot_contact=start_on_foot_contact,
                    end_on_trunk_contact=end_on_trunk_contact
                )



if __name__ == "__main__":
    
    collector = DataCollector(
        model_path='./biped_simple_final.xml', 
        save_dir='./collected_data',
    )

    collect_data = True

    if collect_data:

        # On the shelf (in the air)
        collector.run_data_collection_suite(
            trials_per_pattern=1, 
            duration=500, 
            visualize=True,
            drop_biped=False, 
            start_on_foot_contact=False, # Do not change, fixed for shelf
            end_on_trunk_contact=False # Do not change, fixed for shelf
        )

        # Drop
        collector.run_data_collection_suite(
            trials_per_pattern=10, 
            duration=5, 
            visualize=True,
            drop_biped=True, 
            start_on_foot_contact=True, # Do not change, fixed for drop
            end_on_trunk_contact=True # Do not change, fixed for drop
        )

    else:
        collector.evaluate_model(
            model_path='./trained_models/concatenated/concatenated_model.pth',
            control_type='inverse_kinematics',
            duration=5,
            visualize=True,
            drop_biped=True, 
            start_on_foot_contact=True, 
            end_on_trunk_contact=False
        )